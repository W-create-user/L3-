"""SMPL details live here; importing this module does not create an environment."""
from __future__ import annotations

import functools
import http.server
import os
from pathlib import Path
import sys
import threading

import numpy as np


def legacy_termination_flags(done, info):
    """SMPL's terminal includes timeouts; expose mutually exclusive derived flags."""
    truncated = bool(done and info.get('timeout', False))
    return bool(done and not truncated), truncated


def to_plain(value):
    """Convert NumPy/CasADi output to ordinary JSON-compatible values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, 'full'):
        return value.full().tolist()
    if isinstance(value, dict):
        return {str(k): to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(v) for v in value]
    return value


class EnvironmentAdapter:
    def __init__(self, source_dir=None, *, max_steps=60):
        if type(max_steps) is not int or (max_steps != -1 and max_steps <= 0):
            raise ValueError('max_steps must be positive or -1 (upstream open-loop option)')
        project_root = Path(__file__).resolve().parents[1]
        source = Path(source_dir or os.environ.get('SMPL_SOURCE_DIR')
                      or project_root / 'vendor' / 'smpl').resolve()
        config = source / 'smpl/configdata/atropineenv'
        for filename in ['x0.txt', 'z0.txt', 'model.npy']:
            if not (config / filename).is_file():
                raise FileNotFoundError(f'Missing {config / filename}; set SMPL_SOURCE_DIR')
        os.environ.setdefault('MPLCONFIGDIR', str(project_root / '.mpl-cache'))
        sys.path.insert(0, str(source))
        from smpl.envs.atropineenv import AtropineEnvGym

        class QuietHandler(http.server.SimpleHTTPRequestHandler):
            def log_message(self, *args):
                pass
        handler = functools.partial(QuietHandler, directory=str(config))
        self._server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        self._env = None
        self._observation = None
        self._ready = False
        self._done = False
        self._version = 0
        self._step_calls = 0
        self.source_dir = str(source)
        try:
            self._env = AtropineEnvGym(
                normalize=False,
                max_steps=max_steps,
                x0_loc=np.loadtxt(config / 'x0.txt'),
                z0_loc=np.loadtxt(config / 'z0.txt'),
                model_loc=f'http://127.0.0.1:{self._server.server_port}/model.npy',
            )
        except Exception:
            self.close()
            raise

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def state_version(self):
        return self._version

    @property
    def step_count(self):
        return int(self._env.step_count)

    @property
    def step_calls(self):
        """Cumulative number of real env.step calls, including across resets."""
        return self._step_calls

    @property
    def done(self):
        return self._done

    def require_ready(self):
        if not self._ready:
            raise RuntimeError('Environment is not ready; reset is required')

    def reset(self, seed=1101):
        self._ready = False
        self._observation = None
        self._version += 1  # invalidate all requests created before any reset
        np.random.seed(seed)
        observation = np.asarray(self._env.reset())
        if (observation.shape != self.observation_space.shape
                or not np.isfinite(observation).all()):
            raise RuntimeError('Invalid reset observation')
        self._observation = observation.copy()
        self._done = False
        self._ready = True
        return self.get_observation()

    def get_observation(self):
        """Read the last returned observation; never reset or advance SMPL."""
        if self._observation is None:
            raise RuntimeError('No observation available; call reset first')
        return self._observation.copy().tolist()

    def validate_action(self, action):
        raw = np.asarray(action)
        space = self.action_space
        if (raw.shape != space.shape or raw.dtype.kind not in 'fiu'
                or not np.isfinite(raw).all()):
            raise ValueError('Action must be a finite numeric vector of the live action-space shape')
        if np.any(raw < space.low) or np.any(raw > space.high):
            raise ValueError('Action exceeds the live physical action bounds')
        converted = raw.astype(space.dtype)
        if not space.contains(converted):
            raise ValueError('Action is not contained in the live action space')
        return converted

    def coerce_candidate(self, action):
        """Format a finite candidate for review; do not hide out-of-range values."""
        raw = np.asarray(action)
        space = self.action_space
        if (raw.shape != space.shape or raw.dtype.kind not in 'fiu'
                or not np.isfinite(raw).all()):
            raise ValueError('Candidate must have the live shape and finite numeric values')
        with np.errstate(over='ignore', invalid='ignore'):
            candidate = raw.astype(space.dtype)
        if not np.isfinite(candidate).all():
            raise ValueError('Candidate cannot be represented in the action-space dtype')
        return candidate

    def step(self, action):
        """Low-level adapter call; A must use ExecutionAgent.execute_action with a review."""
        self.require_ready()
        action = self.validate_action(action)
        before = self.step_count
        self._ready = False
        self._step_calls += 1
        self._version += 1
        obs, reward, done, info = self._env.step(action)
        terminated, truncated = legacy_termination_flags(done, info)
        obs = np.asarray(obs)
        valid_obs = (obs.shape == self.observation_space.shape
                     and np.isfinite(obs).all())
        success = (self.step_count == before + 1 and valid_obs
                   and np.isfinite(float(reward))
                   and float(reward) != float(self._env.error_reward))
        self._observation = obs.copy() if valid_obs else None
        self._done = bool(done)
        self._ready = bool(success and not done)
        return {
            'success': bool(success),
            'observation': self.get_observation() if valid_obs else None,
            'action': action.tolist(),
            'reward': float(reward) if np.isfinite(float(reward)) else None,
            'done': bool(done),
            'terminated': terminated,
            'truncated': truncated,
            'termination_flag_source': 'derived_from_SMPL_done_and_info_timeout',
            'info': to_plain(info),
            'step_count': self.step_count,
            'step_calls': self.step_calls,
            'state_version': self.state_version,
            'failure_reason': None if success else 'SMPL returned an error reward, invalid observation, or did not advance',
        }

    def close(self):
        try:
            if self._env is not None:
                self._env.close()
        finally:
            if self._server is not None:
                self._server.shutdown()
                self._server.server_close()
                self._server = None
            self._ready = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
