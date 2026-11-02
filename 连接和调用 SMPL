"""Teaching example: keep all SMPL-specific details in one adapter."""
import functools
import http.server
import os
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.mpl-cache'))
sys.path.insert(0, str(ROOT / 'vendor' / 'smpl'))

import numpy as np
from smpl.envs.atropineenv import AtropineEnvGym


class EnvironmentAdapter:
    def __init__(self):
        config = ROOT / 'vendor/smpl/smpl/configdata/atropineenv'
        class QuietHandler(http.server.SimpleHTTPRequestHandler):
            def log_message(self, *args):
                pass
        handler = functools.partial(QuietHandler, directory=str(config))
        self._server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        self.env = None
        self._ready = False
        try:
            self.env = AtropineEnvGym(
                normalize=False,
                x0_loc=np.loadtxt(config / 'x0.txt'),
                z0_loc=np.loadtxt(config / 'z0.txt'),
                model_loc=f'http://127.0.0.1:{self._server.server_port}/model.npy',
            )
        except Exception:
            self.close()
            raise

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    def reset(self, seed=1101):
        # This SMPL version samples with NumPy's global random generator.
        np.random.seed(seed)
        self._ready = False
        result = self.env.reset()
        obs = np.asarray(result)
        if obs.shape != self.observation_space.shape or not np.isfinite(obs).all():
            raise RuntimeError('Invalid reset observation')
        self._ready = True
        return obs

    def step(self, action):
        if not self._ready:
            raise RuntimeError('Call reset before executing another action')
        action = np.asarray(action, dtype=self.action_space.dtype)
        if not self.action_space.contains(action):
            raise ValueError('Action has invalid shape, nonfinite values, or exceeds bounds')
        previous_step = self.env.step_count
        obs, reward, done, info = self.env.step(action)
        obs = np.asarray(obs)
        # Upstream step catches exceptions; successful return alone is insufficient.
        success = (self.env.step_count == previous_step + 1
                   and float(reward) != float(self.env.error_reward)
                   and obs.shape == self.observation_space.shape
                   and np.isfinite(obs).all() and np.isfinite(float(reward)))
        self._ready = bool(success and not done)
        return {
            'success': bool(success),
            'observation': obs,
            'action': action,
            'reward': float(reward),
            'done': bool(done),
            'info': info,
            'step_count': int(self.env.step_count),
            'failure_reason': None if success else 'SMPL error reward, invalid state, or step did not advance',
        }

    def close(self):
        try:
            if self.env is not None:
                self.env.close()
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
