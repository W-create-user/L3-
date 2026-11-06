"""Inject a configured SMPL/Gym environment; constructor stays outside agents."""
from pathlib import Path
import numpy as np
from .environment_adapter import EnvironmentAdapter, to_plain


class GymEnvironmentAdapter(EnvironmentAdapter):
    def __init__(self, env, *, environment_name, source_dir):
        self._env = env
        self.environment_name = environment_name
        self.source_dir = str(Path(source_dir).resolve())
        self._version = self._step_calls = self._episode_steps = 0
        self._ready = self._done = False
        self._observation = None

    @property
    def step_count(self):
        return self._episode_steps

    def reset(self, seed=1101):
        self._ready = False
        self._version += 1
        self._observation = None
        # Legacy SMPL reset has no seed keyword. Avoid retrying reset on TypeError.
        import inspect
        if 'seed' in inspect.signature(self._env.reset).parameters:
            result = self._env.reset(seed=seed)
        else:
            np.random.seed(seed)
            result = self._env.reset()
        observation = result[0] if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict) else result
        value = np.asarray(observation)
        if value.shape != self.observation_space.shape or not np.isfinite(value).all():
            raise ValueError('Invalid reset observation')
        self._observation = value.copy()
        self._episode_steps = 0
        self._done = False
        self._ready = True
        return self.get_observation()

    def step(self, action):
        self.require_ready()
        action = self.validate_action(action)
        self._ready = False
        self._version += 1
        self._step_calls += 1
        result = self._env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
        else:
            obs, reward, done, info = result
            truncated = bool(done and info.get('timeout', False))
            terminated = bool(done and not truncated)
        value = np.asarray(obs)
        valid = value.shape == self.observation_space.shape and np.isfinite(value).all()
        success = bool(valid and np.isfinite(reward)
                       and float(reward) != float(getattr(self._env, 'error_reward', -np.inf))
                       and info.get('success', True) is not False)
        self._episode_steps += 1
        self._observation = value.copy() if valid else None
        self._done = bool(terminated or truncated)
        self._ready = success and not self._done
        return {'success': success, 'observation': self.get_observation() if valid else None,
                'action': action.tolist(), 'reward': float(reward) if np.isfinite(reward) else None,
                'done': self._done, 'terminated': bool(terminated), 'truncated': bool(truncated),
                'info': to_plain(info), 'step_count': self.step_count, 'step_calls': self.step_calls,
                'state_version': self.state_version,
                'failure_reason': None if success else 'invalid_or_error_environment_result'}

    def close(self):
        self._ready = False
        self._env.close()
