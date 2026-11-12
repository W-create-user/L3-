"""Run the B02 real-model gate, preserving each completed stage and traceback."""
from __future__ import annotations
import argparse
import datetime as dt
import importlib
import importlib.metadata
import json
import platform
from pathlib import Path
try:
    import resource
except ImportError:  # Windows has no resource module
    resource = None
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reset-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(ROOT / 'vendor/smpl'))
    report = {'started_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'python': sys.version, 'platform': platform.platform(),
              'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'backend': 'smpl.envs.mabenv.MAbEnvGym', 'stages': [], 'success': False,
              'configuration': {'normalize': False, 'max_steps': 2, 'seed': 1112,
                                'initial_state_deviation_ratio': 0.0},
              'action_semantics': 'model-scaled action = physical action / uscale; normalize=False is still scaled',
              'sampling_period': {'value': 60, 'unit': 'min', 'source': 'mabenv.py constructor dt_spl'},
              'units': 'per-variable units require B03 verification; no Atropine units reused'}
    def save():
        (args.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    def stage(name, fn):
        start = time.perf_counter()
        entry = {'stage': name}
        report['stages'].append(entry)
        try:
            result = fn()
            entry['success'] = True
            return result
        except Exception:
            entry.update(success=False, traceback=traceback.format_exc())
            raise
        finally:
            entry.update(seconds=time.perf_counter()-start,
                         peak_rss_kib=(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if resource else None))
            save()
    try:
        mpc = stage('import_mpctools', lambda: importlib.import_module('mpctools'))
        report['mpctools'] = {'module_file': mpc.__file__, 'version': getattr(mpc, '__version__', None),
                              'required_symbols': {s: hasattr(mpc, s) for s in ['DiscreteSimulator', 'getCasadiFunc', 'nmpc']},
                              'casadi': importlib.metadata.version('casadi')}
        if not all(report['mpctools']['required_symbols'].values()):
            raise RuntimeError('Wrong mpctools distribution: required CasADi APIs missing')
        mod = stage('import_environment', lambda: importlib.import_module('smpl.envs.mabenv'))
        import numpy as np
        np.random.seed(1112)
        env = stage('initialize', lambda: mod.MAbEnvGym(dataset_dir=str(ROOT / 'vendor/smpl/smpl/configdata/mabenv'),
                    normalize=False, max_steps=2, initial_state_deviation_ratio=0.0))
        obs = stage('reset', env.reset)
        if np.asarray(obs).shape != env.observation_space.shape or not np.isfinite(obs).all():
            raise RuntimeError('Reset returned invalid observations')
        report['spaces'] = {'observation': list(env.observation_space.shape), 'action': list(env.action_space.shape)}
        report['reset'] = np.asarray(obs).tolist()
        report['action_low'] = env.action_space.low.tolist()
        report['action_high'] = env.action_space.high.tolist()
        if not args.reset_only:
            action = np.asarray(env.steady_actions, dtype=env.action_space.dtype)
            if not env.action_space.contains(action):
                raise RuntimeError('Steady action is outside live action space')
            report['requested_action'] = action.tolist()
            obs, reward, done, info = stage('step', lambda: env.step(action))
            report['step'] = {'observation': np.asarray(obs).tolist(), 'reward': float(reward),
                              'done': bool(done), 'info': info, 'step_count': env.step_count}
            if (np.asarray(obs).shape != env.observation_space.shape
                    or not np.isfinite(obs).all() or not np.isfinite(reward)
                    or reward == env.error_reward or env.step_count != 1):
                raise RuntimeError('SMPL step returned an error reward, invalid values or did not advance')
        report['success'] = True
    except Exception:
        report['traceback'] = traceback.format_exc()
        print(report['traceback'], file=sys.stderr)
    finally:
        save()
    return 0 if report['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
