"""Freeze B01 live Atropine reset/step output and environment provenance."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import traceback
from l3_system.execution.environment_adapter import EnvironmentAdapter


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    report = {'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'python': sys.version, 'platform': platform.platform(),
              'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
              'configuration': {'seed': 1112, 'max_steps': 2, 'normalize': False},
              'success': False}
    for name in ['requirements.txt', 'vendor/smpl/SOURCE_VERSION.json']:
        report.setdefault('input_sha256', {})[name] = hashlib.sha256((root/name).read_bytes()).hexdigest()
    try:
        with EnvironmentAdapter(max_steps=2) as env:
            report['reset'] = env.reset(seed=1112)
            report['observation_shape'] = list(env.observation_space.shape)
            report['action_shape'] = list(env.action_space.shape)
            report['steps'] = []
            while not env.done:
                action = (env.action_space.low + env.action_space.high) / 2
                result = env.step(action)
                report['steps'].append(result)
                if not result['success']:
                    raise RuntimeError(result['failure_reason'])
            report['termination_reason'] = report['steps'][-1]['info']
            report['success'] = True
    except Exception:
        report['traceback'] = traceback.format_exc()
    (args.output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    lock = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
    (args.output/'requirements-lock.txt').write_text(lock, encoding='utf-8')
    return 0 if report['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
