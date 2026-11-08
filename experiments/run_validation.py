"""Run the actual SMPL environment. Test doubles are never used here."""
import argparse
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path

from l3_system.main import run_live


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('outputs/validation'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for seed in (1101, 1102, 1103):
        for scenario in ('normal', 'reject', 'inspect'):
            result = run_live(seed=seed, scenario=scenario,
                              output_dir=args.output/f'{scenario}_{seed}')
            summary = result['summary']
            assert summary['all_C_locks_released']
            assert summary['d_demo_mode'] is False
            if scenario == 'reject':
                assert summary['failure_reason'] == 'safety_rejected'
                assert summary['total_step_calls'] == 0
            else:
                assert summary['run_status'] == 'completed'
                assert summary['total_step_calls'] == (1 if scenario == 'normal' else 0)
            rows.append(summary)
    result = run_live(seed=1101, cycles=5, output_dir=args.output/'continuous_5')
    assert result['summary']['run_status'] == 'completed'
    assert result['summary']['finished_cycles'] == 5
    assert result['summary']['total_step_calls'] == 5
    assert all(x['summary']['all_C_locks_released'] for x in result['cycles'])
    rows.append(result['summary'])
    report = {'python':sys.version, 'platform':platform.platform(),
              'packages':{p: version(p) for p in ('numpy','casadi','gym','scipy','matplotlib','requests','tqdm')},
              'environment':'real SMPL.AtropineEnvGym', 'validated_cases': len(rows),
              'results':rows,
              'limits':'Interface baseline; no claim of process optimization, mAb validation or industrial L3 certification.'}
    (args.output/'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Validated {len(rows)} cases; report: {args.output / "summary.json"}')


if __name__ == '__main__':
    main()
