"""Real SMPL engineering validation of the integrated A delivery (no fake env)."""
import argparse
import hashlib
import importlib.metadata
import importlib
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from a_integration.main import run_live, run_control_cycle, B_CODE_ROOT
from a_integration.agents.monitoring_agent import MonitoringAgent
sys.path.insert(0, str(B_CODE_ROOT))
from integration_abd_1103 import EnvironmentAdapter, ExecutionAgent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True, help='A new directory; never overwrite evidence')
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for seed in [1102, 1103]:
        result = run_live(goal='执行一次控制任务', seed=seed, output_dir=out / f'normal_{seed}')
        summary = result['summary']
        assert summary['run_status'] == 'completed' and summary['step_calls'] == 1
        assert summary['completed_subtasks'] == 6 and not summary['d_demo_mode']
        execution = next(s['execution_result'] for s in result['stage_results'] if s['task_type'] == 'execute_approved_action')
        rows.append({'scenario': 'normal', 'seed': seed, 'status': summary['run_status'],
                     'step_calls': summary['step_calls'], 'reward': execution['reward'],
                     'efactor': execution['info'].get('efactor')})
    for scenario, goal in [('reject', '执行一次控制任务'), ('inspect', '检查状态')]:
        case = out / scenario
        case.mkdir()
        with EnvironmentAdapter() as adapter:
            executor = ExecutionAgent(adapter, log_path=case / 'B_events.jsonl')
            executor.reset(seed=1101)
            monitor = MonitoringAgent(action_space=adapter.action_space)
            kwargs = {}
            if scenario == 'reject':
                kwargs['action_generator'] = lambda obs, space, attempt: space.high.astype(float) + 1.
            result = run_control_cycle(goal=goal, executor=executor, monitor=monitor,
                                       event_path=case / 'A_events.jsonl', **kwargs)
            assert adapter.step_calls == 0
            assert result['summary']['run_status'] == ('completed' if scenario == 'inspect' else 'incomplete_or_failed')
            monitor.export_logs(case / 'D_reviews.jsonl')
            (case / 'A_summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
            rows.append({'scenario': scenario, 'seed': 1101,
                         'status': result['summary']['run_status'], 'step_calls': adapter.step_calls})
    code_paths = ['main.py', 'a_integration/main.py', 'a_integration/decision_agent.py',
                  'a_integration/agents/monitoring_agent.py',
                  '执行智能体/code/integration_abd_1103/execution_agent.py',
                  '执行智能体/code/integration_abd_1103/environment_adapter.py',
                  '执行智能体/code/vendor/smpl/smpl/envs/atropineenv.py',
                  '执行智能体/code/vendor/smpl/smpl/envs/models/atropine_process.py']
    report = {'environment': 'real SMPL.AtropineEnvGym', 'python': platform.python_version(),
              'platform': platform.platform(), 'packages': {m: getattr(importlib.import_module(m), '__version__', importlib.metadata.version(m))
              for m in ['numpy','gym','requests','casadi','torch','torchvision','mzutils']},
              'results': rows, 'source_sha256': {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in code_paths},
              'limitations': 'Single-step midpoint baseline; no mAbEnv, optimization, industrial L3 or R03 validation. Seed 1101 CLI run is a separate evidence directory.'}
    (out / 'validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
