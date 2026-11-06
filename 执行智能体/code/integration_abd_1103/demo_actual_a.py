"""Actual supplied A + real SMPL; D is external source or an explicit reference."""
import argparse
from datetime import datetime
import hashlib
import importlib.util
from importlib.metadata import version
import inspect
import json
from pathlib import Path
import sys
import types
from zoneinfo import ZoneInfo

import numpy as np

from . import EnvironmentAdapter, ExecutionAgent
from .a_integration import load_decision_agent, run_a_plan
from .reference_monitor import ReferenceMonitoringAgent


def load_monitor(action_space, monitor_file=None):
    if monitor_file is None:
        return ReferenceMonitoringAgent(action_space), 'B_reference_of_D_API_not_D_source'
    source = Path(monitor_file).resolve()
    if not source.is_file():
        raise FileNotFoundError(f'D monitoring source not found: {source}')
    # Support D's file in an agents/ directory, including simple relative imports.
    package_name = '_d_provided_agents'
    package = types.ModuleType(package_name)
    package.__path__ = [str(source.parent)]
    sys.modules[package_name] = package
    sys.path.insert(0, str(source.parent.parent))
    sys.path.insert(0, str(source.parent))
    spec = importlib.util.spec_from_file_location(f'{package_name}.monitoring_agent', source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.MonitoringAgent(action_space=action_space), 'D_provided_source'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--monitor-file', help='D actual monitoring_agent.py; absent means explicit reference checker')
    parser.add_argument('--a-root', help='Actual A source directory; defaults to the supplied archive source')
    args = parser.parse_args()
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
    output = Path(__file__).parent / 'logs' / stamp
    output.mkdir(parents=True)

    def save(name, value):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                             allow_nan=False), encoding='utf-8')

    checks = []
    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append({'check': name, 'status': 'PASS'})
        print('PASS:', name, flush=True)

    decision, a_root = load_decision_agent(args.a_root)
    with EnvironmentAdapter() as adapter:
        monitor, provenance = load_monitor(adapter.action_space, args.monitor_file)
        monitor_source = Path(inspect.getfile(type(monitor))).resolve()
        manifest = {
            'time': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
            'actual_A_source': str(a_root),
            'monitor_provenance': provenance,
            'monitor_source': str(monitor_source),
            'monitor_source_sha256': hashlib.sha256(monitor_source.read_bytes()).hexdigest(),
            'R03_validation': 'not claimed; D described R03 as not implemented',
            'SMPL_source': adapter.source_dir,
            'normalize': False, 'uss_subtracted': True,
            'observation_shape': list(adapter.observation_space.shape),
            'action_shape': list(adapter.action_space.shape),
            'action_low': adapter.action_space.low.tolist(),
            'action_high': adapter.action_space.high.tolist(),
            'python': sys.version,
            'versions': {p: version(p) for p in ['numpy','gym','casadi','scipy']},
            'A_sha256': {str(p.relative_to(a_root)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in a_root.rglob('*.py')},
            'B_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in Path(__file__).parent.glob('*.py')},
        }
        save('manifest.json', manifest)
        goal = {'goal_type': 'control_cycle', 'description': '完成一次仿真控制任务',
                'priority': 4, 'constraints': {'environment': 'AtropineEnv', 'require_monitoring': True}}
        executor = ExecutionAgent(adapter, log_path=output / 'B_events.jsonl')

        initial = executor.reset(seed=1101)
        before = adapter.step_calls
        normal = run_a_plan(decision, goal, executor, monitor,
                            log_path=output / 'normal_main_events.jsonl')
        save('normal_A_to_B.json', normal)
        save('ActionRequest_normal.json', normal['action_request'])
        normal_result = normal['execution_result']
        check('Actual A control plan contains six subtask IDs', len(normal['original_a_task']['subtasks']) == 6)
        check('A original Task IDs survive B feedback', normal_result['task_id'] == normal['original_a_task']['task_id'])
        check('Candidate has A execution subtask ID', normal_result['subtask_id'] == next(
            item['subtask_id'] for item in normal['original_a_task']['subtasks']
            if item['name'] == 'execute_approved_action'))
        check('Approved candidate executes exactly one real SMPL step',
              normal_result['success'] and adapter.step_calls == before + 1)

        # Same seed and same task template; only the test candidate changes.
        reject_initial = executor.reset(seed=1101)
        check('Normal and rejection cases use the same reset observation', np.array_equal(initial, reject_initial))
        invalid = adapter.action_space.high + (adapter.action_space.high - adapter.action_space.low)
        before_reject = adapter.step_calls
        rejected = run_a_plan(decision, goal, executor, monitor, explicit_action=invalid,
                              log_path=output / 'rejected_main_events.jsonl')
        save('rejected_A_to_B.json', rejected)
        save('ActionRequest_rejected.json', rejected['action_request'])
        rejected_result = rejected['execution_result']
        check('D-format approved=false returns rejection to A',
              not rejected['safety_decision']['approved']
              and rejected_result['execution_status'] == 'rejected')
        check('Rejected candidate never enters env.step', adapter.step_calls == before_reject and adapter.step_count == 0)
        check('Rejected execution subtask is blocked', next(
            item for item in rejected['task_feedback']['subtasks']
            if item['name'] == 'execute_approved_action')['status'] == 'blocked')
        check('Rejection reason reaches original parent Task feedback', bool(rejected['task_feedback']['failure_reason']))

        executor.reset(seed=1101)
        before_inspection = adapter.step_calls
        inspected = run_a_plan(decision, '检查当前仿真状态', executor, monitor,
                               log_path=output / 'inspect_main_events.jsonl')
        save('inspect_A_to_B.json', inspected)
        check('A inspect_state template reads observation without stepping',
              adapter.step_calls == before_inspection and inspected['action_request'] is None)

        # Directly probe D's described R-01 interface; no candidate is sent to the simulator.
        for label, bad_action in [('empty', []), ('wrong_shape', [0]),
                                  ('NaN', np.full(adapter.action_space.shape, np.nan)),
                                  ('Inf', np.full(adapter.action_space.shape, np.inf))]:
            raw = monitor.check_action(executor.get_observation(), bad_action)
            check(f'Monitor rejects {label} input', raw.get('approved') is False)
        summary = {
            'status': 'PASS', 'actual_A_code_used': True, 'real_SMPL_used': True,
            'monitor_provenance': provenance,
            'actual_D_code_used': provenance == 'D_provided_source',
            'normal_step_delta': 1, 'rejected_step_delta': adapter.step_calls - before_reject,
            'normal_reward': normal_result['reward'], 'normal_done': normal_result['done'],
            'R03_validated': False, 'C_scheduler_integrated': False,
            'scope': 'Actual A to B interface with real SMPL; D implementation provenance shown explicitly',
            'checks': checks,
        }
        save('verification.json', summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        print('Evidence:', output, flush=True)


if __name__ == '__main__':
    main()
