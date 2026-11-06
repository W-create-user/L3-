"""Exercise the latest Task supplied in chat; do not claim its generator was supplied."""
import argparse
import copy
from datetime import datetime
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np

from . import EnvironmentAdapter, ExecutionAgent, normalize_task
from .a_integration import run_task_dict
from .demo_actual_a import load_monitor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-file', default=str(Path(__file__).parent / 'examples/a_task_current.json'))
    parser.add_argument('--monitor-file')
    args = parser.parse_args()
    task_file = Path(args.task_file).resolve()
    task = json.loads(task_file.read_text(encoding='utf-8-sig'))
    initial_task = copy.deepcopy(task)
    mapped = normalize_task(task)
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
    output = Path(__file__).parent / 'logs_v12' / stamp
    output.mkdir(parents=True)
    checks = []

    def save(name, value):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

    def check(name, value):
        if not value:
            raise AssertionError(name)
        checks.append({'check': name, 'status': 'PASS'})
        print('PASS:', name, flush=True)

    save('A_Task_original.json', task)
    save('B_Task_mapped.json', mapped)
    with EnvironmentAdapter() as adapter:
        monitor, provenance = load_monitor(adapter.action_space, args.monitor_file)
        executor = ExecutionAgent(adapter, output / 'B_events.jsonl')
        save('manifest.json', {
            'interface_version': '1.2', 'task_source': 'A Task dictionary supplied by user; latest generator code not supplied',
            'input_file': str(task_file), 'input_sha256': hashlib.sha256(task_file.read_bytes()).hexdigest(),
            'monitor_provenance': provenance, 'python': sys.version,
            'versions': {p: version(p) for p in ['numpy','gym','casadi']},
            'normalize': False, 'uss_subtracted': True,
            'action_low': adapter.action_space.low.tolist(), 'action_high': adapter.action_space.high.tolist(),
            'code_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
        })
        initial = executor.reset(seed=1101)
        count = adapter.step_calls
        raw_request = executor.prepare_action(task)
        mapped_request = executor.prepare_action(mapped)
        save('ActionRequest_raw.json', raw_request)
        save('ActionRequest_mapped.json', mapped_request)
        check('A does not supply action or action_policy',
              'action' not in task and 'action' not in mapped['parameters']
              and 'action_policy' not in mapped['parameters'])
        check('Raw and mapped Tasks generate the same B-local candidate', raw_request['action'] == mapped_request['action'])
        check('Preparation and observation reads never call env.step', adapter.step_calls == count)
        check('Original A task_id is preserved', raw_request['task_id'] == task['task_id'])
        expected_step = next(s for s in mapped['parameters']['subtasks'] if s['name'] == 'execute_approved_action')
        check('Original execution subtask_id is preserved', raw_request['subtask_id'] == expected_step['subtask_id'])
        for field in ['task_type','priority','preconditions','dependencies','constraints','subtasks','completion_criteria']:
            check(f'A metadata retained: {field}', raw_request['task_context']['parameters'][field] == mapped['parameters'][field])

        # The pending probes above are intentionally invalidated by reset before each case.
        executor.reset(seed=1101)
        before_raw = adapter.step_calls
        raw_run = run_task_dict(task, executor, monitor, log_path=output / 'raw_plan_events.jsonl')
        save('RawTask_result.json', raw_run)
        check('New raw Task preconditions and completion criteria run successfully',
              raw_run['task_status'] == 'completed' and adapter.step_calls == before_raw + 1)

        mapped_initial = executor.reset(seed=1101)
        check('Both representations use the same initial observation', np.array_equal(initial, mapped_initial))
        before_mapped = adapter.step_calls
        mapped_run = run_task_dict(mapped, executor, monitor, log_path=output / 'mapped_plan_events.jsonl')
        save('MappedTask_result.json', mapped_run)
        check('Three-field representation alone runs the same plan',
              mapped_run['task_status'] == 'completed' and adapter.step_calls == before_mapped + 1)

        executor.reset(seed=1101)
        before_reject = adapter.step_calls
        invalid = adapter.action_space.high + (adapter.action_space.high - adapter.action_space.low)
        rejected = run_task_dict(mapped, executor, monitor, explicit_action=invalid,
                                 log_path=output / 'rejected_plan_events.jsonl')
        save('RejectedTask_result.json', rejected)
        check('Rejected request never calls env.step', adapter.step_calls == before_reject)
        check('Rejected plan cannot satisfy all_subtasks_completed',
              rejected['task_status'] == 'blocked'
              and rejected['task_feedback']['parameters']['satisfied_completion_criteria'] == [])
        check('A original input remains unchanged', task == initial_task)

        summary = {
            'status': 'PASS', 'interface_version': '1.2', 'task_id': task['task_id'],
            'task_source': 'user_supplied_A_Task_dictionary', 'latest_A_generator_tested': False,
            'actual_D_code_used': provenance == 'D_provided_source',
            'monitor_provenance': provenance,
            'raw_step_delta': 1, 'mapped_step_delta': 1, 'reject_step_delta': 0,
            'raw_reward': raw_run['execution_result']['reward'],
            'mapped_reward': mapped_run['execution_result']['reward'],
            'checks': checks,
            'policy_scope': 'B local midpoint interface baseline; observation validity check, not feedback optimization',
        }
        save('verification.json', summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        print('Evidence:', output, flush=True)


if __name__ == '__main__':
    main()
