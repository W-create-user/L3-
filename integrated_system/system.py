"""Four actual role modules, integrated with explicit dependency and review gates."""
from __future__ import annotations
import copy
import importlib.util
import json
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '执行智能体/code'))
sys.path.insert(0, str(ROOT / 'C_member_before_20261108'))
from integration_abd_1103 import EnvironmentAdapter, ExecutionAgent
from agents.scheduling_agent import SchedulingAgent


def load_class(path, name, class_name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return getattr(module, class_name)


DecisionAgent = load_class(ROOT / '决策智能体/agents/decision_agent.py', '_actual_A', 'DecisionAgent')
MonitoringAgent = load_class(ROOT / 'agents/monitoring_agent.py', '_actual_D', 'MonitoringAgent')


class DecisionCoordinator:
    """Original A planning with an integration-layer feedback inbox."""
    def __init__(self):
        self.agent = DecisionAgent()
        self.task = None
        self.feedback = []
    def create_task(self, goal):
        return self.agent.create_task(goal)
    def task_to_dict(self, task):
        self.task = self.agent.task_to_dict(task)
        return copy.deepcopy(self.task)
    def accept_result(self, result):
        if result['task_id'] != self.task['task_id']:
            raise ValueError('Feedback task mismatch')
        if [s['subtask_id'] for s in result['task']['subtasks']] != [s['subtask_id'] for s in self.task['subtasks']]:
            raise ValueError('Feedback subtask mismatch')
        if result['status'] == 'completed' and not all(s['status'] == 'completed' for s in result['task']['subtasks']):
            raise ValueError('Parent completion requires every subtask completed')
        self.feedback.append(copy.deepcopy(result))
        return {'task_id':result['task_id'], 'accepted':True, 'status':result['status'],
                'subtask_statuses':{s['subtask_id']:s['status'] for s in result['task']['subtasks']}}


class MonitoringGateway:
    """Expose check_request while preserving the actual D module's raw result."""
    def __init__(self, action_space):
        self.monitor = MonitoringAgent(action_space=action_space)
        if self.monitor.demo_mode:
            raise ValueError('Real action_space is required')

    def check_request(self, request, observation=None):
        raw = self.monitor.check_action(observation, request['action'])
        if not isinstance(raw, dict) or type(raw.get('approved')) is not bool:
            raise ValueError('D returned a malformed review')
        return {'task_id': request['task_id'], 'request_id': request['request_id'],
                'decision': 'pass' if raw['approved'] else 'reject',
                'rule_id': raw.get('rule_id'), 'reason': raw.get('reason'),
                'risk_level': raw.get('level'), 'demo_mode': raw.get('demo_mode'),
                'raw_review': raw}


class FourAgentSystem:
    """Local synchronous prototype, not a distributed or optimized controller."""
    WORKERS = {'state_read': 'B', 'environment_execution': 'B',
               'decision': 'A', 'monitoring': 'D'}

    def __init__(self, adapter, output, checker=None):
        self.adapter = adapter
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.decision = DecisionCoordinator()
        self.scheduler = SchedulingAgent()
        self.executor = ExecutionAgent(adapter, log_path=self.output / 'B_events.jsonl')
        self.monitor = MonitoringGateway(adapter.action_space)
        self.checker = checker or self.monitor.check_request
        self.events = []

    def record(self, event, **values):
        item = {'event': event, 'monotonic_seconds': time.perf_counter(), **values}
        self.events.append(item)
        with (self.output / 'events.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + '\n')

    def run(self, goal='执行一次仿真控制任务', seed=1101, scenario='normal'):
        if scenario not in {'normal', 'reject'}:
            raise ValueError('scenario must be normal or reject')
        initial = self.executor.reset(seed=seed)
        task = self.decision.task_to_dict(self.decision.create_task(goal))
        (self.output / 'A_task.json').write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding='utf-8')
        steps = sorted(task['subtasks'], key=lambda s: s['order'])
        known, orders = set(), set()
        for s in steps:
            if s['subtask_id'] in known or s['order'] in orders or not set(s['dependencies']) <= known:
                raise ValueError('A generated invalid dependencies/order')
            known.add(s['subtask_id']); orders.add(s['order'])
        c_tasks = [{'task_id': task['task_id'], 'subtask_id': s['subtask_id'],
                    'task_type': task['goal_type'], 'priority': task['priority'],
                    'dependencies': s['dependencies'], 'constraints': task['constraints'],
                    'status': 'pending'} for s in steps]
        self.scheduler.register_tasks(c_tasks)
        by_id = {s['subtask_id']: s for s in steps}
        request = review = execution = None
        observation = initial
        failure = None
        before = self.adapter.step_calls
        transitions = []
        start = time.perf_counter()
        while any(t['status'] == 'pending' for t in self.scheduler.tasks.values()):
            ready = self.scheduler.ready_tasks()
            if not ready:
                failure = 'dependency_blocked'; break
            for c_task in ready:
                sub = by_id[c_task['subtask_id']]
                worker = self.WORKERS.get(sub['required_capability'])
                if worker is None:
                    failure = 'unsupported_capability'; break
                # Configured capability indicator/cost, not measured market bids.
                bid = {'task_id': task['task_id'], 'subtask_id': sub['subtask_id'],
                       'agent_id': worker, 'estimated_cost': 1.0,
                       'capability_match': 1.0, 'load': 0.0}
                assignment = self.scheduler.assignment_for_execution(self.scheduler.schedule(c_task, [bid]))
                if (assignment['winner_id'] != worker or assignment['task_id'] != task['task_id']
                        or assignment['subtask_id'] != sub['subtask_id']):
                    raise ValueError('C Assignment identity mismatch')
                active = self.scheduler.get_active_assignment(task['task_id'], sub['subtask_id'])
                if active is None:
                    raise ValueError('C winner lock missing')
                self.record('assignment_received', subtask_name=sub['name'], bid=bid, assignment=assignment)
                status, reason = 'completed', None
                try:
                    name = sub['name']
                    if name == 'collect_observation':
                        observation = self.executor.get_observation()
                    elif name in {'evaluate_goal', 'evaluate_state'}:
                        if not np.isfinite(observation).all():
                            raise ValueError('Nonfinite observation')
                        self.record('A_state_evaluation', method='finite-state validation only; no optimization', observation=observation)
                    elif name == 'generate_action_intent':
                        action = (self.adapter.action_space.high.astype(float) + 1).tolist() if scenario == 'reject' else None
                        old_calls = self.adapter.step_calls
                        request = self.executor.prepare_action(task, action=action, action_policy=None if action is not None else 'space_midpoint')
                        if self.adapter.step_calls != old_calls:
                            raise AssertionError('Preparation advanced the environment')
                        self.record('B_candidate', request=request, step_delta=0)
                    elif name == 'safety_review':
                        review = self.checker(copy.deepcopy(request), copy.deepcopy(observation))
                        # Strict final contract; missing/unknown review blocks before step.
                        if (not isinstance(review, dict) or review.get('decision') not in {'pass', 'reject'}
                                or review.get('task_id') != task['task_id']
                                or review.get('demo_mode') is True):
                            raise ValueError('Missing, unknown, mismatched or demo safety result')
                        from integration_abd_1103.safety_bridge import normalize_safety_decision
                        normalize_safety_decision(request, review)
                        self.record('D_review', request_id=request['request_id'], review=review)
                    elif name == 'execute_approved_action':
                        # B rechecks request contents, version and review before step.
                        execution = self.executor.execute_action(request, review)
                        self.record('B_execution', result=execution)
                        if not execution['success']:
                            status = 'failed'
                            reason = 'safety_rejected' if not execution['executed'] else 'environment_error'
                        if execution['executed']:
                            transitions.append({'task_id': task['task_id'], 'subtask_id': sub['subtask_id'],
                                'request_id': request['request_id'], 'step_count': execution['step_count'],
                                'action': execution['action'], 'observation': execution['observation'],
                                'reward': execution['reward'], 'terminated': execution['terminated'],
                                'truncated': execution['truncated'], 'info': execution['info']})
                    elif name == 'verify_result':
                        if execution is None or not execution['success']:
                            raise ValueError('No successful execution result')
                        self.record('A_result_verification', result=execution, verified=True)
                    elif name == 'return_state_result':
                        self.record('A_state_result', observation=observation)
                    else:
                        raise ValueError('Unsupported HTN step: ' + name)
                except Exception as error:
                    status, reason = 'failed', ('safety_review_error' if sub['name'] == 'safety_review' else 'environment_error')
                    self.record('module_error', subtask_id=sub['subtask_id'], exception=type(error).__name__, reason=str(error))
                feedback = {'task_id': task['task_id'], 'subtask_id': sub['subtask_id'],
                            'execution_status': status, 'failure_reason': reason,
                            'exclude_failed_agent': status == 'failed' and reason not in {'safety_rejected', 'safety_review_error'}}
                normalized = self.scheduler.update_feedback(feedback)
                if self.scheduler.get_active_assignment(task['task_id'], sub['subtask_id']) is not None:
                    raise AssertionError('Terminal feedback did not release lock')
                self.record('C_feedback', feedback=normalized)
                if status == 'failed':
                    failure = reason; break
            if failure:
                break
        final = copy.deepcopy(task)
        for sub in final['subtasks']:
            sub['status'] = self.scheduler.get_task_status(task['task_id'], sub['subtask_id'])
        final['status'] = 'failed' if failure else 'completed'
        result = {'status': final['status'], 'failure_reason': failure, 'task_id': task['task_id'],
                  'seed': seed, 'scenario': scenario, 'step_delta': self.adapter.step_calls - before,
                  'elapsed_seconds': time.perf_counter() - start, 'task': final,
                  'execution_result': execution, 'all_C_locks_released': not self.scheduler.active_agent_tasks,
                  'observation_shape': list(self.adapter.observation_space.shape),
                  'action_shape': list(self.adapter.action_space.shape),
                  'action_low': self.adapter.action_space.low.tolist(), 'action_high': self.adapter.action_space.high.tolist(),
                  'D_demo_mode': self.monitor.monitor.demo_mode,
                  'scope': 'Actual A template + actual C scheduling + B execution + actual repository D rules + real SMPL; synchronous prototype'}
        accepted = self.decision.accept_result(result)
        self.record('A_task_feedback', feedback=accepted)
        (self.output / 'A_feedback.json').write_text(json.dumps(accepted, ensure_ascii=False, indent=2), encoding='utf-8')
        (self.output / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        import csv
        with (self.output / 'transitions.csv').open('w', encoding='utf-8-sig', newline='') as f:
            fields = ['task_id','subtask_id','request_id','step_count','action','observation','reward','terminated','truncated','info']
            writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
            writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()} for row in transitions)
        return result
