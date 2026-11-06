"""Run the actual A plan through B and a supplied D-compatible monitor."""
import copy
from datetime import datetime
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np

from .safety_bridge import normalize_safety_decision
from .task_contract import as_plan


def load_decision_agent(a_root=None):
    root = Path(a_root or Path(__file__).parent / 'member_A_11_02').resolve()
    filename = root / 'agents/decision_agent.py'
    if not filename.is_file():
        raise FileNotFoundError(f'Missing A source: {filename}')
    existing = sys.modules.get('agents.decision_agent')
    if existing and Path(existing.__file__).resolve() != filename:
        raise RuntimeError('Another agents.decision_agent is already loaded; run in a fresh process')
    sys.path.insert(0, str(root))
    from agents.decision_agent import DecisionAgent
    return DecisionAgent(), root


def run_a_plan(decision_agent, goal, executor, monitor, *, policy='space_midpoint',
               explicit_action=None, log_path):
    """Sequential HTN integration harness; does not implement C's scheduler."""
    task_object = decision_agent.create_task(goal)
    task = decision_agent.task_to_dict(task_object)
    decision_agent.validate_task(task_object)
    return run_task_dict(task, executor, monitor, policy=policy,
                         explicit_action=explicit_action, log_path=log_path)


def run_task_dict(task, executor, monitor, *, policy=None, explicit_action=None,
                  log_path, initial_facts=None, completed_dependencies=None):
    """Accept A's supplied Task dict directly, without requiring A generator source."""
    plan = as_plan(task)
    original_snapshot = copy.deepcopy(task)
    feedback = copy.deepcopy(task)
    feedback_meta = feedback if 'parameters' not in feedback else feedback['parameters']
    if 'parameters' in feedback and 'subtasks' in feedback:
        feedback['subtasks'] = feedback_meta.get('subtasks', [])
    by_id = {item['subtask_id']: item for item in feedback_meta.get('subtasks', [])}
    observation = None
    request = None
    review = None
    result = None
    completed = set()
    facts = set(initial_facts or [])
    blocked_reason = None
    events = []
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    if not set(plan.get('dependencies', [])).issubset(set(completed_dependencies or [])):
        raise RuntimeError('Unmet parent Task dependencies; caller must resolve them before running B')
    if not set(plan.get('preconditions', [])).issubset(facts):
        raise RuntimeError('Unmet parent Task preconditions')
    if set(plan.get('completion_criteria', [])) - {'all_subtasks_completed'}:
        raise ValueError('This prototype cannot evaluate the supplied parent completion criteria')
    produced_by_step = {
        'collect_observation': {'observation_available', 'observation_received'},
        'evaluate_goal': {'goal_evaluated'},
        'generate_action_intent': {'candidate_action_available', 'action_intent_created'},
        'safety_review': {'safety_decision_received', 'monitoring_passed', 'safety_passed'},
        'execute_approved_action': {'execution_feedback_available', 'execution_result_received'},
        'verify_result': {'result_verified'},
        'evaluate_state': {'state_evaluated'},
        'return_state_result': {'state_result_returned'},
    }
    steps = sorted(plan['subtasks'], key=lambda item: item['order'])
    if not steps:
        raise ValueError('No subtasks supplied')
    seen = set()
    orders = set()
    for step in steps:
        if (type(step['order']) is not int or step['order'] < 1
                or step['order'] in orders or not set(step['dependencies']).issubset(seen)):
            raise ValueError('Subtask order/dependencies are invalid')
        if step['name'] not in produced_by_step:
            raise ValueError(f"Unsupported A subtask: {step['name']}")
        if set(step.get('completion_criteria', [])) - produced_by_step[step['name']]:
            raise ValueError(f"Unsupported completion criteria for {step['name']}")
        seen.add(step['subtask_id'])
        orders.add(step['order'])

    def record(event, **values):
        item = {'time': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                'event': event, 'task_id': task['task_id'], **values}
        events.append(item)
        with Path(log_path).open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + '\n')

    record('actual_A_task_received', task=task, scheduler='not connected; original A template order')
    for step in steps:
        status = by_id[step['subtask_id']]
        if blocked_reason:
            status['status'] = 'blocked'
            status['failure_reason'] = blocked_reason
            record('subtask_blocked', subtask_id=step['subtask_id'], name=step['name'], reason=blocked_reason)
            continue
        if not set(step['dependencies']).issubset(completed):
            raise RuntimeError(f"Unmet dependencies: {step['subtask_id']}")
        if not set(step.get('preconditions', [])).issubset(facts):
            raise RuntimeError(f"Unmet preconditions: {step['subtask_id']}")
        name = step['name']
        record('subtask_started', subtask_id=step['subtask_id'], name=name,
               step_calls=executor.adapter.step_calls)
        if name == 'collect_observation':
            observation = executor.get_observation()
            facts.update(produced_by_step[name])
        elif name == 'evaluate_goal':
            # Validate A's declared scope only; do not invent an optimized controller.
            environment = plan.get('constraints', {}).get('environment', 'AtropineEnv')
            if plan['task_type'] != 'control_cycle' or environment not in ('AtropineEnv', 'AtropineEnvGym'):
                raise ValueError('This integration runner supports only the supplied Atropine control template')
            facts.add('goal_evaluated')
        elif name == 'generate_action_intent':
            execution_step = next(item for item in plan['subtasks'] if item['name'] == 'execute_approved_action')
            calls_before = executor.adapter.step_calls
            if explicit_action is None:
                if policy is None:
                    request = executor.prepare_action(task, subtask=execution_step)
                else:
                    request = executor.prepare_action(task, action_policy=policy, subtask=execution_step)
            else:
                request = executor.prepare_action(task, action=explicit_action, subtask=execution_step)
            if executor.adapter.step_calls != calls_before:
                raise AssertionError('Preparing an action advanced SMPL')
            record('action_request_generated', action_request=request,
                   step_calls_unchanged=True)
            facts.update(produced_by_step[name])
        elif name == 'safety_review':
            calls_before = executor.adapter.step_calls
            if callable(getattr(monitor, 'check_request', None)):
                raw_review = monitor.check_request(copy.deepcopy(request), copy.deepcopy(observation))
                monitor_api = 'check_request'
            else:
                raw_review = monitor.check_action(copy.deepcopy(observation), copy.deepcopy(request['action']))
                monitor_api = 'check_action'
            review = normalize_safety_decision(request, raw_review)
            if executor.adapter.step_calls != calls_before:
                raise AssertionError('Monitoring advanced SMPL')
            record('monitoring_returned', raw_review=raw_review, safety_decision=review,
                   monitor_api=monitor_api,
                   step_calls_unchanged=True)
            facts.add('safety_decision_received')
            if review['approved']:
                facts.update({'monitoring_passed', 'safety_passed'})
            else:
                # Close a rejection to produce A/C feedback; this never enters env.step.
                result = executor.reject_action_request(request, review)
                if executor.adapter.step_calls != calls_before:
                    raise AssertionError('A rejected action advanced SMPL')
                status['status'] = 'rejected'
                status['failure_reason'] = review['reason']
                status['satisfied_completion_criteria'] = [c for c in step.get('completion_criteria', []) if c in facts]
                blocked_reason = review['reason']
                record('rejection_feedback', result=result)
                continue
        elif name == 'execute_approved_action':
            if review is None or not review['approved']:
                raise RuntimeError('Execution cannot precede a monitoring pass')
            result = executor.execute_action(request, review)
            record('actual_execution_feedback', result=result)
            if not result['success']:
                status['status'] = 'failed'
                blocked_reason = result['failure_reason']
                continue
            facts.update(produced_by_step[name])
        elif name == 'verify_result':
            if result['task_id'] != task['task_id'] or not result['success']:
                raise AssertionError('Execution feedback does not match A Task')
            record('single_step_result_verified', production_goal_evaluated=False,
                   completion_scope='single-step interface cycle only')
            facts.add('result_verified')
        elif name == 'evaluate_state':
            if not np.isfinite(observation).all():
                raise RuntimeError('Nonfinite observation')
            facts.add('state_evaluated')
            record('observation_format_evaluated', R03_enabled=False,
                   process_risk_assessed=False)
        elif name == 'return_state_result':
            record('observation_returned_to_A', observation=observation)
            facts.add('state_result_returned')
        else:
            raise ValueError(f'Unsupported A subtask: {name}')
        if not set(step.get('completion_criteria', [])).issubset(facts):
            raise RuntimeError(f"Subtask completion criteria not satisfied: {step['subtask_id']}")
        completed.add(step['subtask_id'])
        status['status'] = 'completed'
        status['satisfied_completion_criteria'] = list(step.get('completion_criteria', []))
        record('subtask_completed', subtask_id=step['subtask_id'], name=name,
               step_calls=executor.adapter.step_calls)

    if task != original_snapshot:
        raise AssertionError('Original A Task was modified')
    feedback_meta['status'] = 'blocked' if blocked_reason else 'completed'
    if 'parameters' in feedback and 'status' in feedback:
        feedback['status'] = feedback_meta['status']
    all_completed = all(item['status'] == 'completed' for item in by_id.values())
    feedback_meta['satisfied_completion_criteria'] = ['all_subtasks_completed'] if all_completed else []
    feedback['failure_reason'] = blocked_reason
    feedback['completion_scope'] = 'single-step interface cycle' if plan.get('task_type') == 'control_cycle' else 'observation retrieval and format check'
    record('feedback_ready_for_A', feedback=feedback)
    return {'original_a_task': original_snapshot, 'task_feedback': feedback,
            'task_status': feedback_meta['status'],
            'action_request': request, 'safety_decision': review,
            'execution_result': result, 'events': events}
