"""Synchronous A/C/B/D orchestration; only B may advance the environment."""
from copy import deepcopy
import numpy as np
from agents.decision_agent import DecisionAgent
from agents.scheduling_agent import SchedulingAgent
from agents.monitoring_agent import MonitoringAgent
from .execution_agent import ExecutionAgent


class AgentCluster:
    def __init__(self, adapter, *, log_path=None):
        self.adapter = adapter
        self.decision = DecisionAgent()
        self.execution = ExecutionAgent(adapter, log_path=log_path)
        self.monitor = MonitoringAgent(adapter.action_space)
        self.scheduler = SchedulingAgent()

    def run(self, goal, *, action=None, seed=1101):
        # Every call starts a new episode and invalidates pending action requests.
        self.scheduler = SchedulingAgent()
        task = self.decision.task_to_dict(self.decision.create_task(goal))
        observation = self.execution.reset(seed=seed)
        records = [{
            'task_id': task['task_id'], 'subtask_id': sub['subtask_id'],
            'task_type': sub['name'], 'priority': task['priority'],
            'constraints': deepcopy(task['constraints']),
            'dependencies': sub['dependencies'], 'status': 'pending',
        } for sub in task['subtasks']]
        self.scheduler.register_tasks(records)
        by_id = {sub['subtask_id']: sub for sub in task['subtasks']}
        events, request, review, outcome = [], None, None, None
        error = None
        while ready := self.scheduler.ready_tasks():
            record = ready[0]
            sub = by_id[record['subtask_id']]
            owner = {'state_read': 'B', 'decision': 'A', 'monitoring': 'D',
                     'environment_execution': 'B'}[sub['required_capability']]
            bid = { 'task_id': task['task_id'], 'subtask_id': sub['subtask_id'],
                    'agent_id': owner, 'estimated_cost': 1,
                    'capability_match': 1, 'load': 0 }
            assignment = self.scheduler.schedule(record, [bid])
            contract = self.scheduler.assignment_for_execution(assignment)
            status, reason = 'completed', None
            try:
                name = sub['name']
                if name == 'collect_observation':
                    observation = self.execution.get_observation()
                elif name in ('evaluate_state', 'evaluate_goal'):
                    if not np.isfinite(np.asarray(observation, dtype=float)).all():
                        raise ValueError('Invalid observation')
                elif name == 'generate_action_intent':
                    execution_sub = next(s for s in task['subtasks']
                                         if s['name'] == 'execute_approved_action')
                    request = self.execution.prepare_action(task, action=action,
                                                            subtask=execution_sub)
                elif name == 'safety_review':
                    review = self.monitor.check_action(request['observation'], request['action'])
                    review = {**review, 'request_id': request['request_id'],
                              'task_id': task['task_id']}
                    if not review['approved']:
                        outcome = self.execution.reject_action_request(request, review)
                        status, reason = 'failed', 'safety_rejected'
                elif name == 'execute_approved_action':
                    if contract['winner_id'] != 'B':
                        raise RuntimeError('Execution assignment must belong to B')
                    outcome = self.execution.execute_action(request, review)
                    if not outcome['success']:
                        status, reason = 'failed', outcome['failure_reason']
                elif name == 'verify_result':
                    if not outcome or not outcome['success']:
                        raise RuntimeError('Execution did not complete')
                elif name != 'return_state_result':
                    raise ValueError(f'Unsupported subtask: {name}')
            except Exception as exc:
                status, reason = 'failed', f'{type(exc).__name__}: {exc}'
            feedback = self.scheduler.update_feedback({
                'task_id': task['task_id'], 'subtask_id': sub['subtask_id'],
                'execution_status': status, 'failure_reason': reason,
            })
            sub['status'] = status.upper()
            events.append({'name': sub['name'], 'assignment': contract,
                           'feedback': feedback})
            if status == 'failed':
                error = reason
                break
        complete = all(s['status'] == 'COMPLETED' for s in task['subtasks'])
        task['status'] = 'COMPLETED' if complete else 'FAILED'
        return {'task': task, 'success': complete, 'failure_reason': error,
                'observation': outcome['observation'] if outcome else observation,
                'execution_result': outcome, 'events': events,
                'subtask_statuses': {r['subtask_id']: self.scheduler.get_task_status(
                    task['task_id'], r['subtask_id']) for r in records}}
