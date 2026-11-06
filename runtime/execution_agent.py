"""Receive a Task, prepare an ActionRequest, execute only with a matching review."""
from __future__ import annotations

import copy
from datetime import datetime
import json
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

from .environment_adapter import to_plain
from .safety_bridge import normalize_safety_decision
from .task_contract import normalize_task
from .local_policy import space_midpoint


class ExecutionAgent:
    def __init__(self, adapter, log_path=None, *, default_policy='space_midpoint'):
        if default_policy != 'space_midpoint':
            raise ValueError('Supported B local baseline: space_midpoint')
        self.adapter = adapter
        self.default_policy = default_policy
        self._pending = {}
        stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
        self.log_path = Path(log_path) if log_path else Path(__file__).parent / 'logs' / f'{stamp}.jsonl'
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._record('agent_created', source_dir=adapter.source_dir)

    def _record(self, event, **values):
        record = {'time': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                  'event': event, **values}
        line = json.dumps(to_plain(record), ensure_ascii=False, allow_nan=False)
        with self.log_path.open('a', encoding='utf-8') as stream:
            stream.write(line + '\n')

    def reset(self, seed=1101):
        self._pending.clear()
        observation = self.adapter.reset(seed=seed)
        self._record('environment_reset', seed=seed, observation=observation,
                     state_version=self.adapter.state_version,
                     step_count=self.adapter.step_count)
        return observation

    def get_observation(self):
        return self.adapter.get_observation()

    def prepare_action(self, task, *, action=None, action_policy=None, subtask=None):
        """Pure preparation with respect to SMPL: no reset and no step."""
        self.adapter.require_ready()
        canonical_task = normalize_task(task)
        task_id = canonical_task['task_id']
        parameters = canonical_task['parameters']
        task_type = parameters.get('task_type')
        task_snapshot = json.loads(json.dumps(to_plain(task), ensure_ascii=False, allow_nan=False))
        observation = self.get_observation()
        subtasks = parameters.get('subtasks', [])
        if subtask is None:
            execution_steps = [item for item in subtasks if item.get('name') == 'execute_approved_action']
            if len(execution_steps) > 1:
                raise ValueError('Multiple execution subtasks require an explicit subtask selection')
            if execution_steps:
                subtask = execution_steps[0]
        canonical_subtask = None
        if subtask is not None:
            if not isinstance(subtask, dict):
                raise ValueError('subtask must be an A SubTask dictionary')
            matches = [item for item in subtasks
                       if item.get('subtask_id') == subtask.get('subtask_id')]
            if len(matches) != 1 or matches[0] != subtask:
                raise ValueError('Subtask must match an actual subtask in the original A Task')
            canonical_subtask = matches[0]
        has_top = 'action' in task
        has_nested = 'action' in parameters
        has_argument = action is not None
        if sum([has_top, has_nested, has_argument]) > 1:
            raise ValueError('Provide an action in only one location')
        has_action = has_top or has_nested or has_argument
        if action_policy is not None and 'action_policy' in parameters:
            raise ValueError('Provide action_policy in only one location')
        policy = action_policy if action_policy is not None else parameters.get('action_policy')
        if not has_action and policy is None and task_type == 'control_cycle':
            policy = self.default_policy
        if has_action:
            if policy is not None:
                raise ValueError('Explicit action and action_policy cannot be combined')
            if not has_argument:
                action = task['action'] if has_top else parameters['action']
            action_source = 'caller_explicit_action' if has_argument else 'task_explicit_action'
        elif policy == 'space_midpoint':
            action = space_midpoint(observation, self.adapter.action_space)
            action_source = 'B_local_space_midpoint_baseline'
        else:
            raise ValueError('Provide parameters.action, or explicitly select action_policy=space_midpoint for testing')
        # Deliberately preserve out-of-range finite candidates for D's R-02 review.
        # The adapter still enforces live bounds again immediately before env.step.
        action = self.adapter.coerce_candidate(action)
        request = {
            'schema_version': '1.2',
            'request_id': str(uuid4()),
            'task_id': task_id,
            'task_name': canonical_task['task_name'],
            'parent_task_id': task_id,
            'subtask_id': canonical_subtask.get('subtask_id') if canonical_subtask else None,
            'subtask_name': canonical_subtask.get('name') if canonical_subtask else None,
            'goal_type': task_type,
            'task_type': task_type,
            'task_context': copy.deepcopy(canonical_task),
            'action': action.tolist(),
            'action_source': action_source,
            'observation': observation,
            'state_version': self.adapter.state_version,
            'step_count': self.adapter.step_count,
            'environment': getattr(self.adapter, 'environment_name', 'SMPL.AtropineEnvGym'),
            'normalize': False,
            'uss_subtracted': True,
            'action_space': {
                'shape': list(self.adapter.action_space.shape),
                'dtype': str(self.adapter.action_space.dtype),
                'low': self.adapter.action_space.low.tolist(),
                'high': self.adapter.action_space.high.tolist(),
            },
        }
        self._pending[request['request_id']] = copy.deepcopy(request)
        self._record('action_prepared', task=task_snapshot, normalized_task=canonical_task,
                     action_request=request,
                     step_calls=self.adapter.step_calls)
        return copy.deepcopy(request)

    def execute_action(self, action_request, review=None):
        """Accept D's approved-based review or the older request-bound review."""
        try:
            return self._execute_reviewed(action_request, review)
        except (ValueError, TypeError, RuntimeError) as error:
            self._record('execution_not_permitted', reason=str(error))
            raise

    def reject_action_request(self, action_request, review):
        """Close a rejected request and return feedback, without executing it."""
        decision = normalize_safety_decision(action_request, review)
        if decision['approved']:
            raise ValueError('reject_action_request accepts only a rejection')
        return self.execute_action(action_request, decision)

    def _execute_reviewed(self, request, review):
        if not isinstance(request, dict):
            raise TypeError('Pass the ActionRequest returned by prepare_action, not a bare action vector')
        request_id = request.get('request_id')
        if not isinstance(request_id, str) or request_id not in self._pending:
            raise ValueError('Unknown or already consumed request_id')
        pending = self._pending[request_id]
        if (json.dumps(to_plain(request), sort_keys=True, allow_nan=False)
                != json.dumps(pending, sort_keys=True, allow_nan=False)):
            raise ValueError('ActionRequest was modified; generate and review a new request')
        if pending['state_version'] != self.adapter.state_version:
            raise ValueError('Environment changed; generate and review a new request')
        self.adapter.require_ready()
        review = normalize_safety_decision(pending, review)
        self._record('review_received', review=review, request_id=request_id)
        del self._pending[request_id]
        common = {'task_id': pending['task_id'], 'request_id': request_id,
                  'parent_task_id': pending['parent_task_id'],
                  'subtask_id': pending['subtask_id'],
                  'safety_decision': review}
        if review['decision'] == 'reject':
            result = {
                **common, 'execution_status': 'rejected', 'success': False,
                'executed': False, 'observation': self.get_observation(),
                'action': pending['action'], 'reward': None,
                'done': self.adapter.done, 'info': {},
                'step_count': self.adapter.step_count,
                'step_calls': self.adapter.step_calls,
                'state_version': self.adapter.state_version,
                'failure_reason': review['reason'],
            }
        else:
            try:
                outcome = self.adapter.step(pending['action'])
            except Exception as error:
                self._record('environment_exception', **common,
                             error_type=type(error).__name__, reason=str(error))
                raise
            result = {
                **common, **outcome, 'executed': True,
                'execution_status': 'completed' if outcome['success'] else 'failed',
            }
        self._record('execution_result', result=result)
        return result
