"""Persist actual transitions to JSONL and numeric CSV without advancing SMPL."""
from __future__ import annotations

import copy
import csv
from datetime import datetime
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo


def timestamp():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def json_safe(value):
    if hasattr(value, 'full'):
        value = value.full().tolist()
    elif hasattr(value, 'tolist'):
        value = value.tolist()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {'__nonfinite__': 'NaN' if math.isnan(value) else ('+Infinity' if value > 0 else '-Infinity')}
    return value


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def csv_value(value):
    if value is None:
        return ''
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, float) and not math.isfinite(value):
        return 'NaN' if math.isnan(value) else ('+Infinity' if value > 0 else '-Infinity')
    return value


class ExperimentRecorder:
    def __init__(self, directory, *, run_id, episode_id, observation_dim, action_dim, dt_minutes):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.run_id = run_id
        self.episode_id = episode_id
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.dt_minutes = dt_minutes
        self.attempts = 0
        self.executed = 0
        self.successful = 0
        self.rejected = 0
        self.failures = 0
        self.observation_rows = 0
        self._requests = set()
        self._initialized = False
        self._last_state = None
        self._closed = False
        self._steps_stream = (self.directory / 'steps.csv').open('w', encoding='utf-8-sig', newline='')
        self._observations_stream = (self.directory / 'observations.csv').open('w', encoding='utf-8-sig', newline='')
        self._events_stream = (self.directory / 'events.jsonl').open('w', encoding='utf-8')
        self.step_fields = [
            'run_id', 'episode_id', 'attempt_index', 'task_invocation_id', 'timestamp',
            'task_id', 'subtask_id', 'request_id', 'execution_status', 'approved',
            'executed', 'success', 'reward', 'done_after', 'rule_id', 'risk_level',
            'failure_reason', 'step_count_before', 'step_count_after',
            'step_calls_before', 'step_calls_after', 'state_version_before', 'state_version_after',
            'nominal_sim_time_before_min', 'nominal_sim_time_after_min', 'wall_duration_ms',
            'main_trace_file',
        ] + [f'action_{i:02d}' for i in range(action_dim)]
        self.observation_fields = [
            'run_id', 'episode_id', 'observation_index', 'timestamp', 'source',
            'task_id', 'request_id', 'step_count', 'state_version', 'nominal_sim_time_min',
        ] + [f'obs_{i:03d}' for i in range(observation_dim)]
        self._steps = csv.DictWriter(self._steps_stream, fieldnames=self.step_fields)
        self._observations = csv.DictWriter(self._observations_stream, fieldnames=self.observation_fields)
        self._steps.writeheader()
        self._observations.writeheader()
        self._flush()

    def _flush(self):
        self._steps_stream.flush()
        self._observations_stream.flush()
        self._events_stream.flush()

    def _event(self, kind, **data):
        event = {'event': kind, 'timestamp': timestamp(), 'run_id': self.run_id,
                 'episode_id': self.episode_id, **data}
        self._events_stream.write(json.dumps(json_safe(event), ensure_ascii=False, allow_nan=False) + '\n')

    def _observation(self, state, *, source, task_id='', request_id=''):
        observation = state.get('observation')
        if observation is None:
            return
        if len(observation) != self.observation_dim:
            raise ValueError('Observation length does not match runtime dimension')
        row = {'run_id': self.run_id, 'episode_id': self.episode_id,
               'observation_index': self.observation_rows, 'timestamp': timestamp(),
               'source': source, 'task_id': task_id, 'request_id': request_id,
               'step_count': state['step_count'], 'state_version': state['state_version'],
               'nominal_sim_time_min': state['step_count'] * self.dt_minutes}
        row.update({f'obs_{i:03d}': csv_value(v) for i, v in enumerate(observation)})
        self._observations.writerow(row)
        self.observation_rows += 1

    def record_initial(self, state, initial_conditions):
        if self._initialized:
            raise RuntimeError('Initial conditions are recorded exactly once per episode')
        if len(state['observation']) != self.observation_dim:
            raise ValueError('Invalid initial observation dimension')
        write_json(self.directory / 'initial_conditions.json', initial_conditions)
        self._event('reset', state=state, initial_conditions_file='initial_conditions.json')
        self._observation(state, source='reset')
        self._last_state = copy.deepcopy(state)
        self._initialized = True
        self._flush()

    def record_attempt(self, trace, before, after, *, wall_duration_ms, main_trace_file):
        if not self._initialized:
            raise RuntimeError('Record initial conditions before attempts')
        request = trace['action_request']
        result = trace['execution_result']
        if request is None or result is None:
            raise ValueError('A control attempt must have ActionRequest and ExecutionResult')
        if request['request_id'] in self._requests:
            raise ValueError('Duplicate request cannot create a second transition row')
        if (request['request_id'] != result['request_id'] or request['task_id'] != result['task_id']
                or request['step_count'] != before['step_count']):
            raise ValueError('Request/result/before state mismatch')
        if before != self._last_state or request['observation'] != before['observation']:
            raise ValueError('Observation/state chain is discontinuous')
        if request['action'] != result['action'] or len(request['action']) != self.action_dim:
            raise ValueError('Action/request/result mismatch')
        called = after['step_calls'] - before['step_calls']
        if called != int(result['executed']):
            raise ValueError('Executed flag differs from actual env.step call counter')
        if result['success'] and (not result['executed'] or after['step_count'] != before['step_count'] + 1):
            raise ValueError('A successful step must advance the actual counter exactly once')
        if not result['executed'] and after != before:
            raise ValueError('A rejected action must not change environment state or counters')
        if result['observation'] != after['observation'] or result['step_count'] != after['step_count']:
            raise ValueError('Result does not match the actual post-call state')
        self.attempts += 1
        self._requests.add(request['request_id'])
        review = result['safety_decision']
        event_time = timestamp()
        event = dict(attempt_index=self.attempts, before=before, after=after, trace=trace,
                     wall_duration_ms=wall_duration_ms, main_trace_file=main_trace_file)
        self._event('transition', **event)
        row = {
            'run_id': self.run_id, 'episode_id': self.episode_id,
            'attempt_index': self.attempts,
            'task_invocation_id': f'{self.episode_id}:{self.attempts:04d}', 'timestamp': event_time,
            'task_id': result['task_id'], 'subtask_id': result.get('subtask_id'),
            'request_id': result['request_id'], 'execution_status': result['execution_status'],
            'approved': csv_value(review['approved']), 'executed': csv_value(result['executed']),
            'success': csv_value(result['success']), 'reward': csv_value(result['reward']),
            'done_after': csv_value(result['done']), 'rule_id': review.get('rule_id'),
            'risk_level': review.get('risk_level'), 'failure_reason': result['failure_reason'],
            'step_count_before': before['step_count'], 'step_count_after': after['step_count'],
            'step_calls_before': before['step_calls'], 'step_calls_after': after['step_calls'],
            'state_version_before': before['state_version'], 'state_version_after': after['state_version'],
            'nominal_sim_time_before_min': before['step_count'] * self.dt_minutes,
            'nominal_sim_time_after_min': after['step_count'] * self.dt_minutes,
            'wall_duration_ms': wall_duration_ms, 'main_trace_file': main_trace_file,
        }
        row.update({f'action_{i:02d}': csv_value(v) for i, v in enumerate(request['action'])})
        self._steps.writerow(row)
        self.executed += int(result['executed'])
        self.successful += int(result['success'])
        self.rejected += int(result['execution_status'] == 'rejected')
        self.failures += int(result['execution_status'] == 'failed')
        if result['executed']:
            self._observation(after, source='step' if result['success'] else 'failed_step',
                              task_id=result['task_id'], request_id=result['request_id'])
        self._last_state = copy.deepcopy(after)
        self._flush()

    def record_exception(self, error, *, before, after, main_trace_file, traceback_text):
        self.attempts += 1
        self.failures += 1
        delta = after['step_calls'] - before['step_calls']
        self.executed += delta
        self._event('exception', attempt_index=self.attempts,
                    error_type=type(error).__name__, reason=str(error),
                    before=before, after=after, main_trace_file=main_trace_file,
                    traceback=traceback_text)
        row = {key: '' for key in self.step_fields}
        row.update(run_id=self.run_id, episode_id=self.episode_id,
                   attempt_index=self.attempts, task_invocation_id=f'{self.episode_id}:{self.attempts:04d}',
                   timestamp=timestamp(), execution_status='exception', executed=csv_value(delta > 0),
                   success='false', failure_reason=str(error),
                   step_count_before=before['step_count'], step_count_after=after['step_count'],
                   step_calls_before=before['step_calls'], step_calls_after=after['step_calls'],
                   state_version_before=before['state_version'], state_version_after=after['state_version'],
                   nominal_sim_time_before_min=before['step_count'] * self.dt_minutes,
                   nominal_sim_time_after_min=after['step_count'] * self.dt_minutes,
                   main_trace_file=main_trace_file)
        self._steps.writerow(row)
        self._last_state = copy.deepcopy(after)
        self._flush()

    def finish(self, *, requested_steps, stop_reason):
        summary = {
            'run_id': self.run_id, 'episode_id': self.episode_id,
            'requested_successful_steps': requested_steps, 'attempts': self.attempts,
            'actual_step_calls': self.executed, 'successful_steps': self.successful,
            'rejected_candidates': self.rejected, 'failures': self.failures,
            'observation_rows': self.observation_rows, 'stop_reason': stop_reason,
            'sampling_interval_minutes': self.dt_minutes,
            'experiment_status': 'completed' if self.successful == requested_steps and self.failures == 0 else 'stopped_early',
        }
        self._event('episode_finished', summary=summary)
        write_json(self.directory / 'episode_summary.json', summary)
        self._flush()
        return summary

    def close(self):
        if not self._closed:
            for stream in [self._steps_stream, self._observations_stream, self._events_stream]:
                stream.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
