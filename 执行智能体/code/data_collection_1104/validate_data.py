"""Independently read saved CSV/JSONL and verify linkage and physical step counts."""
import argparse
import csv
import json
from pathlib import Path


def strict_json(line):
    def invalid(value):
        raise ValueError(f'Non-standard JSON constant: {value}')
    return json.loads(line, parse_constant=invalid)


def check(condition, message):
    if not condition:
        raise ValueError(message)


def numeric_equal(text, expected):
    if expected is None:
        return text == ''
    if isinstance(expected, dict) and '__nonfinite__' in expected:
        return text == expected['__nonfinite__']
    return float(text) == float(expected)


def validate_episode(directory):
    directory = Path(directory)
    events = [strict_json(line) for line in (directory / 'events.jsonl').read_text(encoding='utf-8').splitlines() if line]
    with (directory / 'steps.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    with (directory / 'observations.csv').open(encoding='utf-8-sig', newline='') as stream:
        states = list(csv.DictReader(stream))
    resets = [event for event in events if event['event'] == 'reset']
    transitions = [event for event in events if event['event'] == 'transition']
    exceptions = [event for event in events if event['event'] == 'exception']
    attempts = [event for event in events if event['event'] in ('transition', 'exception')]
    summary = strict_json((directory / 'episode_summary.json').read_text(encoding='utf-8'))
    check(len(resets) == 1, 'Exactly one reset is required per episode')
    check(len(rows) == len(attempts) == summary['attempts'], 'CSV/raw transition row count mismatch')
    initial = strict_json((directory / 'initial_conditions.json').read_text(encoding='utf-8'))
    check(initial['state'] == resets[0]['state'], 'Initial conditions file differs from reset log')
    expected_states = [(resets[0]['state'], 'reset')]
    requests = set()
    last_state = resets[0]['state']
    for row, event in zip(rows, attempts):
        if event['event'] == 'exception':
            check(row['execution_status'] == 'exception', 'Exception must remain an exception in CSV')
            check(row['failure_reason'] == event['reason'], 'Exception reason mismatch')
            check(row['reward'] == '', 'An exception cannot invent a reward')
            check(event['before'] == last_state, 'Exception state chain mismatch')
            check(int(row['step_calls_after']) - int(row['step_calls_before'])
                  == event['after']['step_calls'] - event['before']['step_calls'], 'Exception step-call count mismatch')
            last_state = event['after']
            continue
        request = event['trace']['action_request']
        result = event['trace']['execution_result']
        before, after = event['before'], event['after']
        check(before == last_state, 'Raw observation/counter chain is discontinuous')
        check(request['request_id'] not in requests, 'Duplicate request in raw data')
        requests.add(request['request_id'])
        check(row['request_id'] == request['request_id'] == result['request_id'], 'request_id mismatch')
        check(row['task_id'] == request['task_id'] == result['task_id'], 'task_id mismatch')
        check(row['execution_status'] == result['execution_status'], 'Execution status mismatch')
        check(row['executed'] == str(result['executed']).lower(), 'Executed flag mismatch')
        check(row['success'] == str(result['success']).lower(), 'Success flag mismatch')
        check(numeric_equal(row['reward'], result['reward']), 'Reward mismatch')
        check(float(row['nominal_sim_time_before_min']) == before['step_count'] * summary['sampling_interval_minutes'], 'Before simulation time mismatch')
        check(float(row['nominal_sim_time_after_min']) == after['step_count'] * summary['sampling_interval_minutes'], 'After simulation time mismatch')
        for key in ['step_count', 'step_calls', 'state_version']:
            check(int(row[key + '_before']) == before[key], f'{key} before mismatch')
            check(int(row[key + '_after']) == after[key], f'{key} after mismatch')
        check(after['step_calls'] - before['step_calls'] == int(result['executed']), 'Actual call count differs from executed')
        if not result['executed']:
            check(before == after and result['reward'] is None, 'Rejection must not advance state or create reward')
        if result['success']:
            check(after['step_count'] == before['step_count'] + 1, 'Successful step did not advance counter')
        for i, value in enumerate(request['action']):
            check(numeric_equal(row[f'action_{i:02d}'], value), 'CSV action differs from raw candidate')
        check((directory / row['main_trace_file']).is_file(), 'Referenced main trace is missing')
        if result['executed'] and after.get('observation') is not None:
            expected_states.append((after, 'step' if result['success'] else 'failed_step'))
        last_state = after
    check(len(states) == len(expected_states) == summary['observation_rows'], 'Observation rows must equal reset plus actual returned steps')
    for row, (state, source) in zip(states, expected_states):
        check(row['source'] == source, 'Observation source mismatch')
        check(int(row['step_count']) == state['step_count'], 'Observation step count mismatch')
        check(float(row['nominal_sim_time_min']) == state['step_count'] * summary['sampling_interval_minutes'], 'Observation time mismatch')
        for i, value in enumerate(state['observation']):
            check(numeric_equal(row[f'obs_{i:03d}'], value), 'CSV observation differs from raw state')
    check(sum(e['trace']['execution_result']['executed'] for e in transitions)
          + sum(e['after']['step_calls'] - e['before']['step_calls'] for e in exceptions)
          == summary['actual_step_calls'], 'Summary call count mismatch')
    check(sum(e['trace']['execution_result']['success'] for e in transitions) == summary['successful_steps'], 'Summary successful step count mismatch')
    return {'status': 'PASS', 'episode_id': summary['episode_id'], 'attempt_rows': len(rows),
            'observation_rows': len(states), 'actual_step_calls': summary['actual_step_calls'],
            'successful_steps': summary['successful_steps'], 'exceptions': len(exceptions),
            'experiment_status': summary['experiment_status']}


def validate_run(directory):
    directory = Path(directory)
    manifest = strict_json((directory / 'manifest.json').read_text(encoding='utf-8'))
    validations = [validate_episode(directory / 'episodes' / name) for name in manifest['episode_ids']]
    return {'status': 'PASS', 'run_id': manifest['run_id'], 'episodes': validations,
            'meaning': 'Saved CSV and raw JSONL are consistent; this is data verification, not production performance validation'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    args = parser.parse_args()
    print(json.dumps(validate_run(args.directory), ensure_ascii=False, indent=2))
