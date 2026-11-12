"""Validate the B-to-D handoff without requiring simulator dependencies."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / 'handoffs/B_to_D_20261112'


def read(name):
    return json.loads((PACK / name).read_text(encoding='utf-8'))


def require(value, message):
    if not value:
        raise ValueError(message)


def main():
    manifest = read('manifest.json')['files']
    for name, expected in manifest.items():
        require(hashlib.sha256((PACK/name).read_bytes()).hexdigest() == expected, f'Hash mismatch: {name}')
    require(hashlib.sha256((PACK/'observation_schema.json').read_bytes()).hexdigest()
            == read('provenance.json')['original_schema_sha256'], 'Original schema modified')
    fields = read('observation_schema.json')['fields']
    require([f['index'] for f in fields] == list(range(39)), 'Observation schema indices invalid')
    actions = read('action_dictionary.json')
    for backend, dimension in [('AtropineEnvGym', 4), ('MAbEnvGym', 9)]:
        require([a['index'] for a in actions if a['backend'] == backend] == list(range(dimension)), 'Action dictionary incomplete')
    configurations = read('run_configs.json')
    count = 0
    for config in configurations:
        events = [json.loads(line) for line in (PACK/config['raw_path']/'B_events.jsonl').read_text().splitlines()]
        reset = next(e for e in events if e['event'] == 'environment_reset')
        require(reset['seed'] == config['seed'], 'Seed mismatch')
        require(len(reset['observation']) == 39, 'Reset dimension mismatch')
        require(reset['observation'] == config['initial_observation'], 'Initial observation mismatch')
        count += sum(e['event'] == 'execution_result' and e['result']['executed'] for e in events)
    trajectories = [json.loads(line) for line in (PACK/'trajectories.jsonl').read_text().splitlines()]
    previous_steps = {}
    for row in trajectories:
        if row['kind'] == 'reset':
            previous_steps[row['run_id']] = 0
            continue
        raw = [json.loads(line) for line in (PACK/row['source']).read_text().splitlines()]
        result = raw[row['source_line']-1]['result']
        require(result == row['result'], 'Derived trajectory does not match raw event')
        requests = {e['action_request']['request_id']: e['action_request'] for e in raw if e['event'] == 'action_prepared'}
        require(row['requested_action'] == requests[result['request_id']]['action'], 'Request action mismatch')
        require(len(row['previous_observation']) == 39, 'Previous observation dimension mismatch')
        require(len(result['observation']) == 39 and all(math.isfinite(v) for v in result['observation']), 'Invalid observation')
        previous_steps[row['run_id']] += int(result['executed'])
        require(row['step_count'] == previous_steps[row['run_id']], 'Rejected request advanced derived time')
        require(row['step_count'] == result['step_count'], 'Raw and derived step count mismatch')
        require(row['sim_time_min'] == row['step_count']*10, 'Invalid simulation time')
        if result['executed']:
            require(row['actual_executed_action'] == result['action'], 'Executed action mismatch')
        else:
            require(row['actual_executed_action'] is None and result['step_calls'] == 0, 'Rejected action was executed')
    require(count == 8, 'Unexpected number of successful steps')
    rules = {r['id']: r for r in read('rule_thresholds.json')['rules']}
    require(rules['R-03']['status'] == 'unconfirmed_disabled' and rules['R-03']['threshold'] is None, 'Unconfirmed threshold presented as confirmed')
    mab = read('runs/mab_probe/report.json')
    require(mab['success'] and mab['spaces'] == {'observation': [1970], 'action': [9]}, 'mAb gate evidence invalid')
    require(mab['step']['step_count'] == 1, 'mAb step evidence missing')
    for entry in read('label_provenance.json'):
        require(entry['process_anomaly_label'] is None, 'Process anomaly ground truth was fabricated')
    print(f'Validated {len(manifest)} file hashes, original schema, 6 configurations, {count} successful steps, rejection, mAb probe and threshold provenance.')


if __name__ == '__main__':
    main()
