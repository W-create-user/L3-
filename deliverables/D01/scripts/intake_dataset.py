"""Check delivered episode contents against D01 metadata and a supplied dictionary.

Does not infer physical units, labels, permissions or process safety from files.
"""
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

from validate_manifest import validate


def read_json(path):
    def invalid(value):
        raise ValueError(f'nonfinite JSON constant: {value}')
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), parse_constant=invalid)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dictionary_errors(dictionary):
    errors = []
    if not isinstance(dictionary, dict) or set(dictionary) != {
            'schema_version', 'environment', 'observation', 'action', 'review'}:
        return ['variable dictionary fields do not match D01Variables/0.1']
    if dictionary['schema_version'] != 'D01Variables/0.1':
        errors.append('unsupported variable dictionary version')
    if dictionary['environment'] not in ('Atropine', 'mAb'):
        errors.append('unsupported variable dictionary environment')
    for group in ('observation', 'action'):
        variables = dictionary[group]
        if not isinstance(variables, list) or not variables:
            errors.append(f'dictionary {group} must be a nonempty ordered list')
            continue
        names = set()
        for index, item in enumerate(variables):
            if not isinstance(item, dict) or set(item) != {'index', 'name', 'unit', 'coordinate', 'source_reference'}:
                errors.append(f'invalid dictionary entry: {group}[{index}]')
                continue
            if type(item['index']) is not int or item['index'] != index:
                errors.append(f'dictionary indices must be contiguous from zero: {group}')
            for key in ('name', 'unit', 'coordinate', 'source_reference'):
                if not isinstance(item[key], str) or not item[key].strip() or item[key].strip().upper() in ('PENDING', 'UNKNOWN', 'TODO', 'TBD'):
                    errors.append(f'unresolved dictionary {group}[{index}].{key}')
            if isinstance(item['name'], str):
                if item['name'] in names:
                    errors.append(f'duplicate dictionary variable name: {group}')
                names.add(item['name'])
    review = dictionary['review']
    if not isinstance(review, dict) or set(review) != {'reviewed_by', 'reviewed_on', 'source_reference'}:
        errors.append('dictionary review declaration required')
    else:
        for value in review.values():
            if not isinstance(value, str) or not value.strip():
                errors.append('dictionary review declaration contains empty values')
        try:
            datetime.strptime(review['reviewed_on'], '%Y-%m-%d')
        except (ValueError, TypeError):
            errors.append('dictionary reviewed_on requires YYYY-MM-DD')
    return errors


def _vector(value, length):
    return (isinstance(value, list) and len(value) == length
            and all(type(x) in (int, float) and math.isfinite(x) for x in value))


def inspect_episode(episode, root, dictionary, row_validator):
    errors, rows = [], []
    root = Path(root).resolve()
    for artifact in episode['artifacts']['timeseries']:
        path = (root/artifact['path']).resolve()
        if not path.is_relative_to(root):
            errors.append('timeseries file outside dataset root')
            continue
        try:
            lines = path.read_text(encoding='utf-8-sig').splitlines()
            for number, line in enumerate(lines, 1):
                if not line.strip():
                    errors.append(f'{artifact["path"]}:{number}: blank JSONL record')
                    continue
                def invalid(value):
                    raise ValueError(f'nonfinite JSON constant: {value}')
                try:
                    row = json.loads(line, parse_constant=invalid)
                    problems = list(row_validator.iter_errors(row))
                    if problems:
                        errors.append(f'{artifact["path"]}:{number}: row schema: {problems[0].message}')
                    else:
                        rows.append(row)
                except ValueError as exc:
                    errors.append(f'{artifact["path"]}:{number}: invalid JSON: {exc}')
        except (OSError, UnicodeError) as exc:
            errors.append(f'cannot read timeseries {artifact["path"]}: {type(exc).__name__}')
    if len(rows) != episode['step_count']:
        errors.append('record count differs from manifest step_count')
    previous, previous_time = None, None
    observation_size, action_size = len(dictionary['observation']), len(dictionary['action'])
    for index, row in enumerate(rows):
        prefix = f'{episode["episode_id"]} record {index}: '
        if row['episode_id'] != episode['episode_id']:
            errors.append(prefix+'episode_id mismatch')
        if row['step_index'] != index:
            errors.append(prefix+'step_index must be contiguous from zero across ordered shards')
        if not _vector(row['observation_before'], observation_size) or not _vector(row['candidate_action'], action_size):
            errors.append(prefix+'observation/candidate dimensions differ from dictionary')
        if row['observation_after'] is not None and not _vector(row['observation_after'], observation_size):
            errors.append(prefix+'observation_after dimensions differ from dictionary')
        if row['executed_action'] is not None and not _vector(row['executed_action'], action_size):
            errors.append(prefix+'executed_action dimensions differ from dictionary')
        if row['terminated'] and row['truncated']:
            errors.append(prefix+'terminated and truncated cannot both be true')
        if row['executed']:
            if row['execution_status'] == 'rejected' or row['executed_action'] is None:
                errors.append(prefix+'executed result requires actual action and cannot be rejected')
            if row['state_version_after'] != row['state_version_before']+1:
                errors.append(prefix+'executed call must advance state_version exactly once')
            if row['execution_status'] == 'completed' and (row['reward'] is None or row['observation_after'] is None):
                errors.append(prefix+'completed execution requires reward and observation_after')
        else:
            if (row['executed_action'] is not None or row['observation_after'] is not None
                    or row['reward'] is not None or row['execution_status'] == 'completed'
                    or row['state_version_after'] != row['state_version_before']
                    or row['terminated'] or row['truncated']):
                errors.append(prefix+'unexecuted candidate must not invent advancement/output/termination')
        if row['execution_status'] != 'completed' and not row['failure_reason']:
            errors.append(prefix+'rejected/failed result needs a failure reason')
        if row['execution_status'] == 'completed' and row['failure_reason'] is not None:
            errors.append(prefix+'completed result cannot carry failure_reason')
        if row['reward'] is not None and not math.isfinite(row['reward']):
            errors.append(prefix+'nonfinite reward')
        if not math.isfinite(row['simulation_time']):
            errors.append(prefix+'nonfinite simulation time')
        collected_time = None
        try:
            collected_time = datetime.fromisoformat(row['collected_at'].replace('Z','+00:00'))
            if collected_time.tzinfo is None:
                raise ValueError('timezone required')
        except ValueError:
            errors.append(prefix+'collection time must be an ISO timestamp with timezone')
            collected_time = None
        if previous is not None:
            if row['state_version_before'] != previous['state_version_after']:
                errors.append(prefix+'state_version discontinuity; reset/new episode must be separated')
            if row['simulation_time'] < previous['simulation_time']:
                errors.append(prefix+'simulation time moved backwards')
            if collected_time is not None and previous_time is not None and collected_time < previous_time:
                errors.append(prefix+'collection time moved backwards')
            if previous['terminated'] or previous['truncated'] or previous['execution_status'] == 'failed':
                errors.append(prefix+'record after terminal/failed result must start a new episode')
        previous = row
        previous_time = collected_time
    return {'episode_id': episode['episode_id'], 'valid_records': len(rows),
            'executed_calls': sum(r['executed'] for r in rows), 'errors': errors}


def inspect_dataset(manifest, root, config_path, dictionary_path):
    errors = validate(manifest, root)
    if errors:
        return {'passed': False, 'errors': errors, 'episodes': [], 'scope': 'metadata validation failed'}
    try:
        dictionary = read_json(dictionary_path)
        errors.extend(dictionary_errors(dictionary))
        # A valid JSON configuration is required; its physical meaning needs separate review.
        if not isinstance(read_json(config_path), dict):
            errors.append('collection configuration must be an object')
        if digest(config_path) != manifest['collection_config_sha256']:
            errors.append('collection configuration hash mismatch')
        if digest(dictionary_path) != manifest['variable_dictionary_sha256']:
            errors.append('variable dictionary hash mismatch')
    except (OSError, UnicodeError, ValueError) as exc:
        errors.append(f'cannot read collection inputs: {type(exc).__name__}')
    if errors:
        return {'passed': False, 'errors': errors, 'episodes': [], 'scope': 'collection input validation failed'}
    schema = read_json(Path(__file__).resolve().parents[1]/'step_record.schema.json')
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    reports = []
    for episode in manifest['episodes']:
        if episode['environment'] != dictionary['environment']:
            errors.append(f'environment does not match dictionary: {episode["episode_id"]}')
        report = inspect_episode(episode, root, dictionary, validator)
        reports.append(report)
        errors.extend(report['errors'])
    return {'passed': not errors, 'errors': errors, 'episodes': reports,
            'scope': 'structural intake only; permission/labels/physical units require human review'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest',type=Path)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--dictionary',type=Path,required=True)
    parser.add_argument('--report',type=Path)
    args=parser.parse_args()
    try:
        report=inspect_dataset(read_json(args.manifest),args.root,args.config,args.dictionary)
    except (OSError,UnicodeError,ValueError) as exc:
        report={'passed':False,'errors':[f'input read failed: {type(exc).__name__}']}
    rendered=json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if args.report:
        # Never clobber a previous review record or an input file.
        try:
            with args.report.open('x',encoding='utf-8') as stream:stream.write(rendered)
        except OSError as exc:
            parser.exit(1,f'Cannot write new report: {type(exc).__name__}\n')
    print(rendered,end='')
    raise SystemExit(0 if report['passed'] else 1)


if __name__=='__main__':
    main()
