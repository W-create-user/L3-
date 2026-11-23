"""Receive the B-to-D 20261112 simulation package without inventing process labels.

Writes structural review copies only. Original raw files remain authoritative.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from jsonschema import Draft202012Validator, FormatChecker

from intake_dataset import dictionary_errors, inspect_episode, read_json


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def contained(root, relative):
    path=(root/relative).resolve()
    if not path.is_relative_to(root.resolve()):raise ValueError('source escapes handoff directory')
    return path


def verify_hashes(pack):
    manifest=read_json(pack/'manifest.json')
    if manifest['algorithm'].lower()!='sha256':raise ValueError('unsupported hash algorithm')
    for name,expected in manifest['files'].items():
        if hashlib.sha256(contained(pack,name).read_bytes()).hexdigest()!=expected:
            raise ValueError(f'hash mismatch: {name}')
    return len(manifest['files'])


def make_dictionary(schema, actions, source_commit):
    if schema['backend']!='AtropineEnvGym' or schema['dimension']!=39 or schema['normalize'] is not False:
        raise ValueError('only the explicit unnormalized Atropine schema is supported')
    fields=schema['fields']
    if [f['index'] for f in fields]!=list(range(39)):
        raise ValueError('Atropine observation fields must be ordered 0..38')
    action_fields=[a for a in actions if a['backend']=='AtropineEnvGym']
    if [a['index'] for a in action_fields]!=list(range(4)):
        raise ValueError('Atropine action fields must be ordered 0..3')
    if any(a['normalize'] is not False or a['request_scale']!='delta_from_USS' for a in action_fields):
        raise ValueError('action coordinates changed; refuse implicit rescaling')
    ref=f'GitHub:{source_commit}:handoffs/B_to_D_20261112/'
    dictionary={'schema_version':'D01Variables/0.1','environment':'Atropine',
                'observation':[{'index':f['index'],'name':f['name'],'unit':f['unit'],
                                'coordinate':'raw_observation_normalize_false',
                                'source_reference':ref+f'observation_schema_verified.json#fields/{f["index"]}'} for f in fields],
                'action':[{'index':a['index'],'name':a['name'],'unit':a['unit'],
                           'coordinate':a['request_scale'],
                           'source_reference':ref+f'action_dictionary.json#AtropineEnvGym/{a["index"]}'} for a in action_fields],
                'review':{'reviewed_by':'D automated structural review only; not physical-unit approval',
                          'reviewed_on':datetime.now(timezone.utc).date().isoformat(),
                          'source_reference':ref+'observation_schema_verified.json'}}
    problems=dictionary_errors(dictionary)
    if problems:raise ValueError('; '.join(problems))
    return dictionary


def disabled_rules(dictionary):
    variables={}
    for source in ('observation','action'):
        for field in dictionary[source]:
            variables[source+'::'+field['name']]={
                'source':source,'index':field['index'],'unit':field['unit'],'coordinate':field['coordinate']}
    return {'schema_version':'ProcessRules/1.0','enabled':False,'environment':'Atropine',
            'version':'B-to-D-20261112-structural-only','observation_shape':[39],'action_shape':[4],
            'variables':variables,'rules':[]}


def normalize_result(row, request, raw, position):
    result=row['result']
    if row['previous_observation']!=request['observation']:
        raise ValueError('previous observation differs from original prepared request')
    if row['requested_action']!=request['action'] or result['request_id']!=request['request_id']:
        raise ValueError('request/action identity mismatch')
    if result['task_id']!=request['task_id']:
        raise ValueError('task identity mismatch')
    if row['step_count']!=result['step_count'] or row['sim_time_min']!=result['step_count']*10:
        raise ValueError('derived simulation time/count differs from raw result')
    if request['environment']!='SMPL.AtropineEnvGym' or request['normalize'] is not False or request['uss_subtracted'] is not True:
        raise ValueError('Atropine environment/action mode mismatch')
    if type(result['executed']) is not bool or row['executed'] is not result['executed']:
        raise ValueError('inconsistent executed flag')
    executed=result['executed']
    if executed and row['actual_executed_action']!=result['action']:
        raise ValueError('actual executed action differs from raw result')
    if not executed and row['actual_executed_action'] is not None:
        raise ValueError('rejected candidate must not become an executed action')
    return {'schema_version':'D01Step/0.1','episode_id':'B-handoff::'+row['run_id'],'step_index':position,
            'simulation_time':row['sim_time_min']*60,'simulation_time_unit':'s',
            'collected_at':raw['time'],'observation_before':row['previous_observation'],
            'observation_after':result['observation'] if executed else None,
            'candidate_action':row['requested_action'],'executed_action':row['actual_executed_action'],
            'executed':executed,'state_version_before':request['state_version'],
            'state_version_after':result['state_version'],'reward':result['reward'] if executed else None,
            'terminated':result.get('terminated',False),'truncated':result.get('truncated',False),
            'execution_status':result['execution_status'],'failure_reason':result['failure_reason']}


def review(pack, output, source_commit):
    pack,output=Path(pack).resolve(),Path(output)
    hashes=verify_hashes(pack)
    source=read_json(pack/'provenance.json')
    if hashlib.sha256((pack/'observation_schema.json').read_bytes()).hexdigest()!=source['original_schema_sha256']:
        raise ValueError('original schema differs from declared original hash')
    schema=read_json(pack/'observation_schema_verified.json')
    actions=read_json(pack/'action_dictionary.json')
    dictionary=make_dictionary(schema,actions,source_commit)
    rules={r['id']:r for r in read_json(pack/'rule_thresholds.json')['rules']}
    if rules['R-03']['status']!='unconfirmed_disabled' or rules['R-03']['threshold'] is not None:
        raise ValueError('unexpected R-03 activation/threshold; manual review required')
    labels=read_json(pack/'label_provenance.json')
    if any(label['process_anomaly_label'] is not None for label in labels):
        raise ValueError('process labels changed; source-specific label review required')
    configurations={c['run_id']:c for c in read_json(pack/'run_configs.json')}
    raw_by_run={};normalized={};origins=[]
    for rid,config in configurations.items():
        if not re.fullmatch('[A-Za-z0-9_-]+',rid):raise ValueError('unsafe run_id')
        if config['backend']!='AtropineEnvGym' or config['sampling_period_min']!=10:
            raise ValueError('unexpected environment/sampling contract')
        events=[json.loads(line) for line in contained(pack,config['raw_path']+'/B_events.jsonl').read_text().splitlines()]
        reset=next(e for e in events if e['event']=='environment_reset')
        if reset['seed']!=config['seed'] or reset['observation']!=config['initial_observation']:
            raise ValueError('seed/initial-observation mismatch')
        raw_by_run[rid]=events
    for row in [json.loads(line) for line in (pack/'trajectories.jsonl').read_text().splitlines()]:
        if row['kind']=='reset':continue
        if row['kind']!='execution_result':raise ValueError('unknown trajectory row kind')
        rid=row['run_id'];events=raw_by_run[rid]
        expected_source=configurations[rid]['raw_path']+'/B_events.jsonl'
        if row['source']!=expected_source:raise ValueError('trajectory source path mismatch')
        original=events[row['source_line']-1]
        if original.get('event')!='execution_result' or original['result']!=row['result']:
            raise ValueError('derived trajectory differs from raw result')
        requests={e['action_request']['request_id']:e['action_request'] for e in events if e['event']=='action_prepared'}
        request=requests[row['result']['request_id']]
        if request['action_space']['low']!=rules['R-02']['low'] or request['action_space']['high']!=rules['R-02']['high']:
            raise ValueError('R-02 bounds differ from live request space')
        records=normalized.setdefault(rid,[])
        converted=normalize_result(row,request,original,len(records))
        records.append(converted)
        origins.append({'episode_id':converted['episode_id'],'step_index':converted['step_index'],
                        'source':row['source'],'source_line':row['source_line'],
                        'request_id':request['request_id'],'process_anomaly_label':None})
    expected_ids={origin['request_id'] for origin in origins}
    if len(expected_ids)!=len(origins) or {label['request_id'] for label in labels}!=expected_ids:
        raise ValueError('label source entries do not cover unique raw outcomes')
    for origin in origins:
        label=next(label for label in labels if label['request_id']==origin['request_id'])
        if label['source']!=origin['source'] or label['source_line']!=origin['source_line']:
            raise ValueError('label source reference does not match raw outcome')
    output.mkdir(parents=True,exist_ok=False)
    def write(name,value):
        (output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    write('observation_action_dictionary.structure_only.json',dictionary)
    write('process_rules.atropine.disabled.json',disabled_rules(dictionary))
    write('normalized_source_index.json',origins)
    row_schema=read_json(Path(__file__).resolve().parents[1]/'step_record.schema.json')
    row_validator=Draft202012Validator(row_schema,format_checker=FormatChecker())
    reports=[]
    for rid,rows in normalized.items():
        name=rid+'.jsonl'
        (output/name).write_text(''.join(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n' for row in rows))
        episode={'episode_id':'B-handoff::'+rid,'step_count':len(rows),'artifacts':{'timeseries':[{'path':name}]}}
        reports.append(inspect_episode(episode,output,dictionary,row_validator))
    probe=read_json(pack/'runs/mab_probe/report.json')
    mab_shape_ok=(probe['spaces']=={'observation':[1970],'action':[9]}
                  and len(probe['reset'])==1970 and len(probe['step']['observation'])==1970
                  and len(probe['requested_action'])==9 and probe['step']['step_count']==1 and probe['success'] is True)
    if not mab_shape_ok:raise ValueError('mAb one-step evidence shape/status mismatch')
    report={'reviewed_at':datetime.now(timezone.utc).isoformat(),'source_commit':source_commit,
            'runtime_code_commit':source['code_base_commit'],'verified_manifest_hashes':hashes,
            'package_received':True,'field_structure_passed':not any(r['errors'] for r in reports),
            'raw_atropine_configurations':len(configurations),'normalized_interaction_records':sum(len(r) for r in normalized.values()),
            'reported_executed_calls':sum(row['executed'] for rows in normalized.values() for row in rows),
            'read_only_runs':[rid for rid in configurations if rid not in normalized],
            'normalized_record_time':'end-of-interaction simulation time; source sim_time_min * 60, not obs[0]',
            'episodes':reports,'process_labels_available':False,'accepted_for_process_training':False,
            'R03_enabled':False,'mab_reset_and_one_step_evidence_received':mab_shape_ok,
            'mab_dictionary_or_process_rules_accepted':False,
            'uncertain_observation_fields':[{'index':f['index'],'unit':f['unit'],'unit_status':f['unit_status']}
                                            for f in schema['fields'] if f['unit_status'] in ('unconfirmed','inferred_from_source_conversion')],
            'notes':['No train/validation/test split or normal/process-anomaly label has been fabricated.',
                     'Raw rejected result observation is retained in B source; normalized observation_after is null because no step occurred.',
                     'Dictionary approval covers structure only; physical units/sensors/limits remain as declared by B.']}
    write('review.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pack',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-commit',required=True)
    args=parser.parse_args()
    if not re.fullmatch('[0-9a-f]{40}',args.source_commit):parser.error('source-commit requires a full Git SHA')
    report=review(args.pack,args.output,args.source_commit)
    print(json.dumps({k:report[k] for k in ('field_structure_passed','verified_manifest_hashes',
                     'normalized_interaction_records','reported_executed_calls','R03_enabled',
                     'accepted_for_process_training')},ensure_ascii=False))
    raise SystemExit(0 if report['field_structure_passed'] else 1)


if __name__=='__main__':main()
