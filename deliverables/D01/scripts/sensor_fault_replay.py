"""Offline sensor fault replay on D01 observations; never imports a simulator.

Input vectors, injection labels, and integrity results are separate outputs.
No process labels, model predictions, training split or action authorization is made.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from l3_system.agents.safety_rules import observation_integrity_error
from intake_dataset import dictionary_errors, inspect_episode
from jsonschema import Draft202012Validator, FormatChecker


def parse(text):
    def invalid(value):raise ValueError(f'nonfinite JSON constant: {value}')
    return json.loads(text,parse_constant=invalid)


def sha(raw):return hashlib.sha256(raw).hexdigest()


def finite_number(value):return type(value) in (int,float) and math.isfinite(value)


def validate_spec(spec,dictionary,count):
    if not isinstance(spec,dict) or set(spec)!={'schema_version','mode','environment','source_episode_id',
            'source_lineage_group','source_split','source_sha256','dictionary_sha256','events'}:
        raise ValueError('invalid SensorReplay/0.1 specification fields')
    if spec['schema_version']!='SensorReplay/0.1' or spec['mode']!='offline_only' or spec['environment']!='Atropine':
        raise ValueError('only explicit offline Atropine replay is supported')
    if dictionary_errors(dictionary) or dictionary['environment']!='Atropine' or len(dictionary['observation'])!=39 or len(dictionary['action'])!=4:
        raise ValueError('reviewed Atropine 39/4 dictionary structure required')
    for key in ('source_episode_id','source_lineage_group'):
        if not isinstance(spec[key],str) or not spec[key].strip():raise ValueError('source episode/lineage required')
    if spec['source_split'] not in ('train','validation','test','excluded'):raise ValueError('invalid source split')
    for key in ('source_sha256','dictionary_sha256'):
        if not isinstance(spec[key],str) or not re.fullmatch('[0-9a-f]{64}',spec[key]) or spec[key]=='0'*64:
            raise ValueError('real source/dictionary hashes required')
    if not isinstance(spec['events'],list) or not spec['events']:raise ValueError('at least one replay event required')
    ids=set();occupied=set()
    base={'event_id','kind','variable_index','unit','coordinate','start_record','end_record','parameters'}
    for event in spec['events']:
        if not isinstance(event,dict) or set(event)!=base:raise ValueError('invalid replay event fields')
        eid=event['event_id']
        if not isinstance(eid,str) or not eid.strip() or eid in ids:raise ValueError('unique event_id required')
        ids.add(eid)
        kind=event['kind']
        if kind not in ('missing','stuck','bias','drift'):raise ValueError('unsupported sensor replay kind')
        index=event['variable_index'];start=event['start_record'];end=event['end_record']
        if type(index) is not int or not 0<=index<39:raise ValueError('invalid observation variable index')
        if type(start) is not int or type(end) is not int or not 0<=start<=end<count:
            raise ValueError('replay interval outside source episode')
        variable=dictionary['observation'][index]
        if event['unit']!=variable['unit'] or event['coordinate']!=variable['coordinate']:
            raise ValueError('event unit/coordinate differs from source dictionary')
        params=event['parameters']
        expected={'missing':set(),'stuck':{'reference_record'},'bias':{'offset'},'drift':{'increment_per_record'}}[kind]
        if not isinstance(params,dict) or set(params)!=expected:raise ValueError('invalid replay parameters')
        if kind=='stuck' and (type(params['reference_record']) is not int or not 0<=params['reference_record']<=start):
            raise ValueError('stuck reference cannot use a future record')
        for field in ('offset','increment_per_record'):
            if field in params and (not finite_number(params[field]) or params[field]==0):
                raise ValueError('bias/drift requires a nonzero finite parameter')
        for record in range(start,end+1):
            if (record,index) in occupied:raise ValueError('overlapping events on one variable require a separate replay')
            occupied.add((record,index))


def replay(rows,spec,dictionary):
    validate_spec(spec,dictionary,len(rows))
    for position,row in enumerate(rows):
        if (not isinstance(row,dict) or row.get('episode_id')!=spec['source_episode_id'] or row.get('step_index')!=position
                or observation_integrity_error(row.get('observation_before'),(39,)) is not None):
            raise ValueError('source must be a clean, ordered, single-episode D01 observation sequence')
    clean=[{'sample_index':i,'observation':deepcopy(row['observation_before']),'available_mask':[True]*39}
           for i,row in enumerate(rows)]
    faulty=deepcopy(clean)
    labels=[{'sample_index':i,'sensor_fault_active':False,'active_event_ids':[],
             'fault_scope':'synthetic_sensor_measurement','process_anomaly_label':None} for i in range(len(rows))]
    for event in spec['events']:
        index=event['variable_index'];start=event['start_record'];params=event['parameters']
        for position in range(start,event['end_record']+1):
            value=clean[position]['observation'][index]
            if event['kind']=='missing':
                value=None;faulty[position]['available_mask'][index]=False
            elif event['kind']=='stuck':value=clean[params['reference_record']]['observation'][index]
            elif event['kind']=='bias':value+=params['offset']
            else:value+=params['increment_per_record']*(position-start+1)
            if value is not None and not finite_number(value):raise ValueError('fault injection overflowed numeric representation')
            faulty[position]['observation'][index]=value
            labels[position]['sensor_fault_active']=True
            labels[position]['active_event_ids'].append(event['event_id'])
    checks=[]
    for before,after,label in zip(clean,faulty,labels):
        clean_error=observation_integrity_error(before['observation'],(39,))
        fault_error=observation_integrity_error(after['observation'],(39,))
        checks.append({'sample_index':before['sample_index'],'clean_integrity_rejected':clean_error is not None,
                       'replay_integrity_rejected':fault_error is not None,'reason':fault_error,
                       'measurement_changed':before['observation']!=after['observation']})
    summary={'mode':'offline_only','source_records':len(rows),
             'fault_active_records':sum(label['sensor_fault_active'] for label in labels),
             'changed_records':sum(row['measurement_changed'] for row in checks),
             'clean_integrity_rejections':sum(row['clean_integrity_rejected'] for row in checks),
             'replay_integrity_rejections':sum(row['replay_integrity_rejected'] for row in checks),
             'fault_active_integrity_passes':sum(label['sensor_fault_active'] and not check['replay_integrity_rejected']
                                                 for label,check in zip(labels,checks)),
             'new_simulator_calls':0,'R03_enabled':False,'anomaly_model_trained':False,
             'process_anomaly_predictions_available':False,'accepted_for_training':False,
             'limits':'Integrity pass is not sensor/process normality; injected mechanism may be active without changing the value.'}
    return {'clean_inputs':clean,'replay_inputs':faulty,'labels':labels,'integrity_checks':checks,'summary':summary}


def run(source_path,dictionary_path,spec_path,output):
    source_path=Path(source_path).resolve();dictionary_path=Path(dictionary_path).resolve()
    source_bytes=source_path.read_bytes();dictionary_bytes=dictionary_path.read_bytes()
    spec_bytes=Path(spec_path).read_bytes()
    spec=parse(spec_bytes.decode('utf-8-sig'))
    dictionary=parse(dictionary_bytes.decode('utf-8-sig'))
    lines=source_bytes.decode('utf-8-sig').splitlines()
    if not lines or any(not line.strip() for line in lines):raise ValueError('source must contain nonempty JSONL records')
    rows=[parse(line) for line in lines]
    validate_spec(spec,dictionary,len(rows))
    if sha(source_bytes)!=spec['source_sha256'] or sha(dictionary_bytes)!=spec['dictionary_sha256']:
        raise ValueError('source/dictionary hash mismatch')
    schema=parse((Path(__file__).resolve().parents[1]/'step_record.schema.json').read_text())
    validator=Draft202012Validator(schema,format_checker=FormatChecker())
    episode={'episode_id':spec['source_episode_id'],'step_count':len(rows),
             'artifacts':{'timeseries':[{'path':source_path.name}]}}
    inspected=inspect_episode(episode,source_path.parent,dictionary,validator)
    if inspected['errors']:raise ValueError('source intake failed: '+'; '.join(inspected['errors']))
    if sha(source_path.read_bytes())!=spec['source_sha256']:raise ValueError('source changed during intake')
    generated=replay(rows,spec,dictionary)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    for key in ('clean_inputs','replay_inputs','labels','integrity_checks'):
        (output/(key+'.jsonl')).write_text(''.join(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n'
                                                  for row in generated[key]),encoding='utf-8')
    provenance={'schema_version':'SensorReplayReceipt/0.1','mode':'offline_only',
                'source_episode_id':spec['source_episode_id'],'source_lineage_group':spec['source_lineage_group'],
                'source_split':spec['source_split'],'derived_lineage_group':spec['source_lineage_group'],
                'derived_split':spec['source_split'],'source_sha256':sha(source_bytes),
                'dictionary_sha256':sha(dictionary_bytes),'spec_sha256':sha(spec_bytes),
                'generator_sha256':sha(Path(__file__).read_bytes()),
                'output_sha256':{key+'.jsonl':sha((output/(key+'.jsonl')).read_bytes())
                                 for key in ('clean_inputs','replay_inputs','labels','integrity_checks')},
                'observation_coordinate':'observation_before only; sample_index counts interactions, not physical time',
                'drift_convention':'increment_per_record * (sample_index-start_record+1); not drift per second',
                'labels_are_separate_from_inputs':True,'accepted_for_training':False,
                'events':spec['events'],'lineage_correctness':'declared source lineage must still be independently reviewed'}
    for name,value in [('provenance.json',provenance),('summary.json',generated['summary'])]:
        (output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return generated['summary']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','dictionary','spec','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    try:summary=run(args.source,args.dictionary,args.spec,args.output)
    except (OSError,UnicodeError,ValueError,TypeError) as exc:parser.exit(1,f'Offline replay failed: {type(exc).__name__}: {exc}\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
