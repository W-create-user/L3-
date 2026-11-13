"""Validate new R-03 evidence only; never enable a rule or train a model."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PACK=ROOT/'handoffs/B_to_D_20261112/r03_supplement'

def read(name):return json.loads((PACK/name).read_text())
def check(ok,msg):
    if not ok:raise ValueError(msg)

def main():
    manifest=read('manifest.json')['files']
    for name,want in manifest.items():
        check(hashlib.sha256((PACK/name).read_bytes()).hexdigest()==want,f'Hash mismatch: {name}')
    rule=read('rule_candidate.json')
    check(not rule['runtime_enabled'],'Unapproved rule enabled')
    check(rule['variables']['current']['index']==6 and rule['variables']['reference']['index']==5,'Wrong Atropine coordinate')
    for source in rule['source_evidence'].values():
        check(hashlib.sha256((ROOT/source['path']).read_bytes()).hexdigest()==source['file_sha256'],'Source evidence changed')
    for case in read('validation_cases.json'):
        check(case['actual']['status']==case['expected_candidate_status'],'Candidate case mismatch')
        check(case['actual']['operational_decision'] is None,'Candidate was used as an operational decision')
    data=read('sensor_benchmark/episodes.json');splits={s:set() for s in ['train','validation','test']};initial=[];count=positive=0;hashes=[]
    check(data['collector_sha256']==hashlib.sha256((ROOT/'experiments/collect_r03_sensor_evidence.py').read_bytes()).hexdigest(),'Collector code mismatch')
    for episode in data['episodes']:
        check(episode['complete'] and episode['reset_count']==1 and episode['recorded_steps']==30,'Incomplete/stitched episode')
        splits[episode['split']].add(episode['group_id']);initial.append(episode['initial_internal_state_sha256'])
        path=PACK/'sensor_benchmark'/episode['raw_trajectory']
        check(hashlib.sha256(path.read_bytes()).hexdigest()==episode['trajectory_sha256'],'Trajectory hash mismatch')
        rows=[json.loads(s) for s in path.read_text().splitlines()];check(len(rows)==31,'Incomplete horizon')
        canonical=[]
        for step,row in enumerate(rows):
            check(row.get('step')==step and row['sim_time_min']==step*10,'Discontinuous trajectory')
            check(row['process_health_label'] is None,'Physical labels fabricated')
            raw=row['environment_observation'];delivered=row['delivered_observation']
            check(len(raw)==39 and len(delivered)==39,'Wrong observation shape')
            active=episode['split']!='train' and 10<=step<=20
            check(row['sensor_fault_label']==int(active),'Label not grounded in injection')
            for index in range(39):
                check(delivered[index]==raw[index]*(1.2 if active and index==6 else 1),'Unexpected data alteration')
            if step:
                check(not row['done'] and row['source_step_count']==step,'Unexpected terminal or advancement')
                check(row['requested_action']==row['actual_action']==[0.,0.,0.,0.],'Unexpected intervention')
                count+=1;positive+=int(active)
            canonical.append(raw)
        hashes.append(hashlib.sha256(json.dumps(canonical).encode()).hexdigest())
    check(not splits['train']&splits['validation'] and not splits['train']&splits['test'] and not splits['validation']&splits['test'],'Group leakage')
    check(len(set(initial))==12 and len(set(hashes))==12,'Duplicated initial state or complete trajectory')
    check(count==360 and positive==66,'Unexpected dataset counts')
    check(read('physical_training_gap.json')['eligible_confirmed_physical_process_episodes']==0,'Unconfirmed old labels became normal')
    print(f'Validated {len(manifest)} hashes, 1 disabled rule candidate, 2 constructed cases, 12 independent simulator episodes / 360 steps / 66 sensor-fault positives. Physical-process training remains blocked.')

if __name__=='__main__':main()
