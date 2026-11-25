"""D read-only audit of B's 2026-11-25 handoff; no training or simulator calls."""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import contextlib
import io
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from l3_system.agents.monitoring_agent import MonitoringAgent
import numpy as np

B_SOURCE='db7fbe40ff2ca1b56178e58424b31e0035e663f2'
ZIP_SHA='e0c0cced1ff3adb94f3e2d6574a4f5607b1f7f6cebc5bc91c9f5fe728a102117'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def lines(p): return [json.loads(s) for s in Path(p).read_text().splitlines()]
def contained(root, name):
    p=(root/name).resolve()
    if not p.is_relative_to(root): raise ValueError('path escaped package')
    return p

def dataset(root):
    manifest=read(root/'manifest.json'); counts=collections.Counter();groups={};traces={};positive=images=0
    source_bounds=list(csv.DictReader((root.parent/'04_source_and_companions/docs/data_dictionary.csv').open()))
    gates={};gate_counts=collections.Counter();eligibility=collections.defaultdict(collections.Counter)
    for env,n,m in [('AtropineEnvGym',39,4),('MAbEnvGym',1970,9)]:
        action_rows=sorted([r for r in source_bounds if r['backend']==env and r['kind']=='action'],key=lambda r:int(r['index']))
        if [int(r['index']) for r in action_rows]!=list(range(m)):raise ValueError('action dictionary incomplete')
        action=SimpleNamespace(shape=(m,),low=np.array([float(r['model_low']) for r in action_rows]),high=np.array([float(r['model_high']) for r in action_rows]))
        gates[env]=MonitoringAgent(action,SimpleNamespace(shape=(n,)))
    for report in manifest['episodes']:
        eid=report['episode_id'];env=report['environment_id'];split=report['split'];group=report['group_id']
        if group in groups and groups[group]!=split:raise ValueError('source lineage crosses splits')
        groups[group]=split
        kind=report['fault']['kind'] if report['fault'] else 'clean';counts[env+'|'+split+'|'+kind]+=1
        folder=root/eid;steps=lines(folder/'steps.jsonl');samples=lines(folder/'samples.jsonl');previous=None
        if len(steps)!=report['horizon_steps'] or len(samples)!=len(steps):raise ValueError('missing steps')
        numeric=[]
        for index,(step,sample) in enumerate(zip(steps,samples),1):
            if step['state_version_after']!=step['state_version_before']+1 or previous is not None and step['state_version_before']!=previous:
                raise ValueError('nonconsecutive state versions')
            previous=step['state_version_after']
            if step['step_index']!=index or sample['step']!=index:raise ValueError('wrong time order')
            if sample['labels']['physical_health_label'] is not None:raise ValueError('fabricated health label')
            positive+=int(sample['labels']['sensor_fault_label']);images+=int(sample['visual']['path'] is not None)
            values=sample['delivered_observation'];mask=sample['missing_observation_mask']
            if any((v is None)!=bool(m) for v,m in zip(values,mask)):raise ValueError('missing mask mismatch')
            numeric.append(all(v is not None and math.isfinite(v) for v in values))
            if index>=4:
                eligibility[env]['windows']+=1
                eligibility[env]['finite_full_windows']+=int(all(numeric[-4:]))
                eligibility[env]['missing_or_nonfinite_windows']+=int(not all(numeric[-4:]))
            with contextlib.redirect_stdout(io.StringIO()):
                decision=gates[env].check_request({'task_id':eid+'-'+str(index),'action':step['candidate_action']},step['observation_before'])
            gate_counts[env+'|'+decision['decision']]+=1
        trace=hashlib.sha256(json.dumps([[s['observation_after'],s['executed_action']] for s in steps],sort_keys=True,separators=(',',':')).encode()).hexdigest()
        traces.setdefault((env,trace),[]).append({'episode_id':eid,'split':split})
    duplicates=[v for v in traces.values() if len(v)>1]
    if any(len({r['split'] for r in v})>1 for v in duplicates):raise ValueError('exact trace crosses splits')
    return {'episodes':len(manifest['episodes']),'steps':sum(e['recorded_steps'] for e in manifest['episodes']),
            'sensor_positive_steps':positive,'derived_images':images,'unique_initial_groups':len(groups),
            'split_and_fault_counts':dict(sorted(counts.items())),'exact_raw_trace_duplicates':duplicates,
            'full_numeric_window_eligibility':{k:dict(v) for k,v in eligibility.items()},
            'D_basic_gate_archived_action_replay':dict(gate_counts),'D_R03_or_learned_models_enabled':False,
            'semantic_sha256':manifest['semantic_sha256'],'source_independence_proven':False}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--zip',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--git-repo',type=Path,help='Optional local repository containing the recorded B commit')
    a=p.parse_args()
    root=a.root.resolve();source=root/'04_source_and_companions'
    if sha(a.zip)!=ZIP_SHA:raise ValueError('ZIP hash mismatch')
    expected=read(root/'SHA256SUMS.json')
    for name,want in expected.items():
        if sha(contained(root,name))!=want:raise ValueError('file checksum mismatch: '+name)
    validations={}
    for name in ('01_B05_final','02_B05_replay'):
        r=subprocess.run([sys.executable,'-B','-m','data_generation.validation',str(root/name)],cwd=source,capture_output=True,text=True)
        if r.returncode:raise ValueError('original B validator failed: '+r.stderr)
        validations[name]=json.loads(r.stdout)
    source_comparison=None
    if a.git_repo:
        tracked=set(subprocess.check_output(['git','ls-tree','-r','--name-only',B_SOURCE],cwd=a.git_repo,text=True).splitlines())
        same=0;ending_only=[];extra=[]
        for name in expected:
            prefix='04_source_and_companions/'
            if not name.startswith(prefix):continue
            relative=name[len(prefix):];actual=(source/relative).read_bytes()
            if relative not in tracked:extra.append(relative);continue
            original=subprocess.check_output(['git','show',B_SOURCE+':'+relative],cwd=a.git_repo)
            if original==actual:same+=1
            elif original.replace(b'\r\n',b'\n')==actual.replace(b'\r\n',b'\n'):ending_only.append(relative)
            else:raise ValueError('source snapshot differs from B commit: '+relative)
        source_comparison={'byte_identical_tracked_files':same,'line_ending_only':ending_only,'additional_companions':extra,'other_tracked_differences':[]}
    data=dataset(root/'01_B05_final')
    if validations['01_B05_final']['semantic_sha256']!=validations['02_B05_replay']['semantic_sha256']:
        raise ValueError('replay semantics differ')
    profiles=[]
    for row in csv.DictReader((root/'03_B07_profile/performance.csv').open()):
        summary=read(root/'03_B07_profile'/row['case']/'summary.json')
        if row['status']!=summary['status'] or int(row['steps'])!=summary['step_calls'] or not math.isclose(float(row['wall_s']),summary['wall_seconds'],rel_tol=1e-10):
            raise ValueError('performance CSV disagrees with raw summary')
        profiles.append({'case':row['case'],'status':summary['status'],'reason':summary['reason'] if 'reason' in summary else None,
            'step_calls':summary['step_calls'],'successful_steps':summary['successful_steps'],
            'wall_seconds':summary['wall_seconds'],'phases':summary['phases'],
            'R03_runtime_enabled':summary['R03_runtime_enabled'],'expected_outcome_met':summary['expected_outcome_met']})
    manifest=read(root/'03_B07_profile/manifest.json')
    for name,want in manifest['input_hashes'].items():
        if sha(contained(source,name))!=want:raise ValueError('B07 input differs')
    for name,want in manifest['source_hashes'].items():
        if sha(contained(source,name))!=want:raise ValueError('B07 source differs: '+name)
    report={'schema_version':'DB25IntakeAudit/1.0','archive_sha256':ZIP_SHA,'B_source_commit':B_SOURCE,
        'source_snapshot_comparison':source_comparison,'B05_accepted_for_offline_sensor_pipeline':True,
        'B08_training_ready':False,'verified_package_file_count':len(expected),'B_validators':validations,'dataset':data,'B07_profiles':profiles,
        'gateway':read(root/'03_B07_profile/gateway_withdrawal/summary.json'),
        'worker_exit':read(root/'03_B07_profile/worker_exit/summary.json'),
        'B07_recorded_input_and_source_hashes_verified':True,
        'new_simulator_step_calls':0,'model_training_performed':False,'formal_blind_test_started':False,
        'replay_counted_as_independent_samples':False,'physical_health_labels_available':False,
        'limitations':['file and trace integrity do not establish physical safety or independent field modalities',
            'B05 post-step detector-only injection is not a biased-control closed loop',
            'B07 small-sample timing excludes new D learned integration',
            'new candidate and holdout joint freeze remain required']}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'files_verified':len(expected),'dataset':data,'B07_cases':len(profiles),'new_simulator_steps':0},ensure_ascii=False))
if __name__=='__main__':main()
