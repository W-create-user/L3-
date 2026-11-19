"""Read-only B sensor archives through D rules and actual A/B/C protocol code.

Recorded step indices are replay-local versions, not fresh live B state. No
simulator is instantiated. step/reset are traps, never archive advancement.
"""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import numpy as np
from l3_system.agents.monitoring_agent import MonitoringAgent
from l3_system.agents.process_rules import ProcessRuleEngine
from l3_system.execution import ExecutionAgent
from l3_system.runtime import MultiAgentSystem
from .atropine_experiment import experiment_config,AUTHORIZATION
from .runtime_compatibility import require_supported_runtime

ACTION_DICTIONARY_SHA256='0d680df0821d7dda1d67637f4dfe5ceef7888a7ab21fe904a3a417d40f53c14c'


class ReplayBox:
    def __init__(self,low,high):
        self.low=np.asarray(low,dtype=np.float32);self.high=np.asarray(high,dtype=np.float32)
        self.shape=(4,);self.dtype=np.dtype('float32')
        if self.low.shape!=(4,) or self.high.shape!=(4,) or not np.isfinite(self.low).all() or not np.isfinite(self.high).all() or (self.low>self.high).any():raise ValueError('verified four-axis model bounds required')
    def contains(self,value):
        values=np.asarray(value)
        return bool(values.shape==(4,) and values.dtype.kind in 'fiu' and np.isfinite(values).all() and (values>=self.low).all() and (values<=self.high).all())


class ReadOnlyReplayAdapter:
    def __init__(self,episode_id,rows,low,high):
        self.source_dir='READ_ONLY_B_SENSOR_ARCHIVE_NOT_A_LIVE_MODEL'
        self.episode_id=episode_id;self.rows=rows;self.action_space=ReplayBox(low,high)
        self.observation_space=SimpleNamespace(shape=(39,));self._index=None;self.step_attempts=0;self.reset_attempts=0
    def select_record(self,index):
        if type(index)is not int or not 0<=index<len(self.rows):raise ValueError('archive index invalid')
        if self._index is not None and index<=self._index:raise ValueError('replay reader cannot rewind or duplicate archive advancement')
        row=self.rows[index]
        values=np.asarray(row['delivered_observation'])
        if row['episode_id']!=self.episode_id or row['step']!=index or values.shape!=(39,) or values.dtype.kind not in 'fiu' or not np.isfinite(values).all():raise ValueError('mixed/invalid archived record')
        self._index=index
    @property
    def state_version(self):self.require_ready();return self.rows[self._index]['step']
    @property
    def step_count(self):return self.state_version
    @property
    def step_calls(self):return 0
    @property
    def done(self):self.require_ready();return self.rows[self._index].get('done',False)
    def require_ready(self):
        if self._index is None:raise RuntimeError('select an existing archive record first')
    def get_observation(self):self.require_ready();return list(self.rows[self._index]['delivered_observation'])
    def coerce_candidate(self,action):
        values=np.asarray(action)
        if values.shape!=(4,) or values.dtype.kind not in 'fiu' or not np.isfinite(values).all():raise ValueError('invalid replay-only candidate')
        converted=values.astype(self.action_space.dtype)
        if not np.isfinite(converted).all():raise ValueError('unrepresentable replay candidate')
        return converted
    def validate_action(self,action):
        converted=self.coerce_candidate(action)
        if not self.action_space.contains(converted):raise ValueError('candidate outside model-coordinate Box')
        return converted
    def reset(self,*args,**kwargs):self.reset_attempts+=1;raise RuntimeError('archive replay cannot reset a simulator')
    def step(self,*args,**kwargs):self.step_attempts+=1;raise RuntimeError('archive replay cannot execute any action')


def _context(episode_id):
    return {'environment':'Atropine','episode_id':episode_id,
            'variables':{name:{'unit':'dimensionless','coordinate':'model_observation_normalize_false'} for name in ('E_current','E_reference')}}


def exercise_episode(episode,rows,low,high,output_dir):
    """Caller verifies archive provenance. A replay is not a new real episode."""
    adapter=ReadOnlyReplayAdapter(episode['episode_id'],rows,low,high)
    monitor=MonitoringAgent(adapter.action_space,adapter.observation_space,
        process_rules=ProcessRuleEngine(experiment_config(),allow_experimental=True),
        process_context_provider=lambda state,action:_context(adapter.episode_id))
    first_hit=None;hits=0
    # Audit all archived states with pure D rules; never sends passing actions.
    for index in range(1,len(rows)):
        values=rows[index]['delivered_observation']
        violations=monitor.process_rules.evaluate(values,None,_context(adapter.episode_id))
        if violations:
            hits+=1
            if first_hit is None:first_hit=index
    if first_hit is None:
        return {'episode_id':adapter.episode_id,'split':episode['split'],'archived_step_records':len(rows)-1,
                'candidate_trigger_records':0,'takeover_flow_exercised':False,'new_simulator_steps':0,
                'step_attempts':0,'reset_attempts':0,'scope':'offline_archive_only'}
    if first_hit<=1:raise ValueError('replay needs a prior valid archive state to test the old-approval path')
    adapter.select_record(first_hit-1)
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=False)
    executor=ExecutionAgent(adapter,output_dir/'B-replay-private.jsonl')
    prior=executor.prepare_action({'task_id':'replay-prior-'+adapter.episode_id,'task_type':'control_cycle'},action=[0.]*4)
    approval=monitor.check_request(prior,adapter.get_observation())
    if approval['decision']!='pass':raise ValueError('prior record not eligible for pending-approval protocol test')
    adapter.select_record(first_hit)
    system=MultiAgentSystem(executor=executor,monitor=monitor,event_path=output_dir/'A-events-private.jsonl',max_action_regenerations=0)
    result=system.run_cycle()
    if system.state!='takeover_requested' or result['summary']['step_delta']!=0 or adapter.step_attempts:
        raise RuntimeError('replay trigger did not stop before low-level execution')
    blocked_cycle=False
    try:system.run_cycle()
    except RuntimeError:blocked_cycle=True
    # Assert rejection is B's stale-version check, not just the read-only step trap.
    stale_blocked=False
    try:executor.execute_action(prior,approval)
    except ValueError as exc:stale_blocked='Environment changed' in str(exc)
    if not stale_blocked or adapter.step_attempts:raise RuntimeError('old approval reached an execution path')
    cancelled=executor.cancel_action_request(prior['request_id'],'offline takeover disposed unexecuted prepared request')
    replay_blocked=False
    try:executor.execute_action(prior,approval)
    except ValueError as exc:replay_blocked='Unknown or already consumed' in str(exc)
    clear=not system.scheduler.active_agent_tasks and not executor._pending
    if not all((blocked_cycle,cancelled,replay_blocked,clear)) or adapter.step_attempts or adapter.reset_attempts:
        raise RuntimeError('takeover/cleanup/replay invariant failed')
    # Deliberately do not call A.resume: replay rows cannot satisfy the live D
    # facade's requirement for fresh real recovery states.
    return {'episode_id':adapter.episode_id,'split':episode['split'],'archived_step_records':len(rows)-1,
        'candidate_trigger_records':hits,'first_archived_trigger_step':first_hit,'takeover_flow_exercised':True,
        'state':system.state,'new_cycle_blocked':blocked_cycle,'old_approval_stale_blocked':stale_blocked,
        'cancelled_request_replay_blocked':replay_blocked,'local_resources_and_pending_clear':clear,
        'new_simulator_steps':0,'step_attempts':adapter.step_attempts,'reset_attempts':adapter.reset_attempts,
        'source_integrity_verified_by_caller':False,'scope':'offline_archive_only_not_physical_anomaly_truth',
        'archived_step_cannot_be_undone':True}


def run_reviewed(package_root,evidence_root,action_dictionary,output_dir):
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'deliverables/D01/scripts'))
    from review_r03_supplement import review,safe_path
    runtime_review=require_supported_runtime()
    _,receipt=review(package_root,evidence_root)
    raw=Path(action_dictionary).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=ACTION_DICTIONARY_SHA256:raise ValueError('fixed B action dictionary mismatch')
    dictionary=[e for e in json.loads(raw) if e['backend']=='AtropineEnvGym']
    if [e['index'] for e in dictionary]!=list(range(4)) or any(e['normalize'] is not False or e['request_scale']!='delta_from_USS' for e in dictionary):raise ValueError('wrong model-coordinate action dictionary')
    low=[e['low'] for e in dictionary];high=[e['high'] for e in dictionary]
    root=Path(package_root).resolve();output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=False)
    episodes=json.loads((root/'sensor_benchmark/episodes.json').read_text())['episodes'];reports=[]
    for episode in episodes:
        path=safe_path(root/'sensor_benchmark',episode['raw_trajectory'])
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        entry=exercise_episode(episode,rows,low,high,output_dir/episode['episode_id'])
        entry['source_integrity_verified_by_caller']=True
        reports.append(entry)
    _,after=review(package_root,evidence_root)
    if receipt!=after or hashlib.sha256(Path(action_dictionary).read_bytes()).hexdigest()!=ACTION_DICTIONARY_SHA256:
        raise RuntimeError('source evidence changed during read-only replay')
    report={'schema_version':'AtropineTakeoverArchiveReplay/1.0','authorization_commit':AUTHORIZATION,
        'actor_code_sha256':{name:hashlib.sha256((Path(__file__).resolve().parents[1]/name).read_bytes()).hexdigest()
          for name in ('runtime.py','main.py','execution/execution_agent.py','execution/safety_bridge.py','agents/scheduling_agent.py','agents/process_rules.py','monitoring/atropine_replay.py')},
        'A_runtime_compatibility':runtime_review,'input_receipt':receipt,'action_dictionary_sha256':ACTION_DICTIONARY_SHA256,'episodes':reports,
        'input_integrity_verified':True,'source_package_unchanged':True,'new_simulator_steps':0,'live_recovery_exercised':False,
        'limits':'Archived steps were already executed before this replay. Versions here are replay-local record indices, not current live B state. Normal archive states are reviewed only; no passing action is executed. Takeover and stale/cancelled-request probes invoke actual A/B/C/D code on a permanently non-executing archive adapter.'}
    (output_dir/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


if __name__=='__main__':
    import argparse,contextlib,io
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-root',type=Path,required=True);parser.add_argument('--evidence-root',type=Path,required=True)
    parser.add_argument('--action-dictionary',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with contextlib.redirect_stdout(io.StringIO()):result=run_reviewed(args.package_root,args.evidence_root,args.action_dictionary,args.output)
    print(json.dumps({'archive_episodes':len(result['episodes']),'takeover_replays':sum(e['takeover_flow_exercised'] for e in result['episodes']),
        'new_simulator_steps':0,'source_package_unchanged':True,'live_recovery_exercised':False}))
