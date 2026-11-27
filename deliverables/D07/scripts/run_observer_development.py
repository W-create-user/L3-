"""D development: B's unchanged P controller plus trusted-startup observer, old seeds."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import numpy as np
from l3_system.monitoring.buffer_observer import BufferObserver
from l3_system.execution.controllers import BufferLevelPID as BPolicy
from deliverables.D07.scripts import run_mab_paired as old
INSTANCES=[]
class ObserverPolicy:
    def __init__(self,adapter,config):
        self.adapter=adapter;self.base=BPolicy(adapter,config);self.events=[];self.action=None
        self.observer=BufferObserver(startup_level_dm=float(adapter.get_observation()[17]*adapter._xscale[17]),startup_trusted=True,episode_id=adapter.metadata['episode_id'],state_version=adapter.state_version)
        INSTANCES.append(self)
    def predict(self,observation,space):
        obs=np.asarray(observation,dtype=float).copy()
        d=self.observer.propose(float(obs[17]*self.adapter._xscale[17]),episode_id=self.adapter.metadata['episode_id'],state_version=self.adapter.state_version)
        self.events.append({'state_version':self.adapter.state_version,**d,'measurement_dm':float(obs[17]*self.adapter._xscale[17])})
        obs[17]=d['estimated_level_dm']/self.adapter._xscale[17];self.action=self.base.predict(obs,space);return self.action
    def commit(self,executed):
        self.base.commit(executed)
        self.observer.commit(executed=executed,physical_action=self.adapter.action_coordinates.to_physical(self.action),state_version=self.adapter.state_version)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--trusted-startup-for-development',action='store_true');a=p.parse_args()
    if not a.trusted_startup_for_development:p.error('explicit pre-fault startup assumption required')
    a.output.mkdir(parents=True,exist_ok=False);plan=old.make_plan()
    plan.update(schema_version='DObserverDevelopment/1.0',scope='old-seed development, D observer around unchanged B policy, not joint-approved B candidate',observer_assumptions={'startup':'B reset measurement before prediction-only corruption','model':'dh/dt=(executed u3-u7)/Ac, all flows known','unsupported':'arbitrary initial bias, unmeasured flows, model mismatch'},arms={'traditional_controller':'frozen P','single_agent':'B P + D observer, no A/C','four_agent':'same B P + observer + A/C single resource'},formal_blind_test=False)
    selected={'nominal':26001,'sensor_fault':26011,'process_disturbance':26021};plan['runs']=[s for s in plan['runs'] if selected[s['condition']]==s['seed']]
    plan['candidate_sha256']={str(p):old.sha(p) for p in [Path(__file__),Path('l3_system/monitoring/buffer_observer.py')]}
    old.write(a.output/'development_plan.json',plan);old.BufferLevelPID=ObserverPolicy;rows=[]
    for spec in plan['runs']:
        count=len(INSTANCES);row=old.run_episode(spec,plan,a.output);rows.append(row)
        if len(INSTANCES)>count:old.write(a.output/spec['run_id']/'observer_trace.json',INSTANCES[-1].events)
        old.write(a.output/'development_ledger.json',rows);print(spec['condition'],spec['arm'],row['status'],row['step_calls'],flush=True)
    pairs={}
    for r in rows:pairs.setdefault(r['pair_id'],{})[r['arm']]=r
    for pair in pairs.values():
        if len({r.get('initial_internal_state_sha256') for r in pair.values()})!=1:raise ValueError('initial states differ')
    means={arm:float(np.mean([r['simulated_makespan_min'] for r in rows if r['arm']==arm])) if all(r['success'] for r in rows if r['arm']==arm) else None for arm in old.ARMS}
    old.write(a.output/'development_summary.json',{'scope':'old seeds, 3 paired conditions only','runs':rows,'mean_minutes':means,'ratio_of_means_improvement':None if any(v is None for v in means.values()) else 1-means['four_agent']/means['traditional_controller'],'B_candidate_jointly_accepted':False,'formal_blind_test':False,'physical_safety_certified':False});print(json.dumps(means))
if __name__=='__main__':main()
