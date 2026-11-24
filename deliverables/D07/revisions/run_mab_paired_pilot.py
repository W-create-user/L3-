"""D-owned preregistered mAb buffer-goal topology comparison.

Run in an isolated overlay with unchanged B03 execution code and A main goal
runtime. All three arms use the same source-derived proportional controller;
this isolates topology, not algorithm superiority or plant production capacity.
"""
import argparse,contextlib,hashlib,json,math,random,time,traceback
from pathlib import Path
from uuid import uuid4
import numpy as np
from l3_system.execution import MAbAdapter,ExecutionAgent
from l3_system.execution.controllers import BufferLevelPID
from l3_system.agents.monitoring_agent import MonitoringAgent
from l3_system.agents.process_rules import ProcessRuleEngine
from l3_system.runtime import MultiAgentSystem
from l3_system.planning.goal_runtime import run_numeric_goal,GoalCoordinates
from l3_system.planning.goals import evaluate_goal

ROOT=Path(__file__).resolve().parents[3]
ARMS=['traditional_controller','single_agent','four_agent']

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def make_plan():
 area=math.pi*(5/2)**2;rows=[];seeds={'nominal':range(26001,26011),'sensor_fault':range(26011,26021),'process_disturbance':range(26021,26031)}
 for condition,values in seeds.items():
  for seed in values:
   rng=np.random.default_rng(seed)
   offset=float(rng.uniform(-.03,.03)) if condition!='process_disturbance' else float(rng.choice([-1,1])*rng.uniform(.15,.30))
   for arm in ARMS:rows.append({'run_id':f'{condition}-{seed}-{arm}','pair_id':f'{condition}-{seed}','condition':condition,'seed':seed,'arm':arm,'initial_buffer_offset_dm':offset})
 random.Random(261124).shuffle(rows)
 files=['l3_system/runtime.py','l3_system/main.py','l3_system/planning/goal_runtime.py','l3_system/agents/scheduling_agent.py','l3_system/algorithms/cnp_protocol.py','l3_system/execution/mab_adapter.py','l3_system/execution/execution_agent.py','l3_system/execution/controllers/policies.py','vendor/smpl/smpl/envs/helpers/mab_helpers.py','deliverables/D07/scripts/run_mab_paired.py']
 return {'schema_version':'D07MAbPaired/1.0','scope':'isolated model buffer-goal recovery, not full production acceptance','frozen_before_runs':True,
  'conditions':{'nominal':'steady source state with seeded buffer offset in [-.03,.03] dm','sensor_fault':'same initial range; add .10 dm only to controller measurement during first 3 action predictions; never change goal truth or D observations','process_disturbance':'source-state initial buffer offset with random sign and magnitude [.15,.30] dm; not a continuous process disturbance'},
  'goal':{'variable':'buffer_height','target_dm':[14.95,15.05],'hold_poststeps':3,'max_steps':10,'wall_budget_seconds':90},
  'controller':{'setpoint_dm':15.,'kp_L_min_per_dm':area/60,'ki_L_min_per_dm_min':0.,'kd_L_min_min_per_dm':0.,'sampling_period_min':60.,'basis':'mab_helpers D_bf=5 dm, dh/dt=(F_in-F_out)/Ac, dt=60 min. D research proportional cancellation; not missing B private PI configuration.'},
  'arms':{'traditional_controller':'D stateless proportional formula through unchanged B executor and D gate','single_agent':'B BufferLevelPID with the same P-only parameters through B executor and D gate; no A/C','four_agent':'unchanged A numeric goal planner and C local CNP plus same B P policy and D model-envelope gate'},
  'fairness':'same initial internal state, candidate equations, measurement perturbation, goal, horizon and model gate; source C has one executor, so no multi-resource efficiency claim',
  'primary_metric':'success / all validly started runs; execution failure or timeout counts as failure',
  'efficiency_metric':'paired simulated minutes to goal, successful complete pairs separately counted; software wall time separate',
  'exclusions':'none for failures; initialization failures retained as not-started with reason and planned denominator',
  'energy_metric':None,'physical_safety_certified':False,'whole_project_joint_signoff':False,
  'source_sha256':{p:sha(ROOT/p) for p in files},'runs':rows}

class CapturedAdapter:
 def __init__(self,base):self.base=base;self.policy=None;self.traces=[]
 def __getattr__(self,key):return getattr(self.base,key)
 def step(self,action):
  result=self.base.step(action)
  if self.policy is not None:self.policy.commit(bool(result['success']))
  self.traces.append(result);return result

def definitions(b,run):
 variables={'buffer_height':{'source':'observation','index':17,'unit':'dm','coordinate':'physical_projection','bounds':{'lower':float(b.observation_space.low[17]*b._xscale[17]),'upper':float(b.observation_space.high[17]*b._xscale[17])}}}
 for i in range(9):variables[f'model_action_{i}']={'source':'action','index':i,'unit':'dimensionless_model_scaled','coordinate':'physical_divided_by_uscale','bounds':{'lower':float(b.action_space.low[i]),'upper':float(b.action_space.high[i])}}
 registry={'schema_version':'2.0-draft','environment_id':'MAbEnvGym','registry_version':'D24-buffer-model-1','evidence_ref':'mab_helpers h=x[17] dm, xscale[17]=3; live B03 native action bounds','observation_size':1970,'action_size':9,'variables':variables}
 goal={'schema_version':'2.0-draft','goal_id':run['run_id'],'environment_id':'MAbEnvGym','registry_version':registry['registry_version'],'variable':'buffer_height','unit':'dm','coordinate':'physical_projection','target_range':{'lower':14.95,'upper':15.05},'max_steps':10,'hold_steps':3,
  'allowed_actions':{n:{'unit':v['unit'],'coordinate':v['coordinate'],'range':v['bounds']} for n,v in variables.items() if v['source']=='action'}}
 config={'schema_version':'ProcessRules/1.0','enabled':True,'environment':'mAb','version':'D24-model-envelope-1','observation_shape':[1970],'action_shape':[9],
  'variables':{'h_model':{'source':'observation','index':17,'unit':'dm / xscale[17]','coordinate':'physical_divided_by_xscale'}},
  'rules':[{'rule_id':'R-03-MAB-MODEL-BUFFER','kind':'range','enabled':True,'priority':1,'variable':'h_model','lower':float(b.observation_space.low[17]),'upper':float(b.observation_space.high[17]),
   'basis':{'kind':'model_definition','reference':'live source model Box, not physical hazard bounds','reviewed_by':'D source audit','reviewed_on':'2026-11-24'},'suggestion':'stop isolated simulation and inspect context'}]}
 return registry,goal,config

def run_episode(spec,plan,output):
 folder=output/spec['run_id'];folder.mkdir();row={**spec,'started':False,'success':False,'status':'initializing','failure_reason':None,'step_calls':0,'simulated_makespan_min':None,'wall_seconds':None,'initial_state_sha256':None};start=time.perf_counter()
 try:
  with (folder/'stdout.txt').open('w') as log,contextlib.redirect_stdout(log),MAbAdapter(max_steps=12) as base:
   initial=base._env.steady_observations.copy();initial[17]+=spec['initial_buffer_offset_dm']/base._xscale[17]
   base.reset(seed=spec['seed'],initial_state=initial);row['started']=True
   row['initial_state_sha256']=hashlib.sha256(np.asarray(base.get_observation(),dtype=np.float64).tobytes()).hexdigest()
   b=CapturedAdapter(base);ex=ExecutionAgent(b,log_path=folder/'B_events.jsonl');registry,goal,rules=definitions(b,spec)
   context={'environment':'mAb','episode_id':base.metadata['episode_id'],'variables':{'h_model':{'unit':'dm / xscale[17]','coordinate':'physical_divided_by_xscale'}}}
   mon=MonitoringAgent(b.action_space,b.observation_space,process_rules=ProcessRuleEngine(rules),process_context_provider=lambda state,action:context)
   cfg=plan['controller'];source_policy=BufferLevelPID(b,cfg) if spec['arm']!='traditional_controller' else None;b.policy=source_policy
   measurements=[]
   def predict(obs,space,attempt=0):
    delivered=np.asarray(obs,dtype=float).copy();bias=.10 if spec['condition']=='sensor_fault' and base.step_count<3 else 0.
    delivered[17]+=bias/base._xscale[17];measurements.append({'before_step':base.step_count,'bias_dm':bias,'delivered_h_dm':float(delivered[17]*base._xscale[17])})
    if source_policy is not None:return source_policy.predict(delivered,space)
    physical=b.action_coordinates.to_physical(base._env.steady_actions);physical[7]+=cfg['kp_L_min_per_dm']*(delivered[17]*base._xscale[17]-15.)
    low=b.action_coordinates.to_physical(space.low)[7];high=b.action_coordinates.to_physical(space.high)[7];physical[7]=np.clip(physical[7],low,high)
    return b.action_coordinates.from_physical(physical).astype(space.dtype)
   if spec['arm']=='four_agent':
    system=MultiAgentSystem(executor=ex,monitor=mon,event_path=folder/'A_C_events.jsonl')
    result=run_numeric_goal(system,goal,registry,environment='MAbEnvGym',timeout_seconds=90,policy=predict)
    success=result['summary']['status']=='succeeded';status=result['summary']['status'];reason=result['summary']['reason']
    row['C_resources_clear']=not bool(system.scheduler.active_agent_tasks);write(folder/'A_goal_result.json',result)
   else:
    episode=str(uuid4());samples=[];coords=GoalCoordinates(b,registry);deadline=time.monotonic()+90
    def sample(safety=True):return dict(goal_id=goal['goal_id'],environment_id='MAbEnvGym',registry_version=registry['registry_version'],episode_id=episode,step=base.step_count,observation=coords.observation(ex.get_observation()),safety_ok=safety)
    samples.append(sample(mon.check_observation(ex.get_observation())['approved']))
    while True:
     evaluation=evaluate_goal(samples[-1],samples[:-1],goal,registry)
     if evaluation['status']!='pending':break
     if time.monotonic()>=deadline:evaluation.update(status='failed',reason='wall_deadline');break
     action=predict(ex.get_observation(),b.action_space)
     result=ex.execute_assignment({'task_id':f"{spec['run_id']}-{base.step_count}",'task_type':'control_cycle'},mon.check_request,action=action)
     if not result['success'] or not result['executed']:evaluation.update(status='failed',reason=result.get('failure_reason'));break
     samples.append(sample(mon.check_observation(ex.get_observation())['approved']))
    success=evaluation['status']=='succeeded';status=evaluation['status'];reason=evaluation['reason'];write(folder/'direct_goal_result.json',{'evaluation':evaluation,'samples':samples})
   mon.export_logs(folder/'D_reviews.jsonl');write(folder/'steps.json',b.traces);write(folder/'measurement_interventions.json',measurements)
   row.update(success=bool(success),status=status,failure_reason=reason,step_calls=base.step_calls,
    simulated_makespan_min=base.step_count*float(base.metadata['sampling_period']['value']) if success else None,
    final_buffer_height_dm=float(ex.get_observation()[17]*base._xscale[17]),pending_requests_clear=not bool(ex._pending))
 except Exception:
  row.update(status='exception',failure_reason=traceback.format_exc())
 finally:
  row['wall_seconds']=time.perf_counter()-start;write(folder/'run_summary.json',row)
 return row

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--pilot',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 plan=make_plan()
 if a.pilot:plan['runs']=[next(r for r in plan['runs'] if r['arm']==arm and r['condition']=='process_disturbance') for arm in ARMS];plan['scope']='pilot configuration verification, not formal dataset'
 write(a.output/'frozen_plan.json',plan);rows=[]
 for i,spec in enumerate(plan['runs'],1):
  row=run_episode(spec,plan,a.output);rows.append(row);write(a.output/'run_ledger.json',rows)
  print(i,len(plan['runs']),row['run_id'],row['status'],row['step_calls'],flush=True)
 pairs={}
 for row in rows:pairs.setdefault(row['pair_id'],[]).append(row)
 complete=[v for v in pairs.values() if len(v)==3]
 for pair in complete:
  if len({r['initial_state_sha256'] for r in pair})!=1:raise ValueError('initial state pairing mismatch')
 groups={arm:{'planned':sum(r['arm']==arm for r in plan['runs']),'started':sum(r['arm']==arm and r['started'] for r in rows),'success':sum(r['arm']==arm and r['success'] for r in rows),'failures':sum(r['arm']==arm and r['started'] and not r['success'] for r in rows)} for arm in ARMS}
 report={'schema_version':'D24MAbSystemComparison/1.0','pilot':a.pilot,'plan_sha256':sha(a.output/'frozen_plan.json'),'runs':rows,'groups':groups,'paired_groups':len(complete),'all_initial_states_paired':True,'physical_safety_certified':False,'production_acceptance':False,'independent_episode_assumption_verified':False,'limits':plan['fairness']}
 write(a.output/'summary.json',report);print(json.dumps({'groups':groups,'paired_groups':len(complete)}))
if __name__=='__main__':main()
