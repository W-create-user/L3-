"""D model envelope and unit validation on unchanged B mAb, not physical safety."""
import argparse,contextlib,hashlib,json
from pathlib import Path
from l3_system.execution import MAbAdapter,ExecutionAgent
from l3_system.agents.monitoring_agent import MonitoringAgent
from l3_system.agents.process_rules import ProcessRuleEngine

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 root=Path(__file__).resolve().parents[3];source=root/'vendor/smpl/smpl/envs/helpers/mab_helpers.py'
 assert 'h = x[17]  # Liquid level in the buffer tank (dm)' in source.read_text()
 config={'schema_version':'ProcessRules/1.0','enabled':True,'environment':'mAb','version':'D-mAb-model-envelope-1','observation_shape':[1970],'action_shape':[9],
  'variables':{'buffer_level_scaled':{'source':'observation','index':17,'unit':'dm / xscale[17]','coordinate':'physical_divided_by_xscale'}},'rules':[]}
 with (a.output/'stdout.txt').open('w') as log,contextlib.redirect_stdout(log),MAbAdapter(max_steps=6) as b:
  ex=ExecutionAgent(b,log_path=a.output/'B_events.jsonl');ex.reset(seed=1610);assert b._xscale[17]==3.
  lo=float(b.observation_space.low[17]);hi=float(b.observation_space.high[17])
  config['rules']=[{'rule_id':'R-03-MAB-MODEL-BUFFER','kind':'range','enabled':True,'priority':1,
   'basis':{'kind':'model_definition','reference':'MAbEnvGym normalize=False; physical xlb/xub divided by xscale; not physical safety bounds','reviewed_by':'D source audit','reviewed_on':'2026-11-23'},
   'suggestion':'stop simulation and inspect model context; no physical shutdown claim','variable':'buffer_level_scaled','lower':lo,'upper':hi}]
  context={'environment':'mAb','episode_id':b.metadata['episode_id'],'variables':{'buffer_level_scaled':{'unit':'dm / xscale[17]','coordinate':'physical_divided_by_xscale'}}}
  monitor=MonitoringAgent(b.action_space,b.observation_space,process_rules=ProcessRuleEngine(config),process_context_provider=lambda state,action:context)
  cases=[];initial=ex.get_observation();action=b._env.steady_actions.copy()
  for name,value,expected in [('lower_boundary',lo,True),('upper_boundary',hi,True),('below_model_envelope',lo-1,False),('above_model_envelope',hi+1,False)]:
   observation=list(initial);observation[17]=value;result=monitor.check_action(observation,action);assert result['approved'] is expected
   cases.append({'name':name,'approved':result['approved'],'executed':False,'label_scope':'D boundary fixture, not physical anomaly'})
  for name,key in [('unit_mismatch','unit'),('coordinate_mismatch','coordinate')]:
   saved=json.loads(json.dumps(context));context['variables']['buffer_level_scaled'][key]='mismatched'
   result=monitor.check_action(initial,action);assert not result['approved'];cases.append({'name':name,'approved':False,'executed':False});context.clear();context.update(saved)
  steps=[]
  for i in range(3):
   result=ex.execute_assignment({'task_id':f'D-mab-{i}','task_type':'control_cycle'},monitor.check_request,action=action);assert result['success'] and result['executed']
   steps.append({'step_count':b.step_count,'state_version':b.state_version,'buffer_level_model':ex.get_observation()[17],'buffer_level_dm':ex.get_observation()[17]*3,'clipped_indices':result['info'].get('clipped_indices')})
  report={'schema_version':'DMAbModelEnvelope/1.0','passed':True,'cases':cases,'actual_mab_steps':3,'steps':steps,'config':config,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
   'model_buffer_bounds_scaled':[lo,hi],'physical_conversion':{'index':17,'scale':3,'physical_unit':'dm'},'B_code_modified':False,'physical_safety_R03_accepted':False,
   'limits':'Simulator bounds and clipped observations do not establish field hazard limits, raw physical-state safety, or physical anomaly truth.'}
 (a.output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({'passed':True,'actual_mab_steps':3,'boundary_cases':len(cases),'physical_safety_R03_accepted':False}))
if __name__=='__main__':main()
