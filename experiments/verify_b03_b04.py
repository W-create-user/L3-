"""Real B02/B03 gates and limited B04 research control, preserving failures."""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import traceback
import numpy as np
from l3_system.execution import MAbAdapter, EnvironmentAdapter, ExecutionAgent
from l3_system.execution.controllers import BufferLevelPID
from l3_system.agents.monitoring_agent import MonitoringAgent
from l3_system.planning.goals import evaluate_goal

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    config=json.loads((ROOT/'configs/controllers/mab_buffer_pi.json').read_text())
    registry=json.loads((ROOT/'configs/environments/mab_physical_registry.json').read_text())
    summary={'success':False,'research_only':True,'R03_runtime_enabled':False,'physical_safety_verified':False,
             'config_sha256':hashlib.sha256((ROOT/'configs/controllers/mab_buffer_pi.json').read_bytes()).hexdigest(),
             'runs':[],'limitations':['No approved mAb production goal or physical safety fence','PI is a research candidate, not a trained production controller','No MPC solver/controller acceptance']}
    def save():
        (args.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    def run(name,seed,policy=False,offset_dm=None):
        folder=args.output/name;folder.mkdir();report={'name':name,'seed':seed,'success':False};summary['runs'].append(report);save()
        with MAbAdapter(max_steps=10 if not policy else 12) as adapter:
            executor=ExecutionAgent(adapter,log_path=folder/'B_events.jsonl');executor.reset(seed=seed)
            monitor=MonitoringAgent(adapter.action_space,adapter.observation_space)
            controller=BufferLevelPID(adapter,config) if policy else None
            if policy:
                state=np.asarray(adapter.get_observation()).copy();state[17]+=(config['initial_buffer_offset_dm'] if offset_dm is None else offset_dm)/adapter._xscale[17]
                report['initial_buffer_offset_dm']=config['initial_buffer_offset_dm'] if offset_dm is None else offset_dm
                report['initial_state_sha256']=hashlib.sha256(state.astype(np.float32).tobytes()).hexdigest()
                adapter.reset(seed=seed,initial_state=state);controller.reset()
            report['backend_metadata']=adapter.metadata
            # Round trips with live coordinates, not guessed dimensions or units.
            model=adapter._env.steady_actions
            if not np.allclose(adapter.action_coordinates.from_physical(adapter.action_coordinates.to_physical(model)),model):raise RuntimeError('Coordinate round trip failed')
            original=adapter.step_calls
            bad=executor.prepare_action({'task_id':name+'_reject','task_type':'control_cycle'},action=adapter.action_space.high.astype(float)+1)
            review=monitor.check_request(bad,bad['observation']);executor.reject_action_request(bad,review)
            if adapter.step_calls!=original:raise RuntimeError('Rejected request advanced model')
            goal={'schema_version':'2.0-draft','goal_id':name+'_buffer_goal','environment_id':'MAbEnvGym',
                  'registry_version':registry['registry_version'],'variable':'buffer_height','unit':'dm','coordinate':'physical_projection',
                  'target_range':{'lower':config['setpoint_dm']-config['target_tolerance_dm'],'upper':config['setpoint_dm']+config['target_tolerance_dm']},
                  'max_steps':10,'hold_steps':config['hold_steps'],
                  'allowed_actions':{n:{'unit':v['unit'],'coordinate':v['coordinate'],'range':v['bounds']} for n,v in registry['variables'].items() if v['source']=='action'}}
            history=[];errors=[];last_request=None
            with (folder/'trajectory.jsonl').open('w') as f:
                for index in range(10):
                    obs=adapter.get_observation();action=controller.predict(obs,adapter.action_space) if controller else adapter._env.steady_actions
                    request=executor.prepare_action({'task_id':f'{name}_{index}','task_type':'control_cycle'},action=action)
                    if request['environment']!='SMPL.MAbEnvGym' or request['uss_subtracted'] is not False:raise RuntimeError('Wrong backend request metadata')
                    review=monitor.check_request(request,request['observation']);result=executor.execute_action(request,review)
                    if controller:controller.commit(bool(result['success']))
                    f.write(json.dumps({'index':index+1,'request':request,'result':result},allow_nan=False)+'\n');f.flush()
                    if not result['success']:raise RuntimeError('Real mAb step failed')
                    error=float(result['observation'][17]*adapter._xscale[17]-config['setpoint_dm']);errors.append(error)
                    if policy:
                        physical=(np.asarray(result['observation'])*adapter._xscale).tolist()
                        sample={'goal_id':goal['goal_id'],'environment_id':'MAbEnvGym','registry_version':registry['registry_version'],
                                'episode_id':adapter.metadata['episode_id'],'step':index+1,'observation':physical,
                                'safety_ok':bool(result['success'] and review['decision']=='pass')}
                        evaluation=evaluate_goal(sample,history,goal,registry);history.append(sample)
                    last_request=request
            before=adapter.step_calls
            try:executor.execute_action(last_request,review)
            except (ValueError,RuntimeError):pass
            else:raise RuntimeError('Request replay was accepted')
            if adapter.step_calls!=before:raise RuntimeError('Replay advanced model')
            report.update(success=True,successful_steps=10,final_done=adapter.done,final_error_dm=errors[-1],
                          integrated_absolute_error_dm_min=sum(abs(v) for v in errors)*60,
                          saturation_count=controller.saturation_count if controller else 0,
                          no_step_on_rejection=True,no_step_on_replay=True)
            if controller:
                report.update(goal_evaluation=evaluation,goal_config=goal,goal_safety_scope='D action/observation integrity only, not physical safety certification')
                if not evaluation['achieved']:raise RuntimeError('Research level goal not achieved')
            elif not adapter.done or not result['truncated']:raise RuntimeError('Adapter step cap not honored')
        save()
    try:
        with contextlib.redirect_stdout((args.output/'runtime_stdout.log').open('w')):
            run('mab_steady_10',1610)
            run('PI_training',config['tuning_seed'],True)
            for seed in config['evaluation_seeds']:run(f'PI_evaluation_{seed}',seed,True,config['evaluation_initial_buffer_offsets_dm'][str(seed)])
            # B evidence for A-approved Atropine layout only; never enables D runtime.
            authorization=json.loads((ROOT/'configs/research/r03_experiment_authorization_20261116.json').read_text())
            observations=[]
            for seed in authorization['baseline_seeds']:
                with EnvironmentAdapter(max_steps=authorization['baseline_max_steps']) as a:
                    obs=a.reset(seed=seed);metadata=a.metadata
                    if (a.observation_space.shape!=(39,) or a.action_space.shape!=(4,)
                            or metadata['normalize'] is not False or metadata['uss_subtracted'] is not True
                            or metadata['observation_flags']!=authorization['observation_flags']):raise RuntimeError('Atropine authorized layout mismatch')
                    if not 0<obs[5] or not obs[5]*1.1<20:raise RuntimeError('Candidate upper hidden by clip')
                    observations.append({'seed':seed,'layout_verified':True,'candidate_visible':True,'metadata':metadata})
            summary['atropine_authorized_layout_verification']=observations
        summary['success']=True
    except Exception:
        summary['traceback']=traceback.format_exc();print(summary['traceback'],file=sys.stderr)
    finally:save()
    return 0 if summary['success'] else 1

if __name__=='__main__':raise SystemExit(main())
