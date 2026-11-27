"""Real coupled mAb gateway checks, including deliberate feedback loss after a step."""
import argparse
import contextlib
import json
from pathlib import Path
from l3_system.execution import MAbAdapter,ExecutionAgent
from l3_system.agents.monitoring_agent import MonitoringAgent
from l3_system.execution.joint_action_gateway import JointActionGateway
from l3_system.execution.training_wrapper import JointActionResearchEnv

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    config=json.loads((ROOT/'configs/training/mab_action_partitions.draft.json').read_text());checks={};records=[]
    def require(ok,why):
        if not ok:raise RuntimeError(why)
    with (a.output/'runtime_stdout.log').open('w') as log,contextlib.redirect_stdout(log):
        with MAbAdapter(max_steps=6) as adapter:
            executor=ExecutionAgent(adapter,log_path=a.output/'B_events.jsonl');executor.reset(seed=2010)
            monitor=MonitoringAgent(adapter.action_space,adapter.observation_space);tick=[0.]
            gateway=JointActionGateway(executor,monitor,config,clock=lambda:tick[0])
            def fill(context,joint=None):
                joint=adapter._env.steady_actions if joint is None else joint
                for actor in ['capture','buffer','upstream']:
                    indices=gateway.partitions[actor]['action_indices']
                    gateway.submit(context['cycle_id'],actor,joint[indices],state_version=context['state_version'],episode_id=context['episode_id'],coordinate=context['coordinate'])
            context=gateway.begin('valid');fill(context);result=gateway.finish('valid');records.append(result)
            require(result['success'] and adapter.step_calls==1,'Full joint action failed')
            require(gateway.finish('valid')==result and adapter.step_calls==1,'Double finish repeated model step');checks['one_step_after_out_of_order_components']=True
            context=gateway.begin('missing',timeout_seconds=1)
            gateway.submit('missing','buffer',[float(adapter._env.steady_actions[7])],state_version=context['state_version'],episode_id=context['episode_id'],coordinate=context['coordinate'])
            tick[0]=2;result=gateway.finish('missing');records.append(result)
            require(result['status']=='timeout' and adapter.step_calls==1,'Missing actor silently filled');checks['missing_timeout_no_step']=True
            context=gateway.begin('duplicate')
            for index in range(2):
                try:gateway.submit('duplicate','capture',[0.],state_version=context['state_version'],episode_id=context['episode_id'],coordinate=context['coordinate'])
                except ValueError:
                    require(index==1,'First valid component failed')
            records.append(gateway.finish('duplicate'));require(adapter.step_calls==1,'Duplicate actor caused a step');checks['duplicate_no_step']=True
            context=gateway.begin('D_reject');bad=adapter._env.steady_actions.copy();bad[0]=adapter.action_space.high[0]+1
            fill(context,bad);result=gateway.finish('D_reject');records.append(result)
            require(result['rejected'] and result['executed_joint_action'] is None and result['reward'] is None and adapter.step_calls==1,'D refusal created a fictitious transition')
            checks['D_rejection_no_step']=True
            executor.reset(seed=2011);gateway.acknowledge_after_reset(operator='offline-validation',reason='New explicit simulation episode after rejection')
            context=gateway.begin('stale');fill(context);executor.reset(seed=2012);result=gateway.finish('stale');records.append(result)
            require(result['status']=='stale_round' and adapter.step_calls==1,'Stale epoch stepped');checks['stale_round_no_step']=True
            wrapper=JointActionResearchEnv(gateway,research_only=True)
            actions={actor:adapter._env.steady_actions[spec['action_indices']] for actor,spec in gateway.partitions.items()}
            result=wrapper.step(actions);records.append(result)
            require(result['transition_complete'] and not result['learning_eligible'] and adapter.step_calls==2,'Draft wrapper claimed training readiness')
            checks['research_wrapper_not_training_approval']=True
            context=gateway.begin('lost_feedback');fill(context);original=adapter.step
            def lose_feedback(action):
                original(action)
                raise RuntimeError('Offline injected feedback loss AFTER real step')
            adapter.step=lose_feedback;result=gateway.finish('lost_feedback');records.append(result)
            require(result['executed'] is None and result['step_delta']==1 and gateway.takeover_required and adapter.step_calls==3,'Unknown outcome was replayable')
            require(gateway.finish('lost_feedback')==result and adapter.step_calls==3,'Unknown action repeated')
            try:gateway.begin('automatic_retry')
            except RuntimeError:pass
            else:raise RuntimeError('Takeover did not block retries')
            checks['unknown_result_latches_no_replay']=True
    (a.output/'records.json').write_text(json.dumps(records,indent=2)+'\n')
    (a.output/'summary.json').write_text(json.dumps({'success':True,'checks':checks,'real_step_calls':3,
        'complete_successful_transitions':2,'deliberately_lost_feedback_after_real_step':1,
        'C05_contract_confirmed':False,'training_ready':False,'R03_runtime_enabled':False},indent=2)+'\n')

if __name__=='__main__':main()
