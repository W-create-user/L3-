"""Real mAb candidate refusal and lost-feedback checks; no blind holdout data."""
import argparse
import contextlib
import json
from pathlib import Path
from l3_system.execution import MAbAdapter,ExecutionAgent
from l3_system.execution.controllers.acknowledged_buffer import AcknowledgedBufferPolicy
from l3_system.agents.monitoring_agent import MonitoringAgent
from deliverables.D07.scripts.run_mab_paired import make_plan


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);checks={};results=[]
    with (a.output/'stdout.log').open('w') as log,contextlib.redirect_stdout(log),MAbAdapter(max_steps=4) as adapter:
        adapter.reset(seed=26001)
        policy=AcknowledgedBufferPolicy(adapter,make_plan()['controller'],
            startup_level_dm=float(adapter.get_observation()[17]*adapter._xscale[17]),startup_trusted=True,
            episode_id=adapter.metadata['episode_id'],state_version=adapter.state_version,allow_research=True)
        executor=ExecutionAgent(adapter,log_path=a.output/'B_events.jsonl')
        monitor=MonitoringAgent(adapter.action_space,adapter.observation_space)
        action=policy.predict(adapter.get_observation(),adapter.action_space)
        def refuse(request,observation=None):
            response=monitor.check_request(request,observation)
            response.update(decision='reject',rule_id='B26-INJECTED-REFUSAL',reason='development fault injection',risk_level=3)
            return response
        result=executor.execute_assignment({'task_id':'candidate-refused','task_type':'control_cycle'},refuse,action=action)
        policy.commit(False);results.append(result)
        checks['D_refusal_zero_step_no_observer_update']=(adapter.step_calls==0 and policy.version==adapter.state_version and not policy.failed)
        # Deliberately lose the result after exactly one real plant call.
        original=adapter.step
        def lose(action):
            raw=original(action)
            (a.output/'actual_step_before_feedback_loss.json').write_text(json.dumps(raw,indent=2)+'\n')
            raise RuntimeError('B26 deliberate feedback loss after actual model step')
        adapter.step=lose
        action=policy.predict(adapter.get_observation(),adapter.action_space)
        try:
            result=executor.execute_assignment({'task_id':'candidate-lost-feedback','task_type':'control_cycle'},monitor.check_request,action=action)
        except RuntimeError as error:
            result={'success':False,'executed':None,'error':str(error),'actual_step_calls':adapter.step_calls}
        results.append(result)
        try:policy.commit(False)
        except RuntimeError:checks['unknown_execution_latches_candidate']=policy.failed and adapter.step_calls==1
        else:checks['unknown_execution_latches_candidate']=False
        try:policy.predict(adapter.get_observation(),adapter.action_space)
        except RuntimeError:checks['no_candidate_retry_after_unknown']=adapter.step_calls==1
        else:checks['no_candidate_retry_after_unknown']=False
    report={'checks':checks,'success':all(checks.values()),'real_step_calls':1,
            'scope':'development fault injection, one actual step intentionally lacks controller receipt'}
    (a.output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    return 0 if report['success'] else 1


if __name__=='__main__':raise SystemExit(main())
