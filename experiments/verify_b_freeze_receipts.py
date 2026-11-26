"""Public B signoff fixtures: native mAb receipt failures plus full A/D smoke.

Uses only disclosed seed 26001 and a trusted pre-fault simulator reset.
Never reads the private holdout plan. Outputs are local review evidence.
"""
import argparse
import contextlib
import copy
from datetime import datetime,timezone
import json
import time
from pathlib import Path
import numpy as np
from l3_system.execution import MAbAdapter,ExecutionAgent
from l3_system.execution.controllers.acknowledged_buffer import AcknowledgedBufferPolicy
from l3_system.execution.controllers.receipt_boundary import ReceiptBoundary
from l3_system.agents.monitoring_agent import MonitoringAgent
from l3_system.agents.process_rules import ProcessRuleEngine
from l3_system.runtime import MultiAgentSystem
from l3_system.planning.goal_runtime import run_numeric_goal
from deliverables.D07.scripts import run_mab_paired as d24


def new_boundary(adapter,clock=None):
    policy=AcknowledgedBufferPolicy(adapter,d24.make_plan()['controller'],
        startup_level_dm=float(adapter.get_observation()[17]*adapter._xscale[17]),
        startup_trusted=True,episode_id=adapter.metadata['episode_id'],state_version=adapter.state_version,
        allow_research=True,max_innovation_dm=.3)
    certificate=dict(episode_id=adapter.metadata['episode_id'],state_version=adapter.state_version,
        timestamp=datetime.now(timezone.utc).isoformat(),level_dm=policy.anchor[0],
        calibration_source='validated_simulator_reset_before_fault_injection')
    return ReceiptBoundary(adapter,policy,certificate,receipt_timeout_seconds=.5,clock=(lambda:clock[0]) if clock is not None else time.monotonic)


class Owner:
    def __init__(self,boundary):self.boundary=boundary;self.receipts=[]
    def __getattr__(self,name):return getattr(self.boundary,name)
    def step(self,action):
        result=self.boundary.step(action)
        receipt=self.boundary.make_receipt(result)
        self.receipts.append(copy.deepcopy(receipt))
        self.boundary.acknowledge(receipt)
        return result


class MeasurementView:
    def __init__(self,boundary):self.boundary=boundary
    def reset(self):self.boundary.reset()
    def predict(self,observation,space):
        # Owner-side projection: candidate only receives h in a zero-filled
        # shape-compatible vector. Scoring truth for other states is withheld.
        delivered=np.zeros(1970);delivered[17]=float(observation[17])+.1/3
        return self.boundary.predict(delivered,space)
    def commit(self,executed):self.boundary.commit(executed)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    reports=[]
    with (a.output/'native_stdout.log').open('w') as log,contextlib.redirect_stdout(log):
        for kind in ['correct','malformed_action','missing_action','missing_receipt','wrong_request','wrong_episode',
                     'wrong_version','wrong_step','expired','duplicate','scoring_truth_in_receipt']:
            with MAbAdapter(max_steps=10) as adapter:
                adapter.reset(seed=26001);tick=[0.];guard=new_boundary(adapter,tick)
                delivered=np.zeros(1970);delivered[17]=float(adapter.get_observation()[17])+.1/3
                action=guard.predict(delivered,adapter.action_space)
                request=dict(request_id='public-'+kind,episode_id=adapter.metadata['episode_id'],
                    state_version=adapter.state_version,step_count=adapter.step_count,action=action.tolist())
                guard.bind_request(request)
                raw=guard.step(action);receipt=guard.make_receipt(raw)
                assert set(receipt)==guard.RECEIPT_FIELDS
                if kind=='correct':
                    guard.acknowledge(receipt);guard.commit(True)
                    assert not guard.failed and adapter.step_calls==1
                    reports.append({'case':kind,'passed':True,'native_step_calls':1,'allowlisted_receipt':True})
                    continue
                if kind=='malformed_action':receipt['action']=['broken']*9
                elif kind=='missing_action':receipt.pop('action')
                elif kind=='wrong_request':receipt['request_id']='other'
                elif kind=='wrong_episode':receipt['episode_id']='other'
                elif kind=='wrong_version':receipt['state_version']+=1
                elif kind=='wrong_step':receipt['step_count']+=1
                elif kind=='expired':tick[0]=1.
                elif kind=='scoring_truth_in_receipt':receipt['observation']=raw['observation']
                elif kind=='duplicate':guard.acknowledge(receipt)
                before=adapter.step_calls
                try:
                    if kind=='missing_receipt':guard.step(action)
                    else:guard.acknowledge(receipt)
                except RuntimeError:pass
                else:raise AssertionError('Invalid or missing receipt accepted: '+kind)
                for operation in [lambda:guard.step(action),lambda:guard.predict(delivered,adapter.action_space)]:
                    try:operation()
                    except RuntimeError:pass
                    else:raise AssertionError('Boundary accepted further activity')
                assert guard.failed and guard.policy.failed and adapter.step_calls==before==1
                reports.append({'case':kind,'passed':True,'native_step_calls':adapter.step_calls,
                    'extra_native_steps_after_error':0,'boundary_failed':True,'candidate_failed':True})
        # Public full-chain smoke: real A/B/C local path, same D model envelope.
        with MAbAdapter(max_steps=10) as adapter:
            adapter.reset(seed=26001);guard=new_boundary(adapter);owner=Owner(guard)
            ex=ExecutionAgent(owner,log_path=a.output/'public_B_events.jsonl')
            registry,goal,rules=d24.definitions(owner,{'run_id':'public-receipt-boundary'})
            context={'environment':'mAb','episode_id':adapter.metadata['episode_id'],
                'variables':{'h_model':{'unit':'dm / xscale[17]','coordinate':'physical_divided_by_xscale'}}}
            mon=MonitoringAgent(owner.action_space,owner.observation_space,
                process_rules=ProcessRuleEngine(rules),process_context_provider=lambda *_:context)
            original=mon.check_request
            def review(request,observation=None):
                guard.bind_request(request)
                return original(request,observation)
            mon.check_request=review
            system=MultiAgentSystem(executor=ex,monitor=mon,event_path=a.output/'public_A_C_events.jsonl')
            result=run_numeric_goal(system,goal,registry,environment='MAbEnvGym',policy=MeasurementView(guard),timeout_seconds=90)
            assert result['summary']['status']=='succeeded' and adapter.step_calls==3
            assert not system.scheduler.active_agent_tasks and not ex._pending and not guard.failed
            assert len(owner.receipts)==3 and all(set(r)==guard.RECEIPT_FIELDS for r in owner.receipts)
            smoke={'passed':True,'status':result['summary']['status'],'native_step_calls':adapter.step_calls,
                   'C_resources_clear':True,'B_pending_clear':True,'receipt_fields':sorted(guard.RECEIPT_FIELDS),
                   'hard_model_domain_gate_enabled':True,'learned_gate_enabled':False,
                   'scope':'public same-process one-executor fixture; not final frozen runner signoff'}
            (a.output/'public_goal_result.json').write_text(json.dumps(result,indent=2)+'\n')
    report={'success':True,'public_seed':26001,'receipt_cases':reports,'full_chain_smoke':smoke,
        'trusted_startup_before_fault':True,'receipt_timeout_fixture_seconds':.5,
        'formal_timeout_must_be_jointly_fixed':True,'holdout_read':False,'blind_test_started':False,
        'formal_launch_authorized':False,'native_step_calls_total':sum(r['native_step_calls'] for r in reports)+3}
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
