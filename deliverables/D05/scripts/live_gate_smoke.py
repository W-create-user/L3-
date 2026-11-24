"""Actual B adapter spaces + D gate; constructs/resets backend, never steps it."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
from l3_system.agents.monitoring_agent import MonitoringAgent

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=('Atropine','mAb'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.backend=='Atropine':
        from l3_system.execution.environment_adapter import EnvironmentAdapter as Adapter
    else:
        from l3_system.execution.mab_adapter import MAbAdapter as Adapter
    with contextlib.redirect_stdout(io.StringIO()):
        with Adapter(max_steps=10) as adapter:
            adapter.reset(seed=1101)
            monitor=MonitoringAgent(adapter.action_space,adapter.observation_space)
            obs=adapter.get_observation()
            action=(adapter.action_space.low+adapter.action_space.high)/2
            request={'task_id':'D-B-live-space-smoke','request_id':'gate-only','action':action}
            allowed=monitor.check_request(request,obs)
            rejected=monitor.check_request({**request,'action':adapter.action_space.high+1},obs)
            missing=monitor.check_request(request,None)
            assert allowed['decision']=='pass' and rejected['decision']=='reject' and missing['decision']=='reject'
            assert adapter.step_calls==0
            result={'backend':args.backend,'fixture_only':False,'simulator_step_calls':adapter.step_calls,
                'metadata':getattr(adapter,'metadata',None),'model_gate_only':True,'D04_D05_enabled':False,
                'observation_shape':list(adapter.observation_space.shape),'action_shape':list(adapter.action_space.shape),
                'physical_safety_certified':False,'valid_model_action':allowed,
                'out_of_bounds':rejected,'missing_observation':missing}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'backend':args.backend,'checks':3,'all_passed':True,'simulator_step_calls':0}))
if __name__=='__main__':main()
