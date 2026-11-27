"""Real provider process exit: missing component must not advance coupled mAb.

This is a local process-failure test, not C04 network-transport certification.
"""
import argparse
import contextlib
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    from l3_system.execution import MAbAdapter, ExecutionAgent
    from l3_system.agents.monitoring_agent import MonitoringAgent
    from l3_system.execution.joint_action_gateway import JointActionGateway
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    config = json.loads((Path(__file__).resolve().parents[1]/'configs/training/mab_action_partitions.draft.json').read_text())
    with (a.output/'stdout.log').open('w') as log, contextlib.redirect_stdout(log):
        with MAbAdapter(max_steps=3) as adapter:
            executor = ExecutionAgent(adapter, log_path=a.output/'B_events.jsonl')
            executor.reset(seed=231123)
            gateway = JointActionGateway(executor, MonitoringAgent(adapter.action_space, adapter.observation_space), config)
            before = adapter.step_calls
            # Provider exits before the round starts: supervisor observes a real OS exit,
            # supplies no default action, and lets the collection deadline expire.
            worker = subprocess.run([sys.executable, '-c', 'raise SystemExit(23)'], capture_output=True, timeout=5)
            context = gateway.begin('withdrawn-buffer-provider', timeout_seconds=.05)
            for actor in ['upstream', 'capture']:
                indices = gateway.partitions[actor]['action_indices']
                gateway.submit(context['cycle_id'], actor, adapter._env.steady_actions[indices],
                    state_version=context['state_version'], episode_id=context['episode_id'], coordinate=context['coordinate'])
            time.sleep(.06)
            record = gateway.finish(context['cycle_id'])
            again = gateway.finish(context['cycle_id'])
            success = (worker.returncode==23 and record['status']=='timeout' and adapter.step_calls==before
                       and record['reward'] is None and record['executed_joint_action'] is None and record==again)
            (a.output/'record.json').write_text(json.dumps(record, indent=2)+'\n')
            report = {'success': success, 'provider_exit_code': worker.returncode, 'missing_actor': 'buffer',
                      'real_step_delta': adapter.step_calls-before, 'duplicate_finish_same_record': record==again,
                      'default_action_inserted': False, 'training_ready': False,
                      'scope': 'local provider OS exit, real monotonic deadline, real initialized mAb; not distributed transport'}
            (a.output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(main())
