"""Real Atropine validation of bounded regeneration and manual handover."""
import argparse
import json
from pathlib import Path

from l3_system.agents.monitoring_agent import MonitoringAgent
from l3_system.execution import EnvironmentAdapter, ExecutionAgent
from l3_system.runtime import MultiAgentSystem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('outputs/session_validation'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with EnvironmentAdapter(max_steps=60) as adapter:
        executor = ExecutionAgent(adapter, log_path=args.output/'B_events.jsonl')
        executor.reset(seed=1101)
        monitor = MonitoringAgent(adapter.action_space, adapter.observation_space)
        system = MultiAgentSystem(executor=executor, monitor=monitor,
                                  event_path=args.output/'events.jsonl', max_action_regenerations=1)
        def replacement(obs, space, attempt):
            return space.high + 1 if attempt == 0 else (space.low + space.high)/2
        recovered = system.run_cycle(action_generator=replacement)
        if recovered['summary']['system_state'] != 'autonomous' or adapter.step_calls != 1:
            raise RuntimeError('Bounded recovery failed')
        rejected = system.run_cycle(action_generator=lambda o, s, a: s.high + 1)
        if rejected['summary']['system_state'] != 'takeover_requested' or adapter.step_calls != 1:
            raise RuntimeError('Rejected candidates advanced the environment')
        try:
            system.run_cycle()
        except RuntimeError:
            pass
        else:
            raise RuntimeError('Takeover did not block new work')
        system.resume(operator='simulation-validation', reason='Switch to midpoint baseline')
        resumed = system.run_cycle()
        if resumed['summary']['run_status'] != 'completed' or adapter.step_calls != 2:
            raise RuntimeError('Explicit human resume failed')
        monitor.export_logs(args.output/'D_reviews.jsonl')
        report = {'environment':'real SMPL.AtropineEnvGym', 'seed':1101,
                  'human_confirmation':'simulated API call, not a human-factor study',
                  'total_step_calls':adapter.step_calls,
                  'recovered':recovered, 'rejected':rejected, 'resumed':resumed,
                  'transitions':system.transitions}
        (args.output/'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print('Validated real SMPL regeneration, takeover lock and explicit resume; 2 steps')


if __name__ == '__main__':
    main()
