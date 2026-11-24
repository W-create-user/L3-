"""Drive the Flask operator workflow against a live local Atropine simulator."""
import argparse
import json
from pathlib import Path
import re

from l3_system.agents.monitoring_agent import MonitoringAgent
from l3_system.execution import EnvironmentAdapter, ExecutionAgent
from l3_system.runtime import MultiAgentSystem
from l3_system.ui.app import create_app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with EnvironmentAdapter(max_steps=60) as adapter:
        executor = ExecutionAgent(adapter, log_path=args.output / "B_events.jsonl")
        executor.reset(seed=1101)
        monitor = MonitoringAgent(adapter.action_space, adapter.observation_space)
        system = MultiAgentSystem(executor=executor, monitor=monitor,
                                  event_path=args.output / "A_events.jsonl")
        client = create_app(system, environment="AtropineEnvGym", separate_confirmation=True).test_client()
        outcomes = []
        operations = [("cycle", 303, 1), ("pause", 303, 1), ("cycle", 409, 1),
                      ("resume", 409, 1), ("confirm", 303, 1), ("resume", 303, 1),
                      ("cycle", 303, 2)]
        for operation, expected_code, expected_steps in operations:
            page = client.get("/").get_data(as_text=True)
            form = dict(re.findall(r'name="(token|revision)" value="([^"]+)"', page))
            response = client.post("/" + operation, data={
                **form, "operator": "fixture operator", "reason": "A07 simulation verification",
            })
            outcomes.append({"operation": operation, "http_status": response.status_code,
                             "step_calls": adapter.step_calls, "state_version": adapter.state_version})
            if (response.status_code, adapter.step_calls) != (expected_code, expected_steps):
                raise RuntimeError(f"Unexpected console result: {outcomes[-1]}")
        if system.scheduler.active_agent_tasks or executor._pending or system.state != "autonomous":
            raise RuntimeError("Console ended with unresolved resources or restrictions")
        if 'id="observation-trend"' not in client.get("/").get_data(as_text=True):
            raise RuntimeError("Console did not render the observation trend")
        (args.output / "summary.json").write_text(
            json.dumps({"backend": "AtropineEnvGym", "seed": 1101, "operations": outcomes},
                       indent=2), encoding="utf-8",
        )
        print("7 Flask operations verified on Atropine; 2 simulator steps")


if __name__ == "__main__":
    main()
