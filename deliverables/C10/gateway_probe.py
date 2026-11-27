"""Read-only compatibility check against the pinned B gateway's mAb backend."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess

from l3_system.agents.monitoring_agent import MonitoringAgent
from l3_system.execution.execution_agent import ExecutionAgent

B_COMMIT = "68831ef9e06440a1066e9d18bfb3955e1ac3006f"


def run(out):
    if subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() != B_COMMIT:
        raise RuntimeError("Run from the pinned B checkout with the C repository on PYTHONPATH")
    from l3_system.execution.joint_action_gateway import JointActionGateway
    from l3_system.execution.mab_adapter import MAbAdapter
    from l3_system.execution.training_wrapper import JointActionResearchEnv

    out.mkdir(parents=True, exist_ok=False)
    with MAbAdapter(max_steps=4) as adapter:
        executor = ExecutionAgent(adapter, log_path=out / "B_events.jsonl")
        executor.reset(seed=1101)
        monitor = MonitoringAgent(adapter.action_space, adapter.observation_space)
        # This partition only tests collection. It assigns no trained actor or
        # approved local observation, and uses B's existing steady reference.
        config = {"environment_id": adapter.metadata["environment_id"],
                  "action_coordinate": adapter.metadata["action_coordinate"],
                  "action_size": adapter.action_space.shape[0],
                  "observation_size": adapter.observation_space.shape[0],
                  "partitions": [{"actor_id": "other_axes", "action_indices": [0, 1, 2, 3, 4, 5, 6, 8],
                                  "observation_indices": []},
                                 {"actor_id": "buffer_axis", "action_indices": [7],
                                  "observation_indices": [17]}]}
        gateway = JointActionGateway(executor, monitor, config)
        try:
            JointActionResearchEnv(gateway)
        except RuntimeError as exc:
            training_gate = str(exc)
        else:
            raise AssertionError("B wrapper unexpectedly accepted training")
        ref = adapter._env.steady_actions.tolist()
        parts = {name: [ref[i] for i in spec["action_indices"]]
                 for name, spec in gateway.partitions.items()}
        wrapper = JointActionResearchEnv(gateway, research_only=True)
        approved = wrapper.step(parts)
        assert approved["success"] and approved["step_delta"] == 1
        assert not approved["learning_eligible"]
        reject = {**parts, "buffer_axis": [float(adapter.action_space.high[7]) + 1]}
        rejected = wrapper.step(reject)
        assert rejected["rejected"] and rejected["step_delta"] == 0
        assert adapter.step_calls == 1
        for row in (approved, rejected):
            if row["execution_result"]:
                row["execution_result"].pop("observation", None)
            row.pop("observation_before", None)
            row.pop("observation_after", None)
        result = {"backend": adapter.metadata["environment_id"], "B_source": B_COMMIT,
                  "source_sha256": {name: sha256(Path(name).read_bytes()).hexdigest()
                                   for name in ("l3_system/execution/joint_action_gateway.py",
                                                "l3_system/execution/training_wrapper.py")},
                  "fixture_partition": config, "training_gate": training_gate,
                  "model_step_calls": adapter.step_calls, "approved": approved, "rejected": rejected,
                  "D_gate": "MonitoringAgent R-01/R-02, no D26 learned gate",
                  "reviews": monitor.get_logs()}
        (out / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                                     allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args().out)
