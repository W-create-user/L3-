"""Verify A's numeric mAb route using a checkout containing the B03 backend."""
import argparse
import json
from pathlib import Path

from l3_system.main import run_live, _plain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--b-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pi-seed", type=int, help="Run B research PI from its frozen offset for this seed")
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    config = json.loads(args.b_config.read_text(encoding="utf-8"))
    if config.get("research_only") is not True or config.get("production_target_approved") is not False:
        parser.error("This verifier requires B's research configuration")
    observation = registry["variables"]["buffer_height"]
    setpoint, tolerance = config["setpoint_dm"], config["target_tolerance_dm"]
    goal = {
        "schema_version": registry["schema_version"],
        "goal_id": "a-mab-steady-buffer-regression",
        "environment_id": "MAbEnvGym",
        "registry_version": registry["registry_version"],
        "variable": "buffer_height",
        "unit": observation["unit"],
        "coordinate": observation["coordinate"],
        "target_range": {"lower": setpoint - tolerance, "upper": setpoint + tolerance},
        "hold_steps": config["hold_steps"],
        "max_steps": config["horizon_steps"],
        "allowed_actions": {
            name: {"unit": variable["unit"], "coordinate": variable["coordinate"], "range": variable["bounds"]}
            for name, variable in registry["variables"].items() if variable["source"] == "action"
        },
    }
    if args.pi_seed is None:
        result = run_live(
            environment="MAbEnvGym", seed=config["tuning_seed"], max_steps=goal["max_steps"],
            goal_spec=goal, goal_registry=registry, timeout_seconds=60, output_dir=args.output,
        )
    else:
        offsets = config["evaluation_initial_buffer_offsets_dm"]
        if str(args.pi_seed) not in offsets:
            parser.error("--pi-seed must be one of B's frozen evaluation seeds")
        goal["goal_id"] = "a-mab-pi-offset-recovery"
        result = run_pi_case(args.output, args.pi_seed, offsets[str(args.pi_seed)], config, goal, registry)
    summary = result["summary"]
    cycles = [cycle["summary"] for cycle in result["cycles"]]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    steps = summary["total_step_calls"]
    if (summary["status"] != "succeeded" or not goal["hold_steps"] <= steps <= goal["max_steps"]
            or not all(c["all_C_locks_released"] for c in cycles)
            or [c["step_delta"] for c in cycles] != [1] * steps + [0]):
        raise SystemExit("mAb numeric regression did not pass")
    if args.pi_seed is None and steps != goal["hold_steps"]:
        raise SystemExit("mAb steady regression exceeded its expected step count")
    if args.pi_seed is not None and result["plans"][0]["method"] != "adjust_target":
        raise SystemExit("mAb PI regression did not start outside its target window")


def run_pi_case(output, seed, offset_dm, config, goal, registry):
    import numpy as np
    from l3_system.agents.monitoring_agent import MonitoringAgent
    from l3_system.execution import ExecutionAgent
    from l3_system.execution.controllers import BufferLevelPID
    from l3_system.planning.goal_runtime import open_environment, run_numeric_goal
    from l3_system.runtime import MultiAgentSystem

    output.mkdir(parents=True, exist_ok=False)
    with open_environment("MAbEnvGym", max_steps=goal["max_steps"]) as adapter:
        executor = ExecutionAgent(adapter, log_path=output / "B_events.jsonl")
        executor.reset(seed=seed)
        initial = np.asarray(executor.get_observation()).copy()
        index = registry["variables"]["buffer_height"]["index"]
        initial[index] += offset_dm / adapter._xscale[index]
        adapter.reset(seed=seed, initial_state=initial)
        monitor = MonitoringAgent(adapter.action_space, adapter.observation_space)
        system = MultiAgentSystem(executor=executor, monitor=monitor, event_path=output / "A_events.jsonl")
        result = run_numeric_goal(system, goal, registry, environment="MAbEnvGym",
                                  timeout_seconds=60, policy=BufferLevelPID(adapter, config))
        result["summary"].update(seed=seed, initial_offset_dm=offset_dm, total_step_calls=adapter.step_calls)
        (output / "result.json").write_text(json.dumps(_plain(result), indent=2, allow_nan=False),
                                            encoding="utf-8")
        monitor.export_logs(output / "D_reviews.jsonl")
        return result


if __name__ == "__main__":
    main()
