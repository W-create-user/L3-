"""Verify A's numeric mAb route using a checkout containing the B03 backend."""
import argparse
import json
from pathlib import Path

from l3_system.main import run_live


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--b-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
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
    result = run_live(
        environment="MAbEnvGym", seed=config["tuning_seed"], max_steps=goal["max_steps"],
        goal_spec=goal, goal_registry=registry, timeout_seconds=60, output_dir=args.output,
    )
    summary = result["summary"]
    cycles = [cycle["summary"] for cycle in result["cycles"]]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if (summary["status"] != "succeeded"
            or summary["total_step_calls"] != goal["hold_steps"]
            or not all(c["all_C_locks_released"] for c in cycles)
            or [c["step_delta"] for c in cycles] != [1] * goal["hold_steps"] + [0]):
        raise SystemExit("mAb steady numeric regression did not pass")


if __name__ == "__main__":
    main()
