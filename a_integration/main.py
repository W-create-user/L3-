"""L3 A-side orchestrator for the 2026-11-08 acceptance run.

The live runner uses the exact B ExecutionAgent/EnvironmentAdapter and D
MonitoringAgent sources supplied by the team.  The offline integration tests
use an injected test adapter; they are not reported as real SMPL runs.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from typing import Any, Callable, Mapping

import numpy as np

from a_integration.config import EXECUTOR_ID
from a_integration.decision_agent import DecisionAgent
from a_integration.agents.scheduling_agent import SchedulingAgent
from a_integration.agents.monitoring_agent import MonitoringAgent
from a_integration.adapters.a_to_c_adapter import convert_subtasks_for_c
from a_integration.adapters.assignment_adapter import assert_winner, build_b_task
from a_integration.adapters.b_to_d_adapter import make_d_safety_checker
from a_integration.adapters.feedback_adapter import execution_result_to_feedback
from a_integration.planning.replanning import ReplanningController

ROOT = Path(__file__).resolve().parent
B_CODE_ROOT = ROOT.parent / "执行智能体" / "code"
OWNER_BY_TASK = {
    "collect_observation": "ExecutionAgent-01",
    "evaluate_goal": "DecisionAgent-01",
    "evaluate_state": "DecisionAgent-01",
    "return_state_result": "DecisionAgent-01",
    "generate_action_intent": "DecisionAgent-01",
    "safety_review": "MonitoringAgent-01",
    "execute_approved_action": "ExecutionAgent-01",
    "verify_result": "DecisionAgent-01",
}
AGENT_IDS = ("DecisionAgent-01", "ExecutionAgent-01", "MonitoringAgent-01")


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _event(events: list[dict[str, Any]], event: str, **payload: Any) -> None:
    events.append({"timestamp": datetime.now(timezone.utc).isoformat(),
                   "event": event, **_plain(payload)})


def configured_bids(task: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Two deterministic capability-profile bids per task.

    These are local prototype bids (not dynamic telemetry from live agents).
    The winning logical module is then invoked locally.  This meets the CNP
    workflow demonstration requirement without misrepresenting bids as live.
    """
    owner = OWNER_BY_TASK.get(str(task["task_type"]))
    if owner is None:
        raise ValueError(f"No capability owner configured for {task['task_type']!r}")
    alternate = next(agent for agent in AGENT_IDS if agent != owner)
    common = {"task_id": task["task_id"], "subtask_id": task["subtask_id"]}
    return [
        {**common, "agent_id": owner, "estimated_cost": 0.5,
         "capability_match": 0.95, "load": 0.10},
        {**common, "agent_id": alternate, "estimated_cost": 4.0,
         "capability_match": 0.40, "load": 0.80},
    ]


def _default_action_generator(observation, action_space, attempt=0):
    """B's documented space-midpoint baseline; not an optimized controller."""
    from integration_abd_1103.local_policy import space_midpoint
    return space_midpoint(observation, action_space)


def run_control_cycle(*, goal: str | dict[str, Any], executor: Any,
                      monitor: Any, executor_id: str = EXECUTOR_ID,
                      scheduler: Any | None = None,
                      action_generator: Callable[..., Any] | None = None,
                      event_path: str | Path | None = None) -> dict[str, Any]:
    """Run the six-step A-HTN -> CNP -> B/D -> feedback cycle.

    The supplied executor must already be reset.  It can be B's real
    ExecutionAgent or a test double.  The final control action is checked once
    as the explicit HTN safety-review stage and again inside B immediately
    before env.step; B's second check is the authoritative execution gate.
    """
    if getattr(monitor, "demo_mode", True) is not False:
        raise RuntimeError("Formal run requires D MonitoringAgent with the live B action_space")
    if executor_id != EXECUTOR_ID:
        raise ValueError(f"Configured executor identity must be {EXECUTOR_ID!r}")

    decision = DecisionAgent()
    parent_task = decision.task_to_dict(decision.create_task(goal))
    c_tasks = convert_subtasks_for_c(parent_task)
    scheduler = scheduler or SchedulingAgent()
    scheduler.register_tasks(c_tasks)
    action_generator = action_generator or _default_action_generator
    safety_checker = make_d_safety_checker(monitor)
    subtask_by_id = {item["subtask_id"]: item for item in parent_task["subtasks"]}
    task_by_id = {item["subtask_id"]: item for item in c_tasks}
    events: list[dict[str, Any]] = []
    event_file = Path(event_path) if event_path else None
    if event_file:
        event_file.parent.mkdir(parents=True, exist_ok=True)
        event_file.write_text("", encoding="utf-8")

    def emit(event: str, **payload: Any) -> None:
        record = {"timestamp": datetime.now(timezone.utc).isoformat(),
                  "event": event, **_plain(payload)}
        events.append(record)
        if event_file:
            with event_file.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    stage_results: list[dict[str, Any]] = []
    replanner = ReplanningController(max_action_regenerations=2, max_environment_retries=1)
    context: dict[str, Any] = {"observation": None, "action": None,
                               "execution_result": None, "step_calls_before": None,
                               "preflight_review": None}

    for c_task in c_tasks:
        subtask_id = c_task["subtask_id"]
        subtask = subtask_by_id[subtask_id]
        ready_ids = {item["subtask_id"] for item in scheduler.ready_tasks()}
        if subtask_id not in ready_ids:
            raise RuntimeError(f"HTN dependency did not release {subtask_id}; ready={sorted(ready_ids)}")

        bids = configured_bids(c_task)
        started = time.perf_counter()
        assignment_full = scheduler.schedule(c_task, bids)
        assignment = scheduler.assignment_for_execution(assignment_full)
        expected_owner = OWNER_BY_TASK[c_task["task_type"]]
        assert_winner(assignment, expected_owner)
        emit("assignment_awarded", subtask_id=subtask_id,
               task_type=c_task["task_type"], bids=bids,
               ranked_bids=assignment_full.get("ranked_bids", []),
               assignment=assignment, bid_source="configured_local_capability_profiles")

        name = subtask["name"]
        stage_payload: dict[str, Any]
        if name == "collect_observation":
            before = getattr(executor.adapter, "step_calls", None)
            observation = executor.get_observation()
            arr = np.asarray(observation, dtype=float)
            if arr.ndim != 1 or not arr.size or not np.isfinite(arr).all():
                raise ValueError("B returned an invalid observation")
            after = getattr(executor.adapter, "step_calls", None)
            if before is not None and after != before:
                raise RuntimeError("Read-only observation unexpectedly advanced env.step")
            context["observation"] = _plain(observation)
            stage_payload = {"execution_status": "completed", "observation_shape": list(arr.shape),
                             "observation_finite": True, "step_delta": 0}

        elif name in {"evaluate_goal", "evaluate_state"}:
            if context["observation"] is None:
                raise RuntimeError("Goal evaluation requires the collected observation")
            arr = np.asarray(context["observation"], dtype=float)
            stage_payload = {"execution_status": "completed",
                             "evaluation": "observation_received_and_finite",
                             "observation_dimension": int(arr.size),
                             "numeric_process_target": None,
                             "note": "No numeric process target was supplied; this stage validates state availability only."}

        elif name == "return_state_result":
            stage_payload = {"execution_status": "completed", "observation": context["observation"]}

        elif name == "generate_action_intent":
            if context["observation"] is None:
                raise RuntimeError("Action intent requires a current observation")
            action = action_generator(context["observation"], executor.adapter.action_space, 0)
            arr = np.asarray(action, dtype=float)
            if arr.shape != tuple(executor.adapter.action_space.shape) or not np.isfinite(arr).all():
                raise ValueError("Action generator returned wrong-shaped or non-finite action")
            context["action"] = arr.tolist()
            stage_payload = {"execution_status": "completed", "action": context["action"],
                             "action_source": "space_midpoint_baseline" if action_generator is _default_action_generator else "injected_action_generator"}

        elif name == "safety_review":
            if context["action"] is None:
                raise RuntimeError("Safety review requires a generated action")
            request = {"task_id": parent_task["task_id"], "action": context["action"]}
            review = monitor.check_request(request, context["observation"])
            context["preflight_review"] = _plain(review)
            stage_payload = {"execution_status": "completed" if review.get("decision") == "pass" else "failed",
                             "safety_decision": _plain(review)}
            if review.get("decision") != "pass":
                feedback = {"task_id": parent_task["task_id"], "subtask_id": subtask_id,
                            "execution_status": "failed", "failure_reason": "safety_rejected",
                            "exclude_failed_agent": False}
                scheduler.update_feedback(feedback)
                failure_plan = replanner.classify({"task_id": parent_task["task_id"],
                    "subtask_id": subtask_id, "execution_status": "rejected",
                    "safety_decision": review})
                emit("subtask_failed", subtask_id=subtask_id, feedback=feedback,
                       replanning=failure_plan)
                stage_results.append({"subtask_id": subtask_id, "task_type": name,
                                      "duration_seconds": time.perf_counter()-started,
                                      **stage_payload})
                break

        elif name == "execute_approved_action":
            if context["action"] is None:
                raise RuntimeError("Execution requires a generated action")
            if context["preflight_review"] is None or context["preflight_review"].get("decision") != "pass":
                raise RuntimeError("Execution blocked: no successful HTN safety-review stage")
            b_task = build_b_task(assignment, parent_task, subtask)
            context["step_calls_before"] = getattr(executor.adapter, "step_calls", None)
            try:
                execution_result = executor.execute_assignment(
                    b_task, safety_checker, action=context["action"], subtask=subtask)
            except Exception as exc:
                # Preserve the exception as an orchestration failure; do not invent a B ExecutionResult.
                feedback = {"task_id": parent_task["task_id"], "subtask_id": subtask_id,
                            "execution_status": "failed", "failure_reason": "environment_error",
                            "exclude_failed_agent": False}
                scheduler.update_feedback(feedback)
                failure_plan = replanner.classify({"task_id": parent_task["task_id"],
                    "subtask_id": subtask_id, "execution_status": "failed",
                    "failure_reason": "environment_error"})
                stage_payload = {"execution_status": "failed", "exception_type": type(exc).__name__,
                                 "exception": str(exc), "feedback": feedback,
                                 "replanning": failure_plan}
                emit("b_execution_exception", subtask_id=subtask_id,
                       exception_type=type(exc).__name__, exception=str(exc),
                       feedback=feedback, replanning=failure_plan)
                stage_results.append({"subtask_id": subtask_id, "task_type": name,
                                      "duration_seconds": round(time.perf_counter()-started, 6),
                                      **stage_payload})
                break
            if execution_result.get("task_id") != parent_task["task_id"] or execution_result.get("subtask_id") != subtask_id:
                raise ValueError("B ExecutionResult task/subtask identity mismatch")
            context["execution_result"] = _plain(execution_result)
            feedback = execution_result_to_feedback(execution_result)
            if feedback is None:
                raise RuntimeError("B returned not_assigned for an awarded task")
            scheduler.update_feedback(feedback)
            stage_payload = {"execution_status": execution_result.get("execution_status"),
                             "execution_result": _plain(execution_result),
                             "feedback": feedback}
            if execution_result.get("execution_status") != "completed":
                failure_plan = replanner.classify(execution_result)
                emit("execution_not_completed", subtask_id=subtask_id,
                       execution_result=_plain(execution_result), feedback=feedback,
                       replanning=failure_plan,
                       automatic_retry="not_attempted_without_a_confirmed_compatible_live_backup_executor")
                stage_payload["replanning"] = failure_plan
                stage_results.append({"subtask_id": subtask_id, "task_type": name,
                                      "duration_seconds": time.perf_counter()-started,
                                      **stage_payload})
                break

        elif name == "verify_result":
            result = context["execution_result"]
            if not isinstance(result, dict):
                raise RuntimeError("Verification requires B ExecutionResult")
            after = getattr(executor.adapter, "step_calls", None)
            before = context["step_calls_before"]
            delta = after - before if before is not None and after is not None else None
            passed = (result.get("execution_status") == "completed"
                      and result.get("success") is True
                      and result.get("executed") is True
                      and delta == 1)
            stage_payload = {"execution_status": "completed" if passed else "failed",
                             "verified": bool(passed), "step_delta": delta,
                             "verification_rule": "one successful B execution must advance exactly one env.step"}
            if not passed:
                feedback = {"task_id": parent_task["task_id"], "subtask_id": subtask_id,
                            "execution_status": "failed", "failure_reason": "execution_verification_failed",
                            "exclude_failed_agent": False}
                scheduler.update_feedback(feedback)
                emit("verification_failed", subtask_id=subtask_id, feedback=feedback)
                stage_results.append({"subtask_id": subtask_id, "task_type": name,
                                      "duration_seconds": time.perf_counter()-started,
                                      **stage_payload})
                break
        else:
            raise NotImplementedError(f"No A-side handler for HTN task {name!r}")

        if not (name == "execute_approved_action"):
            feedback = {"task_id": parent_task["task_id"], "subtask_id": subtask_id,
                        "execution_status": "completed"}
            scheduler.update_feedback(feedback)
        # execute task's feedback was already applied using the real B result.
        stage_results.append({"subtask_id": subtask_id, "task_type": name,
                              "duration_seconds": round(time.perf_counter()-started, 6),
                              **stage_payload})
        emit("subtask_completed", subtask_id=subtask_id,
               task_type=name, duration_seconds=stage_results[-1]["duration_seconds"],
               status=scheduler.get_task_status(parent_task["task_id"], subtask_id))

    statuses = {item["subtask_id"]: scheduler.get_task_status(parent_task["task_id"], item["subtask_id"])
                for item in c_tasks}
    all_completed = len(statuses) == len(c_tasks) and all(x == "completed" for x in statuses.values())
    summary = {
        "run_status": "completed" if all_completed else "incomplete_or_failed",
        "task_id": parent_task["task_id"], "task_type": parent_task["task_type"],
        "subtask_count": len(c_tasks), "completed_subtasks": sum(x == "completed" for x in statuses.values()),
        "subtask_statuses": statuses, "executor_id": executor_id,
        "step_calls": getattr(executor.adapter, "step_calls", None),
        "step_count": getattr(executor.adapter, "step_count", None),
        "d_demo_mode": getattr(monitor, "demo_mode", None),
        "bids_are_live": False,
        "bids_note": "Configured local capability-profile bids; not dynamic bid telemetry from independent live agents.",
        "controller_note": "B space-midpoint is an interface baseline, not an optimized process controller.",
        "r03_enabled": False,
        "replanning_policy": {"max_action_regenerations": replanner.max_action_regenerations,
                               "max_environment_retries": replanner.max_environment_retries,
                               "automatic_live_reassignment": False},
    }
    emit("run_finished", summary=summary)
    return {"summary": summary, "parent_task": _plain(parent_task),
            "c_tasks": _plain(c_tasks), "stage_results": stage_results,
            "events": events, "assignments": [scheduler.get_assignment_history(parent_task["task_id"], t["subtask_id"])
                                                   for t in c_tasks],
            "d_logs": monitor.get_logs() if hasattr(monitor, "get_logs") else []}



def execute_awarded_subtask(*, parent_task, c_task, assignment, scheduler,
                            executor, monitor, executor_id=EXECUTOR_ID,
                            action=None, action_policy=None):
    """Backward-compatible one-subtask adapter retained for prior A tests/tools."""
    assert_winner(assignment, executor_id)
    matches = [item for item in parent_task.get("subtasks", [])
               if item.get("subtask_id") == assignment.get("subtask_id")]
    if len(matches) != 1 or assignment.get("task_id") != parent_task.get("task_id"):
        raise ValueError("Assignment does not identify exactly one original A HTN subtask")
    subtask = matches[0]
    name = subtask.get("name")
    if name == "collect_observation":
        observation = executor.get_observation()
        result = {"task_id": parent_task["task_id"],
                  "subtask_id": subtask["subtask_id"],
                  "execution_status": "completed", "success": True,
                  "executed": False, "observation": _plain(observation),
                  "failure_reason": None}
    elif name == "execute_approved_action":
        checker = make_d_safety_checker(monitor)
        result = executor.execute_assignment(parent_task, checker, action=action,
                                             action_policy=action_policy, subtask=subtask)
    else:
        raise NotImplementedError(f"No compatibility handler for {name!r}")
    from a_integration.adapters.feedback_adapter import update_scheduler_from_result
    feedback_result = update_scheduler_from_result(scheduler, result)
    return {"execution_result": result, "scheduler_feedback": feedback_result}

def run_live(*, goal: str | dict[str, Any], source_dir: str | None = None,
             seed: int = 1101, max_steps: int = 60,
             output_dir: str | Path | None = None) -> dict[str, Any]:
    """Initialize the real B AtropineEnv, real B executor and real D monitor."""
    if not B_CODE_ROOT.is_dir():
        raise FileNotFoundError(f"B runtime code not found: {B_CODE_ROOT}")
    sys.path.insert(0, str(B_CODE_ROOT))
    from integration_abd_1103 import EnvironmentAdapter, ExecutionAgent

    out = Path(output_dir) if output_dir else ROOT.parent / "outputs" / "a-final" / (datetime.now().strftime("run_%Y%m%d_%H%M%S_") + uuid4().hex[:8])
    out.mkdir(parents=True, exist_ok=False)
    monitor = None
    try:
        with EnvironmentAdapter(source_dir=source_dir, max_steps=max_steps) as adapter:
            monitor = MonitoringAgent(action_space=adapter.action_space)
            if monitor.demo_mode is not False:
                raise RuntimeError("D demo_mode must be False for live SMPL integration")
            executor = ExecutionAgent(adapter, log_path=out / "B_events.jsonl")
            reset_observation = executor.reset(seed=seed)
            result = run_control_cycle(goal=goal, executor=executor, monitor=monitor,
                                       executor_id=EXECUTOR_ID, event_path=out / "A_events.jsonl")
            summary = dict(result["summary"])
            summary.update({"environment": "SMPL.AtropineEnvGym", "seed": seed,
                            "observation_space_shape": list(adapter.observation_space.shape),
                            "action_space_shape": list(adapter.action_space.shape),
                            "action_low": adapter.action_space.low.tolist(),
                            "action_high": adapter.action_space.high.tolist(),
                            "reset_observation_shape": list(np.asarray(reset_observation).shape),
                            "b_source_dir": adapter.source_dir})
            result["summary"] = summary
            (out / "A_summary.json").write_text(json.dumps(_plain(result), ensure_ascii=False, indent=2), encoding="utf-8")
            monitor.export_logs(out / "D_reviews.jsonl")
            return result
    except Exception as exc:
        if monitor is not None:
            monitor.export_logs(out / "D_reviews.jsonl")
        error_record = {"run_status": "blocked_or_failed", "exception_type": type(exc).__name__,
                        "exception": str(exc), "python_version": sys.version,
                        "note": "No successful A-main live SMPL run is claimed from this attempt."}
        (out / "run_error.json").write_text(json.dumps(error_record, ensure_ascii=False, indent=2), encoding="utf-8")
        raise


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="L3 four-module AtropineEnv integration runner")
    parser.add_argument("--source-dir", default=None, help="Optional path to B's SMPL source tree")
    parser.add_argument("--seed", type=int, default=1101)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--offline-preview", action="store_true", help="Only show A HTN plan; no environment initialization")
    args = parser.parse_args(argv)
    goal = {"task_type": "control_cycle", "description": "L3 one-cycle integration acceptance test",
            "priority": 3, "constraints": {"environment": "AtropineEnv"}}
    if args.offline_preview:
        decision = DecisionAgent(); task = decision.task_to_dict(decision.create_task(goal))
        print(json.dumps(_plain(task), ensure_ascii=False, indent=2))
        return
    result = run_live(goal=goal, source_dir=args.source_dir, seed=args.seed,
                      max_steps=args.max_steps, output_dir=args.output_dir)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    if result["summary"]["run_status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
