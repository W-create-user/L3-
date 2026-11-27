"""One local orchestration path for A planning, C scheduling, B execution and D review."""
from __future__ import annotations

import argparse
import copy
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np

from l3_system.config import EXECUTOR_ID
from l3_system.decision_agent import DecisionAgent
from l3_system.agents.scheduling_agent import SchedulingAgent
from l3_system.agents.monitoring_agent import MonitoringAgent
from l3_system.adapters.a_to_c_adapter import convert_subtasks_for_c
from l3_system.adapters.assignment_adapter import assert_winner, build_b_task
from l3_system.adapters.b_to_d_adapter import make_d_safety_checker
from l3_system.adapters.feedback_adapter import execution_result_to_feedback
from l3_system.execution import EnvironmentAdapter, ExecutionAgent

OWNER_BY_TASK = {
    "collect_observation": EXECUTOR_ID,
    "evaluate_goal": "DecisionAgent-01",
    "select_hold_policy": "DecisionAgent-01",
    "select_adjust_policy": "DecisionAgent-01",
    "verify_goal": "DecisionAgent-01",
    "evaluate_state": "DecisionAgent-01",
    "return_state_result": "DecisionAgent-01",
    "generate_action_intent": "DecisionAgent-01",
    "safety_review": "MonitoringAgent-01",
    "execute_approved_action": EXECUTOR_ID,
    "verify_result": "DecisionAgent-01",
}


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def configured_bids(task):
    """Only the installed, capable role bids; no fictitious backup worker."""
    return [{"task_id": task["task_id"], "subtask_id": task["subtask_id"],
             "agent_id": OWNER_BY_TASK[task["task_type"]],
             "estimated_cost": 1.0, "capability_match": 1.0, "load": 0.0}]


class NoAvailableWorker(RuntimeError):
    """C did not award a worker to the ready subtask."""


class DependencyBlocked(RuntimeError):
    """The scheduler did not release the subtask's prerequisites."""


def _default_action_generator(observation, action_space, attempt=0):
    from l3_system.execution.local_policy import space_midpoint
    return space_midpoint(observation, action_space)


def run_control_cycle(*, goal, executor, monitor, executor_id=EXECUTOR_ID,
                      scheduler=None, decision=None, action_generator=None, event_path=None,
                      goal_cycle=None, bid_provider=None):
    """Run one HTN chain. Every awarded stage returns terminal feedback to C.

    B owns pending ActionRequests and invalidates them on use/reset. D reviews
    the same candidate before dispatch and immediately before B advances SMPL.
    Failures stop dependent work; no automatic repetition of a physical step.
    """
    if getattr(monitor, "demo_mode", True) is not False:
        raise RuntimeError("Formal run requires D initialized with the live action_space")
    if executor_id != EXECUTOR_ID:
        raise ValueError(f"Configured executor identity must be {EXECUTOR_ID!r}")
    from l3_system.planning.goal_runtime import GoalDeadlineExceeded

    decision = decision if decision is not None else DecisionAgent()
    parent = (goal_cycle.parent_task() if goal_cycle is not None
              else decision.task_to_dict(decision.create_task(goal)))
    # Do not report a production goal as satisfied when its predicates have no
    # implementation. The finite MVP only verifies its documented task chain.
    if goal_cycle is None and (parent['preconditions']
            or parent['completion_criteria'] != ['all_subtasks_completed']
            or set(parent['constraints']) - {'environment'}
            or parent['constraints'].get('environment', 'AtropineEnv') != 'AtropineEnv'):
        raise NotImplementedError('Unsupported goal predicates or process constraints')
    tasks = convert_subtasks_for_c(parent)
    scheduler = scheduler if scheduler is not None else SchedulingAgent()
    scheduler.register_tasks(tasks)
    checker = make_d_safety_checker(monitor)
    metadata = getattr(executor.adapter, "metadata", {})
    is_mab = metadata.get("environment_id") == "MAbEnvGym"
    if action_generator is not None:
        generator, controller_note = action_generator, "Caller-supplied simulation policy."
    elif is_mab:
        # B03's verification script uses this source reference for its steady-run case.
        reference_action = executor.adapter._env.steady_actions.copy()
        generator = lambda *_: reference_action.copy()
        controller_note = "B03 mAb steady reference action."
    else:
        generator, controller_note = _default_action_generator, "Atropine space midpoint interface baseline."
    events, stages = [], []
    event_file = Path(event_path) if event_path else None
    if event_file:
        event_file.parent.mkdir(parents=True, exist_ok=True)
    def emit(event, **payload):
        record = {"timestamp": datetime.now(timezone.utc).isoformat(),
                  "event": event, "task_id": parent["task_id"], **_plain(payload)}
        events.append(record)
        if event_file:
            with event_file.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    before = executor.adapter.step_calls
    observation = goal_cycle.observation if goal_cycle is not None else None
    request = review = execution = None
    failure = None
    execution_subtask = next((s for s in parent["subtasks"]
                              if s["name"] == "execute_approved_action"), None)
    for task, subtask in zip(tasks, parent["subtasks"]):
        sid, name = subtask["subtask_id"], subtask["name"]
        started = time.perf_counter()
        status, reason, payload = "completed", None, {}
        dispatch_started = False
        collecting_bids = False
        try:
            if sid not in {t["subtask_id"] for t in scheduler.ready_tasks()}:
                raise DependencyBlocked("HTN dependency not completed")
            collecting_bids = True
            bids = (configured_bids if bid_provider is None else bid_provider)(task)
            collecting_bids = False
            awarded = scheduler.schedule(task, bids)
            if awarded["assignment_status"] == "unassigned":
                raise NoAvailableWorker("No eligible bid for the subtask")
            assignment = scheduler.assignment_for_execution(awarded)
            assert_winner(assignment, OWNER_BY_TASK[name])
            if assignment["task_id"] != parent["task_id"] or assignment["subtask_id"] != sid:
                raise ValueError("Assignment identity mismatch")
            if goal_cycle is not None:
                if name == "execute_approved_action":
                    goal_cycle.facts["resource_available"] = True
                goal_cycle.require_ready(name)
            emit("assignment_awarded", subtask_id=sid, task_type=name,
                 bids=bids, assignment=assignment, ranked_bids=awarded.get("ranked_bids", []))
            if name == "collect_observation":
                count = executor.adapter.step_calls
                observation = executor.get_observation()
                arr = np.asarray(observation, dtype=float)
                if arr.shape != executor.adapter.observation_space.shape or not np.isfinite(arr).all():
                    raise ValueError("Invalid observation")
                if executor.adapter.step_calls != count:
                    raise RuntimeError("Observation read advanced the environment")
                payload = {"observation": observation}
            elif name in {"evaluate_goal", "evaluate_state"}:
                if observation is None:
                    raise ValueError("Missing current observation")
                payload = ({"evaluation": goal_cycle.plan.evaluation,
                            "numeric_process_target": goal_cycle.target_range}
                           if goal_cycle is not None else
                           {"evaluation": "finite_current_observation", "numeric_process_target": None})
            elif name in {"select_hold_policy", "select_adjust_policy"}:
                payload = {"policy": goal_cycle.plan.method}
            elif name == "verify_goal":
                payload = {"evaluation": goal_cycle.plan.evaluation}
            elif name == "return_state_result":
                payload = {"observation": observation}
            elif name == "generate_action_intent":
                action = generator(observation, executor.adapter.action_space, 0)
                request = executor.prepare_action(parent, action=action, subtask=execution_subtask)
                if goal_cycle is not None:
                    goal_cycle.check_action(request["action"])
                if executor.adapter.step_calls != before:
                    raise RuntimeError("Preparing an action advanced the environment")
                payload = {"request": request}
            elif name == "safety_review":
                review = checker(copy.deepcopy(request), copy.deepcopy(observation))
                payload = {"safety_decision": review}
                if review["decision"] != "pass":
                    status, reason = "failed", "safety_rejected"
                    execution = executor.reject_action_request(request, review)
            elif name == "execute_approved_action":
                assert_winner(assignment, executor_id)
                build_b_task(assignment, parent, subtask)
                if review is None or review["decision"] != "pass":
                    raise RuntimeError("No approved preflight review")
                # Fresh synchronous review is bound to the original pending request.
                final_review = checker(copy.deepcopy(request), executor.get_observation())
                if goal_cycle is not None:
                    goal_cycle.require_dispatch()
                dispatch_started = final_review["decision"] == "pass"
                execution = executor.execute_action(request, final_review)
                if (execution.get("task_id") != parent["task_id"] or execution.get("subtask_id") != sid
                        or execution.get("request_id") != request["request_id"]):
                    raise ValueError("B feedback identity mismatch")
                feedback = execution_result_to_feedback(execution)
                if feedback is None or feedback["execution_status"] not in {"completed", "failed"}:
                    raise ValueError("Synchronous B execution must return terminal feedback")
                status, reason = feedback["execution_status"], feedback.get("failure_reason")
                if execution.get("execution_status") == "rejected":
                    reason = "safety_rejected"
                payload = {"execution_result": execution}
            elif name == "verify_result":
                if not (execution and execution.get("request_id") == request["request_id"]
                        and execution.get("state_version") == executor.adapter.state_version
                        and execution.get("success") is True
                        and execution.get("executed") is True
                        and executor.adapter.step_calls - before == 1):
                    raise RuntimeError("Expected exactly one successful environment step")
                observed = executor.get_observation()
                state_review = monitor.check_observation(observed, parent["task_id"])
                if state_review["approved"] is not True:
                    status, reason = "failed", "invalid_post_execution_observation"
                elif not np.array_equal(execution.get("observation"), observed):
                    status, reason = "failed", "execution_result_unknown"
                payload = {"verified": status == "completed", "step_delta": 1,
                           "state_review": state_review}
            else:
                raise ValueError(f"Unsupported HTN stage: {name}")
            if goal_cycle is not None and status == "completed":
                goal_cycle.complete(name)
        except Exception as exc:
            status = "failed"
            if dispatch_started:
                reason = "execution_result_unknown"
            elif isinstance(exc, GoalDeadlineExceeded):
                reason = "goal_deadline_exceeded"
            elif collecting_bids and isinstance(exc, TimeoutError):
                reason = "message_timeout"
            elif isinstance(exc, NoAvailableWorker) or (collecting_bids and isinstance(exc, ConnectionError)):
                reason = "no_resource"
            elif isinstance(exc, DependencyBlocked):
                reason = "dependency_blocked"
            else:
                reason = "safety_review_error" if name == "safety_review" else "module_error"
            payload = {"exception_type": type(exc).__name__, "exception": str(exc)}
            emit("module_error", subtask_id=sid, task_type=name, **payload)
        feedback = {"task_id": parent["task_id"], "subtask_id": sid,
                    "execution_status": status, "failure_reason": reason,
                    "exclude_failed_agent": False}
        try:
            scheduler.update_feedback(feedback)
            if scheduler.get_active_assignment(parent["task_id"], sid) is not None:
                raise RuntimeError("Terminal feedback did not release C's lock")
        except Exception:
            if request is not None:
                executor.cancel_action_request(request["request_id"], "feedback_release_unconfirmed")
            raise
        stages.append({"subtask_id": sid, "task_type": name, "execution_status": status,
                       "duration_seconds": time.perf_counter() - started, **_plain(payload)})
        emit("subtask_completed" if status == "completed" else "subtask_failed",
             subtask_id=sid, task_type=name, feedback=feedback)
        if status != "completed":
            failure = reason
            break
    if failure and request is not None:
        executor.cancel_action_request(request["request_id"], failure)
    for subtask in parent["subtasks"]:
        subtask["status"] = scheduler.get_task_status(parent["task_id"], subtask["subtask_id"])
        if failure and subtask["status"] == "pending":
            subtask["status"] = "blocked"
            scheduler.tasks[subtask["subtask_id"]]["status"] = "blocked"
    parent["status"] = "failed" if failure else "completed"
    summary = {
        "run_status": "incomplete_or_failed" if failure else "completed",
        "failure_reason": failure, "task_id": parent["task_id"], "task_type": parent["task_type"],
        "subtask_count": len(tasks),
        "completed_subtasks": sum(s["status"] == "completed" for s in parent["subtasks"]),
        "subtask_statuses": {s["subtask_id"]: s["status"] for s in parent["subtasks"]},
        "step_calls": executor.adapter.step_calls, "step_delta": executor.adapter.step_calls - before,
        "step_count": executor.adapter.step_count, "executor_id": executor_id,
        "d_demo_mode": monitor.demo_mode, "bids_are_live": False,
        "bids_note": ("Caller-supplied bids; execution is limited to the installed role owners."
                      if bid_provider is not None else "One installed capable worker per role; local deterministic bids."),
        "all_C_locks_released": not scheduler.active_agent_tasks,
        "controller_note": controller_note,
        "r03_enabled": getattr(getattr(monitor, "process_rules", None), "enabled", False),
    }
    if goal_cycle is not None:
        summary.update(goal_id=goal_cycle.plan.goal_id, htn_method=goal_cycle.plan.method,
                       parent_plan_id=goal_cycle.plan.parent_plan_id)
    emit("A_task_feedback", task=parent, summary=summary)
    return {"summary": summary, "parent_task": _plain(parent), "c_tasks": _plain(tasks),
            "stage_results": stages, "events": events,
            "assignments": [scheduler.get_assignment_history(parent["task_id"], t["subtask_id"]) for t in tasks],
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
    from l3_system.adapters.feedback_adapter import update_scheduler_from_result
    feedback_result = update_scheduler_from_result(scheduler, result)
    return {"execution_result": result, "scheduler_feedback": feedback_result}

def run_live(*, goal="执行一次仿真控制任务", source_dir=None, seed=1101,
             max_steps=60, output_dir=None, cycles=1, scenario="normal",
             environment="AtropineEnvGym", goal_spec=None, goal_registry=None, timeout_seconds=None,
             bid_provider=None):
    if type(cycles) is not int or cycles < 1:
        raise ValueError("cycles must be a positive integer")
    if scenario not in {"normal", "reject", "inspect"}:
        raise ValueError("Unknown scenario")
    if (goal_spec is None) != (goal_registry is None):
        raise ValueError("goal_spec and goal_registry must be supplied together")
    if goal_spec is not None and (scenario != "normal" or cycles != 1):
        raise ValueError("A numeric goal manages its own cycles and requires the normal scenario")
    from l3_system.planning.goal_runtime import open_environment
    out = Path(output_dir) if output_dir else Path("outputs") / (datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid4().hex[:8])
    out.mkdir(parents=True, exist_ok=False)
    monitor = None
    try:
        with open_environment(environment, source_dir=source_dir, max_steps=max_steps,
                              atropine_factory=EnvironmentAdapter) as adapter:
            monitor = MonitoringAgent(action_space=adapter.action_space,
                                      observation_space=adapter.observation_space)
            executor = ExecutionAgent(adapter, log_path=out / "B_events.jsonl")
            executor.reset(seed=seed)
            from l3_system.runtime import MultiAgentSystem
            system = MultiAgentSystem(executor=executor, monitor=monitor,
                                      event_path=out / "events.jsonl", bid_provider=bid_provider)
            results = []
            if goal_spec is not None:
                from l3_system.planning.goal_runtime import run_numeric_goal
                result = run_numeric_goal(system, goal_spec, goal_registry,
                                          environment=environment, timeout_seconds=timeout_seconds)
                result["summary"].update(seed=seed, adapter_max_steps=max_steps,
                                         total_step_calls=adapter.step_calls)
                (out / "result.json").write_text(json.dumps(_plain(result), ensure_ascii=False, indent=2),
                                                 encoding="utf-8")
                monitor.export_logs(out / "D_reviews.jsonl")
                return result
            for cycle in range(cycles):
                if adapter.done:
                    break
                result = system.run_cycle(
                    goal="检查当前仿真状态" if scenario == "inspect" else goal,
                    action_generator=(lambda obs, space, attempt: space.high.astype(float) + 1) if scenario == "reject" else None)
                results.append(result)
                if result["summary"]["run_status"] != "completed":
                    break
            summary = dict(results[-1]["summary"])
            summary.update({"environment": f"SMPL.{environment}", "seed": seed,
                            "scenario": scenario, "requested_cycles": cycles,
                            "finished_cycles": len(results), "total_step_calls": adapter.step_calls,
                            "observation_space_shape": list(adapter.observation_space.shape),
                            "action_space_shape": list(adapter.action_space.shape),
                            "action_low": adapter.action_space.low.tolist(),
                            "action_high": adapter.action_space.high.tolist()})
            if summary["run_status"] == "completed" and len(results) < cycles:
                summary["run_status"] = "incomplete_or_failed"
                summary["failure_reason"] = "episode_ended_before_requested_cycles"
            result = {"summary": summary, "cycles": results}
            (out / "result.json").write_text(json.dumps(_plain(result), ensure_ascii=False, indent=2), encoding="utf-8")
            monitor.export_logs(out / "D_reviews.jsonl")
            return result
    except Exception as exc:
        if monitor is not None:
            monitor.export_logs(out / "D_reviews.jsonl")
        (out / "run_error.json").write_text(json.dumps({"run_status": "blocked_or_failed",
            "exception_type": type(exc).__name__, "exception": str(exc)}, ensure_ascii=False, indent=2), encoding="utf-8")
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description="四智能体 HTN → CNP → 安全监管 → SMPL 执行闭环")
    parser.add_argument("--transport", choices=["local", "zmq"], default="local",
                        help="Local runtime or the Atropine ZeroMQ integration exercise")
    parser.add_argument("--scenario", choices=["normal", "reject", "inspect"], default="normal")
    parser.add_argument("--source-dir", help="External SMPL source tree (defaults to vendor/smpl)")
    parser.add_argument("--seed", type=int, default=1101)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--output-dir", "--output", dest="output_dir")
    parser.add_argument("--offline-preview", action="store_true")
    parser.add_argument("--environment", choices=["AtropineEnvGym", "MAbEnvGym"], default="AtropineEnvGym")
    parser.add_argument("--goal-spec", type=Path)
    parser.add_argument("--goal-registry", type=Path)
    parser.add_argument("--goal-timeout", type=float, help="Wall-clock seconds for an explicitly configured numeric goal")
    args = parser.parse_args(argv)
    if args.transport == "zmq":
        if (args.environment != "AtropineEnvGym" or args.scenario != "normal"
                or args.cycles != 1 or args.max_steps != 60 or args.source_dir
                or args.goal_spec or args.goal_registry or args.goal_timeout is not None
                or args.offline_preview or not args.output_dir):
            parser.error("ZeroMQ requires an output directory and supports the standard Atropine exercise only")
        from deliverables.C10.live import run as run_zmq
        out = Path(args.output_dir)
        run_zmq(out, seed=args.seed, timeout_ms=750, backend="atropine")
        print((out / "summary.json").read_text(encoding="utf-8"))
        return
    if args.offline_preview:
        goal = "检查当前仿真状态" if args.scenario == "inspect" else "执行一次仿真控制任务"
        print(json.dumps(DecisionAgent.task_to_dict(DecisionAgent().create_task(goal)), ensure_ascii=False, indent=2))
        return
    result = run_live(source_dir=args.source_dir, seed=args.seed, max_steps=args.max_steps,
                      output_dir=args.output_dir, cycles=args.cycles, scenario=args.scenario,
                      environment=args.environment,
                      goal_spec=json.loads(args.goal_spec.read_text(encoding="utf-8")) if args.goal_spec else None,
                      goal_registry=json.loads(args.goal_registry.read_text(encoding="utf-8")) if args.goal_registry else None,
                      timeout_seconds=args.goal_timeout)
    summary = result["summary"]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.goal_spec:
        if summary["status"] != "succeeded":
            raise SystemExit(1)
        return
    expected_reject = (args.scenario == "reject" and summary["failure_reason"] == "safety_rejected"
                       and summary["total_step_calls"] == 0)
    completed = (summary["run_status"] == "completed" and summary["finished_cycles"] == args.cycles)
    if not summary["all_C_locks_released"] or not (expected_reject or completed):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
