"""Run A's session through separate C, B and D processes on local ZeroMQ."""

import argparse
from contextlib import ExitStack
import json
import multiprocessing as mp
from pathlib import Path
from time import monotonic, sleep
from types import SimpleNamespace
from uuid import uuid4

import numpy as np

from l3_system.agents.monitoring_agent import MonitoringAgent
from l3_system.execution.environment_adapter import EnvironmentAdapter, to_plain
from l3_system.execution.execution_agent import ExecutionAgent
from l3_system.runtime import MultiAgentSystem
from deliverables.C10.learning_adapter import TrainingFixtureAdapter, frozen_policy
from l3_system.scheduling.transport import TransportError, ZmqEndpoint
from l3_system.agents.scheduling_agent import SchedulingAgent
from l3_system.scheduling.leased_scheduler import LeasedScheduler

METHODS = {
    "B": {"reset", "get_observation", "prepare_action", "execute_action",
          "reject_action_request", "cancel_action_request"},
    "D": {"check_request", "check_observation", "get_logs"},
    "C": {"register_tasks", "ready_tasks", "schedule", "assignment_for_execution",
          "update_feedback", "get_active_assignment", "get_task_status",
          "get_assignment_history", "block", "record_unknown", "reconcile_executed"},
}
STEP_METHODS = {"execute_action", "reject_action_request"}


class SchedulingService(SchedulingAgent):
    """Bind feedback to the current award at the wire boundary."""

    def __init__(self, lease_seconds):
        super().__init__()
        self.lease_seconds = lease_seconds
        self.unknown_requests = {}

    def schedule(self, task, bids):
        LeasedScheduler.validate_bid_times(bids, monotonic())
        result = super().schedule(task, bids)
        key = self._key(task)
        if result["assignment_status"] == "awarded":
            identity = {"assignment_id": uuid4().hex, "attempt": len(self.assignment_history[key]),
                        "lease_deadline": monotonic() + self.lease_seconds}
            self.assignments[key].update(identity)
            self.assignment_history[key][-1].update(identity)
            result.update(identity)
        return result

    def assignment_for_execution(self, assignment):
        active = self.get_active_assignment(assignment["task_id"], assignment.get("subtask_id"))
        if active is None or any(assignment.get(k) != v for k, v in active.items()):
            raise ValueError("Assignment is inactive or mismatched")
        if monotonic() >= active["lease_deadline"]:
            raise TimeoutError("Assignment lease expired before dispatch")
        return active

    def update_feedback(self, feedback, assignment_id=None):
        current = self.get_assignment(feedback["task_id"], feedback.get("subtask_id"))
        if (None if current is None else current["assignment_id"]) != assignment_id:
            raise ValueError("Feedback has the wrong assignment_id")
        if assignment_id in self.unknown_requests:
            raise RuntimeError("Execution result unknown; reconciliation required")
        if current is not None and self.get_active_assignment(
                feedback["task_id"], feedback.get("subtask_id")) is not None:
            if monotonic() >= current["lease_deadline"]:
                raise TimeoutError("Expired active lease requires reconciliation")
        return super().update_feedback(feedback)

    def record_unknown(self, feedback, assignment_id, request_id):
        active = self.get_active_assignment(feedback["task_id"], feedback["subtask_id"])
        if not active or active["assignment_id"] != assignment_id or not request_id:
            raise ValueError("Uncertain execution must match an active assignment")
        previous = self.unknown_requests.setdefault(assignment_id, request_id)
        if previous != request_id:
            raise ValueError("Conflicting unknown request identity")
        raise RuntimeError("Execution result unknown; reconciliation required")

    def reconcile_executed(self, assignment, receipt):
        active = self.get_active_assignment(assignment["task_id"], assignment["subtask_id"])
        request_id = self.unknown_requests.get(assignment["assignment_id"])
        if (active != assignment or request_id is None or receipt.get("request_id") != request_id
                or receipt.get("task_id") != assignment["task_id"]
                or receipt.get("subtask_id") != assignment["subtask_id"]
                or receipt.get("executed") is not True or receipt.get("success") is not True):
            raise ValueError("Owner receipt does not resolve the active assignment")
        result = super().update_feedback({"task_id": assignment["task_id"],
                                         "subtask_id": assignment["subtask_id"],
                                         "execution_status": "completed"})
        del self.unknown_requests[assignment["assignment_id"]]
        return result

    def block(self, task_key):
        if self.tasks[task_key]["status"] != "pending":
            raise ValueError("Only pending dependent work can be blocked")
        self.tasks[task_key]["status"] = "blocked"

    def locks(self):
        return dict(self.active_agent_tasks)


def _space(box):
    return {"shape": list(box.shape), "low": box.low.tolist(), "high": box.high.tolist(),
            "dtype": str(box.dtype)}


def _view(box):
    return SimpleNamespace(shape=tuple(box["shape"]), low=np.asarray(box["low"]),
                           high=np.asarray(box["high"]), dtype=np.dtype(box["dtype"]))


def _serve(role, ready, options):
    with ExitStack() as stack:
        if role == "B":
            if options["backend"] == "atropine":
                environment = EnvironmentAdapter(max_steps=options["max_steps"])
            else:
                environment = TrainingFixtureAdapter()
            adapter = stack.enter_context(environment)
            instance = ExecutionAgent(adapter, log_path=options["b_log"])
        elif role == "D":
            instance = MonitoringAgent(_view(options["action"]), _view(options["observation"]))
        else:
            instance = SchedulingService(options["lease_seconds"])
        channel = stack.enter_context(ZmqEndpoint(server=True, timeout_ms=60_000,
                                                 max_frame=1_048_576))
        ready.send(channel.address)
        ready.close()
        receipts = {}
        drop_next = options.get("drop_first_step_reply", False)
        while True:
            request = channel.receive()
            identity = request["correlation_id"]
            operation = request["operation"]
            replay = receipts.get(identity)
            if replay is not None:
                old_request, response = replay
                if old_request != request:
                    response = {"role": role, "correlation_id": identity,
                                "error": "Conflicting correlation ID"}
                channel.send(response)
                continue
            response = {"role": role, "correlation_id": identity}
            try:
                if request["role"] != role:
                    raise ValueError("Wrong process role")
                if operation == "close":
                    channel.send({**response, "value": None})
                    return
                if operation == "heartbeat":
                    value = True
                elif operation == "spaces" and role == "B":
                    value = {"action": _space(adapter.action_space),
                             "observation": _space(adapter.observation_space)}
                elif operation == "lookup_execution_receipt" and role == "B":
                    request_id, = request["args"]
                    cached = receipts.get(f"execute:{request_id}")
                    value = cached[1].get("value") if cached else None
                elif operation == "state" and role == "B":
                    value = {k: getattr(adapter, k) for k in
                             ("step_calls", "state_version", "step_count", "done")}
                    value["pending"] = list(instance._pending)
                    value["metadata"] = getattr(adapter, "metadata", {})
                elif operation == "require_ready" and role == "B":
                    value = adapter.require_ready()
                elif operation == "locks" and role == "C":
                    value = instance.locks()
                elif operation in METHODS[role]:
                    value = getattr(instance, operation)(*request["args"], **request["kwargs"])
                else:
                    raise ValueError("Operation is outside the role contract")
                response["value"] = to_plain(value)
            except Exception as error:
                response["error"] = f"{type(error).__name__}: {error}"
            if role == "B" and operation in STEP_METHODS:
                receipts[identity] = (request, response)
            if drop_next and operation == "execute_action" and (response.get("value") or {}).get("executed"):
                drop_next = False
                # Reply after A's receive deadline. B's completed result stays in its ledger.
                sleep(options["reply_delay_seconds"])
            channel.send(response)


class RoleProcess:
    def __init__(self, role, options, trace, *, timeout_ms):
        self.role = role
        self.timeout_ms = timeout_ms
        self.trace = trace
        ctx = mp.get_context("spawn")
        ready, sender = ctx.Pipe(duplex=False)
        self.process = ctx.Process(target=_serve, args=(role, sender, options))
        self.process.start()
        sender.close()
        try:
            if not ready.poll(20):
                raise TransportError(f"{role} did not start")
            self.address = ready.recv()
        except (OSError, TransportError, EOFError) as error:
            self.close(graceful=False)
            raise TransportError(f"{role} exited or timed out during startup") from error
        finally:
            ready.close()

    def call(self, operation, *args, correlation_id=None, **kwargs):
        identity = correlation_id or uuid4().hex
        request = to_plain({"role": self.role, "operation": operation, "correlation_id": identity,
                            "args": args, "kwargs": kwargs})
        try:
            with ZmqEndpoint(server=False, timeout_ms=self.timeout_ms, address=self.address,
                             max_frame=1_048_576) as client:
                client.send(request)
                reply = client.receive()
        except TransportError as error:
            self.trace.append({"role": self.role, "operation": operation,
                               "correlation_id": identity, "transport_error": str(error)})
            raise
        self.trace.append({"role": self.role, "operation": operation, "correlation_id": identity,
                           "reply": reply})
        if (reply.get("role"), reply.get("correlation_id")) != (self.role, identity):
            raise TransportError("Reply identity mismatch")
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return reply["value"]

    def close(self, *, graceful=True):
        if graceful and self.process.is_alive():
            try:
                self.call("close")
            except (OSError, RuntimeError):
                pass
        self.process.join(1)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(2)


class AdapterProxy:
    def __init__(self, remote, spaces):
        self.remote = remote
        self.action_space = _view(spaces["action"])
        self.observation_space = _view(spaces["observation"])

    def require_ready(self):
        self.remote.call("require_ready")

    def __getattr__(self, key):
        if key not in {"step_calls", "state_version", "step_count", "done", "metadata"}:
            raise AttributeError(key)
        return self.remote.call("state")[key]


class RoleProxy:
    demo_mode = False

    def __init__(self, remote):
        self.remote = remote

    def __getattr__(self, key):
        if key not in METHODS[self.remote.role]:
            raise AttributeError(key)
        return lambda *args, **kwargs: self.remote.call(key, *args, **kwargs)


class ExecutionProxy(RoleProxy):
    @property
    def _pending(self):
        return self.remote.call("state")["pending"]

    def execute_action(self, request, review):
        self.last_args = (request, review)
        self.last_dispatch = self.remote.call("execute_action", request, review,
                                              correlation_id=f'execute:{request["request_id"]}')
        return self.last_dispatch


class PendingTask:
    def __init__(self, remote, key):
        self.remote = remote
        self.key = key

    def __setitem__(self, field, value):
        if (field, value) != ("status", "blocked"):
            raise ValueError("Only A's dependency block is supported")
        self.remote.call("block", self.key)


class Tasks:
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def __getitem__(self, key):
        return PendingTask(self.scheduler.remote, key)


class SchedulingProxy(RoleProxy):
    def __init__(self, remote, execution):
        super().__init__(remote)
        self.execution = execution
        self.tasks = Tasks(self)
        self.current_ids = {}

    def schedule(self, task, bids):
        result = self.remote.call("schedule", task, bids)
        self.current_ids[(task["task_id"], task.get("subtask_id"))] = result.get("assignment_id")
        return result

    def update_feedback(self, feedback):
        key = (feedback["task_id"], feedback.get("subtask_id"))
        assignment_id = self.current_ids.get(key)
        if feedback.get("failure_reason") == "execution_result_unknown":
            request, _ = getattr(self.execution, "last_args", (None, None))
            if request is None or (request["task_id"], request["subtask_id"]) != key:
                raise RuntimeError("Execution result unknown without matching request; keep lock")
            return self.remote.call("record_unknown", feedback, assignment_id, request["request_id"])
        return self.remote.call("update_feedback", feedback, assignment_id)

    @property
    def active_agent_tasks(self):
        return self.remote.call("locks")


def verify_lost_reply(system, execution, scheduler, goal, generator):
    try:
        system.run_cycle(goal, action_generator=generator)
    except RuntimeError as error:
        assert "reconciliation required" in str(error)
    else:
        raise AssertionError("Missing execution reply did not trigger takeover")
    assert system.state == "takeover_requested"
    request, _ = execution.last_args
    assignment = scheduler.get_active_assignment(request["task_id"], request["subtask_id"])
    assert assignment and scheduler.active_agent_tasks
    try:
        system.resume(operator="c-acceptance", reason="premature resume probe")
    except RuntimeError as error:
        assert "Outstanding assignments" in str(error)
    else:
        raise AssertionError("Unresolved assignment allowed resume")
    receipt = execution.remote.call("lookup_execution_receipt", request["request_id"])
    current = execution.remote.call("state")
    assert receipt["step_calls"] == current["step_calls"] == 1
    assert receipt["state_version"] == current["state_version"] and not current["pending"]
    try:
        scheduler.remote.call("reconcile_executed", assignment,
                              {**receipt, "request_id": "older-request"})
    except RuntimeError as error:
        assert "does not resolve" in str(error)
    else:
        raise AssertionError("Out-of-order receipt released the current lock")
    assert scheduler.active_agent_tasks
    scheduler.remote.call("reconcile_executed", assignment, receipt)
    assert not scheduler.active_agent_tasks and execution.adapter.step_calls == 1
    system.resume(operator="c-acceptance", reason="B cached result confirmed without replay")
    return {"step_calls": 1, "takeover": True, "premature_resume_blocked": True,
            "out_of_order_receipt_rejected": True, "reconciled_without_replay": True,
            "request_id": request["request_id"], "assignment_id": assignment["assignment_id"]}


def run(out, *, seed, timeout_ms, backend, lost_reply=False):
    out.mkdir(parents=True, exist_ok=False)
    trace = []
    runs = {"backend": backend, "seed": seed, "rpc_timeout_ms": timeout_ms}
    generator = None
    if backend == "learning-fixture":
        generator, runs["weight_sha256"] = frozen_policy()
    goal = {"task_type": "control_cycle", "description": f"C distributed {backend} acceptance"}
    with ExitStack() as stack:
        def start(role, **options):
            peer = RoleProcess(role, options, trace, timeout_ms=timeout_ms)
            stack.callback(peer.close)
            return peer

        d_path = out / "D_reviews.json"
        try:
            execution = ExecutionProxy(start(
                "B", backend=backend, max_steps=12, b_log=str(out / "B_events.jsonl"),
                drop_first_step_reply=lost_reply, reply_delay_seconds=1.5 * timeout_ms / 1000))
            spaces = execution.remote.call("spaces")
            execution.adapter = AdapterProxy(execution.remote, spaces)
            monitor = RoleProxy(start("D", **spaces))
            scheduler = SchedulingProxy(start("C", lease_seconds=10), execution)
            system = MultiAgentSystem(executor=execution, monitor=monitor,
                                      scheduler=scheduler, event_path=out / "A_events.jsonl")
            execution.reset(seed=seed)
            if lost_reply:
                runs["lost_execution_reply"] = verify_lost_reply(system, execution, scheduler, goal, generator)
            runs["normal"] = system.run_cycle(goal, action_generator=generator)["summary"]
            baseline = 1 if lost_reply else 0
            assert runs["normal"]["step_delta"] == 1 and runs["normal"]["run_status"] == "completed"
            result = execution.last_dispatch
            request, review = execution.last_args
            replay = execution.execute_action(request, review)
            assert replay == result and execution.adapter.step_calls == baseline + 1
            try:
                execution.remote.call("execute_action", request, {**review, "decision": "reject"},
                                      correlation_id=f'execute:{request["request_id"]}')
            except RuntimeError as error:
                assert "Conflicting" in str(error)
            else:
                raise AssertionError("Conflicting execution replay accepted")
            assert execution.adapter.step_calls == baseline + 1
            runs["replay"] = {"identical_cached": True, "conflict_rejected": True, "step_calls": baseline + 1}

            scheduler.remote.close()
            assert not scheduler.remote.process.is_alive()
            try:
                system.run_cycle(goal, action_generator=generator)
            except TransportError:
                assert system.state == "takeover_requested"
            else:
                raise AssertionError("C exit did not stop A")
            try:
                system.run_cycle(goal, action_generator=generator)
            except RuntimeError:
                assert execution.adapter.step_calls == baseline + 1
            else:
                raise AssertionError("Paused system accepted another cycle")
            runs["process_exit"] = {"state": system.state, "step_calls": execution.adapter.step_calls,
                                    "dispatch_blocked": True}
            scheduler.remote = start("C", lease_seconds=10)
            assert scheduler.remote.call("heartbeat")
            runs["resume"] = system.resume(operator="c-acceptance", reason="C is responsive; B did not step")
            runs["recovered"] = system.run_cycle(goal, action_generator=generator)["summary"]
            assert runs["recovered"]["step_calls"] == baseline + 2
            assert runs["recovered"]["all_C_locks_released"]
            runs["rejected"] = system.run_cycle(
                goal, action_generator=lambda _, space, __: space.high + 1)["summary"]
            assert runs["rejected"]["failure_reason"] == "safety_rejected"
            assert runs["rejected"]["step_calls"] == baseline + 2
            assert runs["rejected"]["all_C_locks_released"]
            d_path.write_text(json.dumps(monitor.get_logs(), ensure_ascii=False, indent=2), encoding="utf-8")
        finally:
            (out / "wire.jsonl").write_text("".join(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n"
                                                   for event in trace), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(runs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1101)
    parser.add_argument("--timeout-ms", type=int, default=750)
    parser.add_argument("--backend", choices=("atropine", "learning-fixture"), default="atropine")
    parser.add_argument("--lost-reply", action="store_true")
    args = parser.parse_args()
    run(args.out, seed=args.seed, timeout_ms=args.timeout_ms, backend=args.backend,
        lost_reply=args.lost_reply)
