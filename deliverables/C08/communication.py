"""Local and ZeroMQ four-role protocol fixture, with a single simulated owner."""
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import time

from l3_system.scheduling.transport import TransportError, ZmqEndpoint

ROLES = ("C", "D", "B")


class Role:
    def __init__(self, name):
        self.name = name
        self.steps = 0
        self.receipts = {}

    def handle(self, request):
        identity = request.get("correlation_id")
        operation = request.get("operation")
        response = {"role": self.name, "correlation_id": identity, "status": "ok"}
        if request.get("role") != self.name or not isinstance(identity, str) or not identity:
            return {**response, "status": "invalid_identity"}
        if operation == "heartbeat":
            return {**response, "steps": self.steps} if self.name == "B" else response
        if self.name == "C" and operation == "allocate":
            return {**response, "winner": "fixture-worker"}
        if self.name == "D" and operation == "review":
            return {**response, "approved": request.get("action") in (-1, 0, 1)}
        if self.name == "B" and operation == "step":
            if identity in self.receipts:
                previous, result = self.receipts[identity]
                return result if previous == request else {**response, "status": "conflicting_duplicate"}
            if type(request.get("expected_steps")) is not int or request["expected_steps"] != self.steps:
                return {**response, "status": "stale_state", "steps": self.steps}
            if request.get("approved") is not True or request.get("action") not in (-1, 0, 1):
                return {**response, "status": "rejected", "steps": self.steps}
            self.steps += 1
            result = {**response, "steps": self.steps, "applied": request["action"]}
            self.receipts[identity] = (dict(request), dict(result))
            return result
        return {**response, "status": "invalid_operation"}


def serve(name, ready):
    role = Role(name)
    with ZmqEndpoint(server=True, timeout_ms=10_000) as endpoint:
        ready.send(endpoint.address)
        ready.close()
        while True:
            request = endpoint.receive()
            if request.get("operation") == "shutdown" and request.get("role") == name:
                endpoint.send({"status": "ok", "role": name,
                               "correlation_id": request["correlation_id"]})
                return
            endpoint.send(role.handle(request))


def check(call, check_exit=None):
    events = []

    def send(role, operation, identity, **payload):
        request = {"role": role, "operation": operation, "correlation_id": identity, **payload}
        response = call(role, request)
        if response.get("role") != role or response.get("correlation_id") != identity:
            raise TransportError("Reply correlation or role mismatch")
        events.append({"request": request, "response": response})
        return response

    for role in ROLES:
        assert send(role, "heartbeat", f"heartbeat-{role}")["status"] == "ok"
    assert send("C", "allocate", "round-1")["winner"] == "fixture-worker"
    authorization = send("D", "review", "round-1", action=1)
    action = {"action": 1, "approved": authorization["approved"], "expected_steps": 0}
    first = send("B", "step", "round-1", **action)
    assert first["steps"] == 1
    assert first == send("B", "step", "round-1", **action)
    assert send("B", "step", "round-1", **{**action, "action": 0})["status"] == "conflicting_duplicate"
    assert send("B", "step", "late-round", **action)["status"] == "stale_state"
    veto = send("D", "review", "veto-round", action=4)
    assert send("B", "step", "veto-round", action=4, approved=veto["approved"],
                expected_steps=1) == {"role": "B", "correlation_id": "veto-round",
                                      "status": "rejected", "steps": 1}
    if check_exit:
        check_exit(events)
        assert send("C", "allocate", "resume-round")["winner"] == "fixture-worker"
        approval = send("D", "review", "resume-round", action=0)
        assert send("B", "step", "resume-round", action=0, approved=approval["approved"],
                    expected_steps=1)["steps"] == 2
    return events


def run(mode):
    if mode == "local":
        roles = {name: Role(name) for name in ROLES}
        return {"mode": mode, "role_process_count": 1,
                "events": check(lambda name, request: roles[name].handle(request))}

    ctx = mp.get_context("spawn")
    addresses = {}
    processes = {}

    def start(role):
        ready, sender = ctx.Pipe(duplex=False)
        process = ctx.Process(target=serve, args=(role, sender))
        process.start()
        processes[role] = process
        sender.close()
        try:
            if not ready.poll(4):
                raise TransportError(f"{role} did not bind")
            try:
                addresses[role] = ready.recv()
            except EOFError as error:
                raise TransportError(f"{role} exited before binding") from error
        finally:
            ready.close()

    paused = False

    def call(role, request, timeout_ms=500):
        nonlocal paused
        if paused and role == "B" and request["operation"] == "step":
            raise TransportError("Dispatch is paused pending explicit recovery")
        try:
            with ZmqEndpoint(server=False, timeout_ms=timeout_ms, address=addresses[role]) as endpoint:
                endpoint.send(request)
                return endpoint.receive()
        except TransportError:
            paused = True
            raise

    def stop(role):
        call(role, {"role": role, "operation": "shutdown", "correlation_id": "shutdown"})
        processes[role].join(2)
        if processes[role].exitcode != 0:
            raise TransportError(f"{role} did not stop cleanly")

    def check_exit(events):
        nonlocal paused
        stop("C")
        started = time.monotonic()
        try:
            call("C", {"role": "C", "operation": "heartbeat", "correlation_id": "C-down"}, 100)
        except TransportError as error:
            events.append({"event": "heartbeat_timeout", "role": "C", "status": "takeover_required",
                           "elapsed_ms": round((time.monotonic() - started) * 1000, 2),
                           "reason": str(error)})
        else:
            raise AssertionError("Exited C process responded")
        try:
            call("B", {"role": "B", "operation": "step", "correlation_id": "paused-round",
                       "expected_steps": 1, "approved": True, "action": 0})
        except TransportError as error:
            assert "paused" in str(error)
        else:
            raise AssertionError("Step dispatched during takeover")
        state = call("B", {"role": "B", "operation": "heartbeat", "correlation_id": "B-during-pause"})
        assert state["steps"] == 1
        events.append({"event": "step_blocked_in_takeover", "observed_b_steps": state["steps"]})
        start("C")
        probe = call("C", {"role": "C", "operation": "heartbeat", "correlation_id": "C-recovery"})
        assert probe == {"role": "C", "correlation_id": "C-recovery", "status": "ok"}
        paused = False
        events.append({"event": "explicit_fixture_resume", "role": "A", "restart_probe": probe})

    try:
        for role in ROLES:
            start(role)
        events = check(call, check_exit)
        for role in ROLES:
            stop(role)
        return {"mode": mode, "role_process_count": 4, "events": events}
    finally:
        for process in processes.values():
            if process.is_alive():
                process.terminate()
            process.join(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("local", "zmq"))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.write_text(json.dumps(run(args.mode), indent=2) + "\n", encoding="utf-8")
