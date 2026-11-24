"""Loopback Flask console for a single local A simulation session."""
import argparse
import json
import math
from pathlib import Path
from secrets import compare_digest, token_urlsafe
from threading import Lock

from flask import Flask, abort, redirect, render_template, request, url_for

from l3_system.main import _plain
from l3_system.planning.goal_runtime import open_environment
from .operator_session import OperatorSession


def create_app(system, *, environment, goal=None, registry=None, separate_confirmation=False):
    """The CLI opts into two-step recovery; existing D HTTP harnesses retain their contract."""
    app = Flask(__name__)
    app.config["TRUSTED_HOSTS"] = ["localhost", "127.0.0.1"]
    session = OperatorSession(system, environment, goal, registry, separate_confirmation=separate_confirmation)
    lock = Lock()
    revision = 0
    token = token_urlsafe(32)

    @app.get("/")
    def dashboard():
        with lock:
            latest = system.history[-1] if system.history else {}
            try:
                observation = _plain(system.executor.get_observation())
            except RuntimeError:
                observation = None
            review = next((review for item in reversed(system.history)
                           for stage in reversed(item["stage_results"])
                           for review in (stage.get("state_review"),
                                          stage.get("execution_result", {}).get("safety_decision"),
                                          stage.get("safety_decision"))
                           if review is not None), None)
            points = []
            for item in system.history[max(session.episode_history_start, len(system.history) - 30):]:
                for stage in item["stage_results"]:
                    result = stage.get("execution_result", {})
                    values = result.get("observation")
                    if result.get("success") is True and values:
                        index = 6 if environment == "AtropineEnvGym" else 17
                        if len(values) > index and math.isfinite(values[index]):
                            points.append((result["step_count"], values[index]))
            return render_template(
                "dashboard.html", environment=environment, state=system.state,
                step=system.executor.adapter.step_count, latest=latest.get("summary", {}),
                observation=observation, transitions=system.transitions[-10:],
                stages=latest.get("stage_results", []),
                active_workers=sorted(system.scheduler.active_agent_tasks),
                token=token, revision=revision, review=review,
                points=points, updated_at=session.updated_at, goal=session.goal,
                acknowledged=session.acknowledged == session.confirmation_key(),
                separate_confirmation=separate_confirmation,
                numeric_result=session.last_goal_result["summary"] if session.last_goal_result else None,
            )

    @app.post("/<operation>")
    def operate(operation):
        nonlocal revision
        if operation not in {"cycle", "pause", "confirm", "revise", "resume"}:
            abort(404)
        if not compare_digest(request.form.get("token", "").encode(), token.encode()):
            abort(403)
        with lock:
            if request.form.get("revision") != str(revision):
                abort(409, "This form has already been used or is out of date.")
            revision += 1
            operator, reason = request.form.get("operator"), request.form.get("reason")
            try:
                if operation == "cycle":
                    session.cycle()
                elif operation == "pause":
                    session.pause(operator, reason)
                elif operation == "confirm":
                    session.confirm(operator, reason)
                elif operation == "revise":
                    session.revise_goal(operator, reason, lower=request.form.get("lower", ""),
                                        upper=request.form.get("upper", ""))
                else:
                    session.resume(operator, reason, reset=request.form.get("reset") == "yes")
            except ValueError:
                abort(400, "Invalid operator, reason or target range.")
            except RuntimeError as exc:
                abort(409, str(exc))
        return redirect(url_for("dashboard"), code=303)

    @app.after_request
    def prevent_stale_forms(response):
        response.headers["Cache-Control"] = "no-store"
        return response

    return app


def main():
    parser = argparse.ArgumentParser(description="Local L3 simulation console")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--port", default=5000, type=int)
    parser.add_argument("--seed", default=1101, type=int)
    parser.add_argument("--environment", choices=("AtropineEnvGym", "MAbEnvGym"), default="AtropineEnvGym")
    parser.add_argument("--goal-spec", type=Path)
    parser.add_argument("--goal-registry", type=Path)
    args = parser.parse_args()
    if (args.goal_spec is None) != (args.goal_registry is None):
        parser.error("--goal-spec and --goal-registry must be supplied together")
    goal = json.loads(args.goal_spec.read_text(encoding="utf-8")) if args.goal_spec else None
    registry = json.loads(args.goal_registry.read_text(encoding="utf-8")) if args.goal_registry else None
    from l3_system.agents.monitoring_agent import MonitoringAgent
    from l3_system.execution import ExecutionAgent
    from l3_system.runtime import MultiAgentSystem
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with open_environment(args.environment) as adapter:
        monitor = MonitoringAgent(adapter.action_space, adapter.observation_space)
        executor = ExecutionAgent(adapter, log_path=args.output_dir / "B_events.jsonl")
        executor.reset(seed=args.seed)
        system = MultiAgentSystem(executor=executor, monitor=monitor,
                                  event_path=args.output_dir / "events.jsonl")
        app = create_app(system, environment=args.environment, goal=goal, registry=registry,
                         separate_confirmation=True)
        app.run(host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
