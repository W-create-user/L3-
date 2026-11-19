"""Loopback Flask console for a single local A simulation session."""
import argparse
from pathlib import Path
from secrets import compare_digest, token_urlsafe
from threading import Lock

from flask import Flask, abort, redirect, render_template, request, url_for

from l3_system.main import _plain
from l3_system.planning.goal_runtime import open_environment


def create_app(system, *, environment):
    app = Flask(__name__)
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
            return render_template(
                "dashboard.html", environment=environment, state=system.state,
                step=system.executor.adapter.step_count, latest=latest.get("summary", {}),
                observation=observation, transitions=system.transitions[-10:],
                stages=latest.get("stage_results", []),
                active_workers=sorted(system.scheduler.active_agent_tasks),
                token=token, revision=revision,
            )

    @app.post("/cycle")
    @app.post("/resume")
    def operate():
        nonlocal revision
        if not compare_digest(request.form.get("token", "").encode(), token.encode()):
            abort(403)
        with lock:
            if request.form.get("revision") != str(revision):
                abort(409, "This form has already been used or is out of date.")
            # Consume the form even on failure and never repeat a physical action.
            revision += 1
            try:
                if request.path == "/resume":
                    system.resume(operator=request.form.get("operator"), reason=request.form.get("reason"))
                else:
                    system.run_cycle()
            except ValueError:
                abort(400, "Supply a valid operator and reason.")
            except RuntimeError:
                abort(409, "The simulation session cannot perform this operation in its current state.")
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
    args = parser.parse_args()
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
        create_app(system, environment=args.environment).run(host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
