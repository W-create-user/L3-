"""Reproduce C's public acceptance evidence from the repository root."""

import argparse
import csv
from hashlib import sha256
import json
from math import isclose, isfinite
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SCHEDULING_REFERENCE = ROOT / "evidence/c_20261127/scheduling/metrics.csv"
LEARNING_REFERENCE = ROOT / "evidence/c_20261127/learning/learning.json"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def scheduling_checks(output):
    with (output / "metrics.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    with SCHEDULING_REFERENCE.open(newline="", encoding="utf-8") as stream:
        reference = list(csv.DictReader(stream))
    require(rows == reference, "Paired scheduling metrics differ from the archived results")
    require(len(rows) == 90, "Expected 90 scheduling runs")
    return {key: sum(int(row[key]) for row in rows)
            for key in ("jobs", "completed", "overlapping_awards", "old_feedback_rejected")}


def learning_checks(output):
    result = read_json(output / "learning.json")
    reference = read_json(LEARNING_REFERENCE)
    require(result["config"] == reference["config"], "Training configuration changed")
    selected_fields = ("seed", "actor_rate")
    require(all(result["selected"][key] == reference["selected"][key] for key in selected_fields),
            "Validation selected a different candidate")
    require(len(result["candidates"]) == 6 and all(
        row["updates"] > 0 and all(isfinite(delta) and delta > 0 for delta in row["actor_parameter_delta"])
        for row in result["candidates"]), "Expected six candidates with finite parameter updates")
    require(result["veto"]["rejected"] and result["veto_steps"] == 0, "Veto advanced the training fixture")
    for name, expected in reference["evaluation"].items():
        observed = result["evaluation"][name]
        require(len(observed) == len(expected) and all(
            left["seed"] == right["seed"] and isclose(left["return"], right["return"], rel_tol=0, abs_tol=1e-9)
            for left, right in zip(observed, expected)), f"Evaluation diverged in {name}")
    selected = result["selected"]
    rebuilt = sha256((output / selected["weight_file"]).read_bytes()).hexdigest()
    require(rebuilt == selected["weight_sha256"], "Rebuilt weight is inconsistent with its digest")
    return {"candidates": len(result["candidates"]), "evaluation_mean": result["evaluation_mean"],
            "rebuilt_weight_sha256": rebuilt,
            "archived_weight_sha256": reference["selected"]["weight_sha256"]}


def communication_checks(output):
    delayed = read_json(output / "atropine-delayed/summary.json")
    recovered = delayed["lost_execution_reply"]
    require(recovered["step_calls"] == 1 and recovered["takeover"]
            and recovered["premature_resume_blocked"] and recovered["out_of_order_receipt_rejected"]
            and recovered["reconciled_without_replay"], "Delayed result did not reconcile safely")
    require(delayed["normal"]["step_delta"] == 1 and delayed["recovered"]["step_delta"] == 1
            and delayed["rejected"]["step_delta"] == 0, "Unexpected Atropine step deltas")
    lost = read_json(output / "atropine-owner-exit/summary.json")["owner_exit"]
    require(lost["audited_step_calls"] == 1 and lost["resource_lock_retained"]
            and lost["resume_blocked"] and lost["new_dispatch_blocked"] and not lost["resolved"],
            "Execution owner exit did not keep the session blocked")
    fixture = read_json(output / "learning-gated/summary.json")
    require(fixture["normal"]["step_delta"] == 1 and fixture["recovered"]["step_delta"] == 1
            and fixture["rejected"]["step_delta"] == 0, "Synthetic policy failed its execution gate")
    digest = read_json(LEARNING_REFERENCE)["selected"]["weight_sha256"]
    require(fixture["weight_sha256"] == digest, "Runtime did not use the archived actor weight")
    return {"delayed_reply_reconciled": True, "owner_exit_left_locked": True,
            "learning_fixture_gate_passed": True}


def run(output):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    summary = {"revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                                  text=True).strip(),
               "python": platform.python_version(), "scope": "public C simulation",
               "checks": {}, "status": "failed"}
    env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
           "PYTHONUNBUFFERED": "1"}

    def execute(name, *arguments):
        logfile = output / f"{name}.log"
        with logfile.open("w", encoding="utf-8") as stream:
            process = subprocess.run([sys.executable, *arguments], cwd=ROOT, env=env,
                                     stdout=stream, stderr=subprocess.STDOUT, timeout=180)
        require(process.returncode == 0, f"{name} failed. Details in {logfile}")
        summary["checks"][name] = "passed"

    try:
        execute("main-tests", "-m", "unittest", "discover", "-s", "l3_system/tests", "-t", ".", "-v")
        execute("scheduling-tests", "-m", "unittest", "discover", "-s", "deliverables/C08/tests", "-v")
        execute("wire-tests", "-m", "unittest", "discover", "-s", "deliverables/C10/tests", "-v")
        execute("local-communication", "-m", "deliverables.C08.communication", "--mode", "local",
                "--out", str(output / "local.json"))
        execute("local-atropine", "-m", "experiments.run_validation",
                "--output", str(output / "local-atropine"))
        summary["local_atropine_cases"] = read_json(
            output / "local-atropine/summary.json")["validated_cases"]
        require(summary["local_atropine_cases"] == 10, "Expected ten local Atropine cases")
        execute("paired-scheduling", "-m", "deliverables.C08.run", "--out", str(output / "scheduling"))
        summary["scheduling"] = scheduling_checks(output / "scheduling")
        execute("train-candidates", "-m", "deliverables.C07.train", "--out", str(output / "learning"))
        summary["learning"] = learning_checks(output / "learning")
        for name, options in (("atropine-delayed", ("--lost-reply",)),
                              ("atropine-owner-exit", ("--owner-exit",)),
                              ("learning-gated", ("--backend", "learning-fixture"))):
            execute(name, "-m", "deliverables.C10.live", *options, "--out", str(output / name))
        summary["communication"] = communication_checks(output)
        summary["status"] = "passed"
    except (RuntimeError, subprocess.TimeoutExpired) as error:
        summary["error"] = str(error)
        raise
    finally:
        (output / "acceptance.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args().out)
