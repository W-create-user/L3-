"""Run public A01–A10 release checks and retain commands, logs and checksums."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def cases(output, profile):
    yield "contracts", ("-m", "unittest", "discover", "-s", "l3_system/tests", "-t", ".")
    yield "docs", ("-m", "experiments.check_docs")
    yield "atropine", ("main.py", "--cycles", "3", "--output", str(output / "atropine"))
    yield "atropine-reject", ("main.py", "--scenario", "reject", "--output", str(output / "atropine-reject"))
    yield "atropine-suite", ("-m", "experiments.run_validation", "--output", str(output / "atropine-suite"))
    yield "console", ("-m", "experiments.verify_a_console", "--output", str(output / "console"))
    if profile == "local":
        return
    yield "c08", ("-m", "unittest", "discover", "-s", "deliverables/C08/tests")
    yield "c10", ("-m", "unittest", "discover", "-s", "deliverables/C10/tests")
    yield "zmq", ("main.py", "--transport", "zmq", "--output", str(output / "zmq"))
    for fault in ("lost-reply", "owner-exit"):
        yield fault, ("-m", "deliverables.C10.live", f"--{fault}", "--out", str(output / fault))
    yield "mab-goal", (
        "main.py", "--environment", "MAbEnvGym", "--seed", "26001", "--max-steps", "10",
        "--goal-timeout", "90", "--goal-spec", "configs/goals/mab_public_buffer_goal.json",
        "--goal-registry", "configs/goals/mab_public_buffer_registry.json",
        "--output", str(output / "mab-goal"),
    )
    yield "mab-receipts", ("-m", "experiments.verify_b_freeze_receipts", "--output", str(output / "mab-receipts"))


def run(output, profile):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    environment = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "PYTHONUTF8": "1"}
    results = []
    for name, args in cases(output, profile):
        log = output / f"{name}.log"
        started = time.monotonic()
        with log.open("wb") as stream:
            try:
                result = subprocess.run(
                    [sys.executable, *args], cwd=ROOT, env=environment,
                    stdout=stream, stderr=subprocess.STDOUT, timeout=180, check=False,
                )
                exit_code = result.returncode
            except subprocess.TimeoutExpired:
                exit_code = 124
                stream.write(b"\nRelease check reached its 180-second limit.\n")
        entry = {"case": name, "args": list(args), "exit_code": exit_code,
                 "wall_seconds": round(time.monotonic() - started, 3), "log": log.name}
        results.append(entry)
        print(f"{name}: {'PASS' if exit_code == 0 else 'FAIL'} ({entry['wall_seconds']} s)", flush=True)
    success = all(entry["exit_code"] == 0 for entry in results)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    (output / "report.json").write_text(
        json.dumps({"commit": commit, "python": sys.version.split()[0], "profile": profile,
                    "public_simulation_only": True, "blind_holdout_executed": False,
                    "passed": success, "cases": results}, indent=2) + "\n",
        encoding="utf-8",
    )
    checksums = [f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(output).as_posix()}"
                 for path in sorted(output.rglob("*")) if path.is_file()]
    (output / "SHA256SUMS").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    return success


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--profile", choices=("local", "full"), default="full")
    args = parser.parse_args()
    if not run(args.output, args.profile):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
