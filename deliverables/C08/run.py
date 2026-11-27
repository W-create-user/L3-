"""Paired, deterministic logical-resource experiment. Run with python -m."""
import argparse
import csv
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import platform
from pathlib import Path
from random import Random
from statistics import mean

from l3_system.algorithms.cnp_protocol import CNPError, CNPProtocol, FixedOrderProtocol
from l3_system.scheduling.leased_scheduler import LeasedScheduler
from l3_system.scheduling.worker_registry import Worker, WorkerRegistry

SEEDS = tuple(range(27101, 27111))
SCENARIOS = ("normal", "burst", "worker_exit")
POLICIES = ("fixed", "cnp", "cnp_no_load")
RESOURCES = {"r1": 1.0, "r2": 1.2, "r3": 0.8}
HORIZON = 81
LEASE_GRACE = 2


@dataclass(frozen=True)
class Work:
    job_id: str
    arrival: int
    service: int
    priority: int
    capability: str

    def task(self):
        return {"task_id": "resource_fixture", "subtask_id": self.job_id,
                "task_type": self.capability, "priority": self.priority}


def make_fixture(seed, scenario):
    rng = Random(seed)
    jobs = [Work(f"j{i:02}", 2 * i, rng.randint(2, 6), 1, "standard")
            for i in range(12)]
    if scenario == "burst":
        jobs += [Work(f"b{i:02}", 8, rng.randint(2, 5), 3,
                      "special" if i % 2 == 0 else "standard") for i in range(6)]
    return {"jobs": [asdict(job) for job in jobs],
            "exit": {"worker": "r1", "start": 1, "end": 11} if scenario == "worker_exit" else None,
            "background_load": {key: [rng.randint(0, 8) / 10 for _ in range(HORIZON)]
                                for key in RESOURCES}}


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def simulate(fixture, policy, emit, namespace):
    protocol = {"fixed": FixedOrderProtocol, "cnp": CNPProtocol,
                "cnp_no_load": lambda: CNPProtocol({"capability": .5, "cost": .3, "load": 0})}
    scheduler = LeasedScheduler(protocol[policy](), assignment_namespace=namespace)
    registry = WorkerRegistry(heartbeat_timeout=2)
    for key, factor in RESOURCES.items():
        capabilities = {"standard"} if key == "r1" else {"standard", "special"}
        registry.register(Worker(key, frozenset(capabilities), frozenset({"resource_fixture"}),
                                 factor, 0))
    jobs = {row["job_id"]: Work(**row) for row in fixture["jobs"]}
    active = {}
    starts = {}
    finishes = {}
    waiting = {}
    prior_receipts = {}
    fault = fixture["exit"]
    totals = dict(contention_attempts=0, resolved_conflicts=0, overlapping_awards=0,
                  timeouts=0, reassignments=0, awards=0, busy_ticks=0, available_ticks=0,
                  old_feedback_rejected=0, duplicates_idempotent=0)

    for tick in range(HORIZON):
        offline = {fault["worker"]} if fault and fault["start"] <= tick < fault["end"] else set()
        for key, worker in registry.workers.items():
            registry.update(worker, heartbeat=tick, load=fixture["background_load"][key][tick],
                            available=key not in offline)
        expired = {row["assignment_id"] for row in scheduler.expire(tick)}
        for key, (assignment, completion_tick, cancelled) in list(active.items()):
            winner = assignment["winner_id"]
            if winner in offline and not cancelled:
                cancelled = True
                active[key] = (assignment, completion_tick, True)
                emit(tick, "exit_before_commit", job=key, winner=winner,
                     assignment_id=assignment["assignment_id"])
            if cancelled:
                if assignment["assignment_id"] not in expired:
                    continue
                totals["timeouts"] += 1
                # This atomic fixture commits work only on completion_tick. The
                # simulated owner can prove a cancelled award never committed.
                receipt = {**assignment, "execution_status": "failed",
                           "outcome": "not_executed", "failure_reason": "fixture_cancel_ack"}
                scheduler.update_feedback(receipt, now=tick, reconciled=True)
                waiting[key] = jobs[key]
                prior_receipts[key] = receipt
                emit(tick, "timeout_reconciled", job=key, receipt=receipt)
            elif tick == completion_tick:
                receipt = {**assignment, "execution_status": "completed", "outcome": "executed"}
                first = scheduler.update_feedback(receipt, now=tick)
                assert scheduler.update_feedback(receipt, now=tick) == first
                totals["duplicates_idempotent"] += 1
                finishes[key] = tick
                emit(tick, "completed", job=key, receipt=receipt, duplicate_idempotent=True)
            else:
                continue
            del active[key]

        for job in jobs.values():
            if job.arrival == tick:
                waiting[job.job_id] = job
                emit(tick, "arrival", job=asdict(job))
        for job in sorted(waiting.values(), key=lambda j: (-j.priority, j.arrival, j.job_id)):
            key = job.job_id
            task = job.task()
            matching = registry.bids(task, now=tick, environment="resource_fixture",
                                     base_cost=job.service, occupied=())
            occupied = scheduler.active_agent_tasks
            busy = sorted(b["agent_id"] for b in matching if b["agent_id"] in occupied)
            bids = [b for b in matching if b["agent_id"] not in occupied]
            totals["contention_attempts"] += bool(busy)
            lease_duration = max(1, round(job.service * max(RESOURCES.values()))) + LEASE_GRACE
            if key in prior_receipts:
                result = scheduler.retry_failed_task(task["task_id"], key, bids, now=tick,
                                                     lease_duration=lease_duration)
            else:
                result = scheduler.schedule(task, bids, now=tick, lease_duration=lease_duration)
            emit(tick, "auction", job=key, bids=bids, busy_candidates=busy, award=result)
            if not result["winner_id"]:
                continue
            winner = result["winner_id"]
            if any(assignment["winner_id"] == winner for assignment, _, _ in active.values()):
                totals["overlapping_awards"] += 1
                raise AssertionError("overlapping awards")
            totals["awards"] += 1
            totals["resolved_conflicts"] += bool(busy)
            assignment = scheduler.assignment_for_execution(result, now=tick)
            if key in prior_receipts:
                totals["reassignments"] += 1
                try:
                    scheduler.update_feedback(prior_receipts.pop(key), now=tick)
                except CNPError:
                    totals["old_feedback_rejected"] += 1
                    emit(tick, "stale_receipt_rejected", job=key, new_assignment=assignment)
                else:
                    raise AssertionError("obsolete receipt was accepted")
                assert scheduler.get_active_assignment(task["task_id"], key) == assignment
            active[key] = (assignment, tick + max(1, round(job.service * RESOURCES[winner])), False)
            starts.setdefault(key, tick)
            del waiting[key]
        totals["busy_ticks"] += sum(not cancelled for _, _, cancelled in active.values())
        totals["available_ticks"] += len(RESOURCES) - len(offline)
        if len(finishes) == len(jobs):
            break

    return {**totals, "jobs": len(jobs), "completed": len(finishes),
            "unfinished": len(jobs) - len(finishes),
            "makespan_ticks": tick - min(j.arrival for j in jobs.values()),
            "mean_initial_wait_ticks": mean(starts.get(key, tick) - j.arrival for key, j in jobs.items()),
            "utilization": totals["busy_ticks"] / totals["available_ticks"],
            "timeout_rate": totals["timeouts"] / totals["awards"] if totals["awards"] else 0.0}


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    inputs = [{"scenario": scenario, "seed": seed, **make_fixture(seed, scenario)}
              for scenario in SCENARIOS for seed in SEEDS]
    write_json(output / "fixtures.json", inputs)
    metrics = []
    with (output / "events.jsonl").open("w", encoding="utf-8") as events:
        for fixture in inputs:
            fingerprint = sha256(json.dumps(fixture, sort_keys=True).encode()).hexdigest()
            for policy in POLICIES:
                identity = {"scenario": fixture["scenario"], "seed": fixture["seed"],
                            "policy": policy, "input_sha256": fingerprint}

                def emit(tick, event, **details):
                    events.write(json.dumps({**identity, "tick": tick, "event": event, **details},
                                            allow_nan=False) + "\n")

                namespace = f'{fixture["scenario"]}:{fixture["seed"]}:{policy}'
                metrics.append({**identity, **simulate(fixture, policy, emit, namespace)})
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=metrics[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(metrics)
    summary = {}
    for scenario in SCENARIOS:
        entries = [row for row in metrics if row["scenario"] == scenario]
        fixed = {row["seed"]: row["makespan_ticks"] for row in entries if row["policy"] == "fixed"}
        summary[scenario] = {}
        for policy in POLICIES:
            rows = [r for r in entries if r["policy"] == policy]
            differences = [r["makespan_ticks"] - fixed[r["seed"]] for r in rows]
            totals = ("jobs", "completed", "unfinished", "timeouts", "reassignments",
                      "overlapping_awards", "contention_attempts", "resolved_conflicts",
                      "old_feedback_rejected", "duplicates_idempotent")
            summary[scenario][policy] = {
                "episodes": len(rows),
                **{field: sum(r[field] for r in rows) for field in totals},
                "mean_makespan_ticks": mean(r["makespan_ticks"] for r in rows),
                "mean_initial_wait_ticks": mean(r["mean_initial_wait_ticks"] for r in rows),
                "mean_utilization": mean(r["utilization"] for r in rows),
                "mean_timeout_rate": mean(r["timeout_rate"] for r in rows),
                "paired_delta_vs_fixed_ticks": mean(differences),
                "paired_deltas_min_max": [min(differences), max(differences)],
                "paired_win_tie_loss": [sum(d < 0 for d in differences),
                                        sum(d == 0 for d in differences),
                                        sum(d > 0 for d in differences)],
            }
    write_json(output / "summary.json", {"runtime": platform.python_version(), "results": summary})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    run(parser.parse_args().out)
