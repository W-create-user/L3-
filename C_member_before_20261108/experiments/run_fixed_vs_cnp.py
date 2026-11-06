from __future__ import annotations
import csv, json
from pathlib import Path
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
from algorithms.cnp_protocol import Bid, CNPProtocol
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

# Deterministic synthetic benchmark. These are NOT B's live bids or SMPL results.
AGENTS = ["Exec-A", "Exec-B", "Exec-C"]
ROWS = []
protocol = CNPProtocol()

for i in range(1, 31):
    task_id = f"BENCH-{i:03d}"
    subtask_id = f"SUB-{i:03d}"
    # Rotating deterministic profiles; identical candidates are presented to both policies.
    bids = []
    for j, agent in enumerate(AGENTS):
        capability = round(0.62 + ((i * 7 + j * 11) % 34) / 100, 2)
        cost = float(1 + ((i * 3 + j * 5) % 9))
        load = round(((i * 13 + j * 23) % 80) / 100, 2)
        bids.append({
            "task_id": task_id,
            "subtask_id": subtask_id,
            "agent_id": agent,
            "estimated_cost": cost,
            "capability_match": capability,
            "load": load,
        })

    # Baseline: fixed first-fit policy (alphabetical first eligible agent).
    fixed_winner = sorted(bids, key=lambda x: x["agent_id"])[0]

    # CNP: weighted, explainable score from the production protocol.
    bid_objects = [Bid.from_mapping(x) for x in bids]
    ranked = protocol.evaluate(bid_objects)
    cnp_winner = next(x for x in bids if x["agent_id"] == ranked[0].bid.agent_id)

    for policy, winner, score in [
        ("fixed_first_fit", fixed_winner, None),
        ("cnp_weighted", cnp_winner, ranked[0].score),
    ]:
        ROWS.append({
            "trial": i,
            "task_id": task_id,
            "subtask_id": subtask_id,
            "policy": policy,
            "winner_id": winner["agent_id"],
            "capability_match": winner["capability_match"],
            "estimated_cost": winner["estimated_cost"],
            "load": winner["load"],
            "cnp_score": score,
            "candidate_count": len(bids),
            "data_origin": "synthetic_deterministic",
        })

csv_path = OUT / "fixed_vs_cnp_raw.csv"
with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=list(ROWS[0]))
    writer.writeheader()
    writer.writerows(ROWS)

summary = {}
for policy in ("fixed_first_fit", "cnp_weighted"):
    data = [r for r in ROWS if r["policy"] == policy]
    summary[policy] = {
        "trials": len(data),
        "assignment_successes": len(data),
        "assignment_success_rate": 1.0,
        "mean_capability_match": round(mean(r["capability_match"] for r in data), 6),
        "mean_estimated_cost": round(mean(r["estimated_cost"] for r in data), 6),
        "mean_load": round(mean(r["load"] for r in data), 6),
        "mean_cnp_score": (
            round(mean(r["cnp_score"] for r in data), 6)
            if policy == "cnp_weighted" else None
        ),
    }

summary["experiment_design"] = {
    "trials_per_policy": 30,
    "candidate_agents_per_trial": 3,
    "baseline": "alphabetical first-fit; ignores bid metrics",
    "cnp": "0.50 capability + 0.30/(1+estimated_cost) + 0.20*(1-load)",
    "weights_status": "manual prototype weights; not trained or optimized",
    "data_origin": "synthetic deterministic test data; not real ExecutionAgent bids and not SMPL measurements",
    "limitations": [
        "assignment success rate is trivially 100% because every trial supplies three valid bids",
        "no wall-clock bid timeout or asynchronous communication is measured",
        "no environment execution reward or task completion outcome is measured",
        "the experiment compares selection behavior only, not end-to-end system performance"
    ]
}
(OUT / "fixed_vs_cnp_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
)
print(json.dumps(summary, indent=2, ensure_ascii=False))
