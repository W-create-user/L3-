# Member C — deliverable snapshot for 2026-11-06 to 2026-11-08

## Implemented C responsibilities
- HTN task registration keyed by `subtask_id`, preserving parent `task_id`.
- Dependency-aware `ready_tasks()` and priority ordering.
- Simplified CNP announcement, Bid validation, weighted scoring, and Assignment.
- Feedback states `running`, `completed`, `failed`.
- Failed-task retry with optional `exclude_failed_agent`.
- Safety rejection defaults to not excluding the execution agent.
- Assignment history and active-assignment query.
- Empty Bid result is `unassigned` and leaves task `announced`; caller can call `schedule()` again with a fresh bid set.
- Active-agent lock to prevent simultaneous active awards to one agent.

## Run unit tests
From this directory:
```bash
python -m unittest discover -s tests -v
```

## Run fixed-policy vs CNP selection experiment
```bash
python experiments/run_fixed_vs_cnp.py
```
Outputs:
- `experiments/results/fixed_vs_cnp_raw.csv`
- `experiments/results/fixed_vs_cnp_summary.json`

**The comparison uses deterministic synthetic bids. It is a reproducible algorithm-level simulation, not real B-agent bid data, not SMPL measurements, and not end-to-end performance evidence.** Because every trial supplies three bids, assignment success rate is 100% by construction and should not be interpreted as a performance advantage.

## Known limitations / integration dependencies
- No wall-clock bid timeout or asynchronous bid collection; A/main.py must detect timeout and request another bid round.
- No live B ExecutionAgent or D MonitoringAgent is bundled in this C package.
- No live SMPL environment integration is claimed.
- C does not generate control actions or make safety decisions.
- A/ReplanningController chooses replan/retry/terminate policy; C performs the requested scheduling operation.
- Fixed-vs-CNP experiment evaluates candidate selection only, not execution success, rewards, or throughput.
