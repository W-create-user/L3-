# C Member Freeze & Acceptance Checklist — 2026-11-08

## Freeze
- [ ] Record repository commit hash and Python/package versions.
- [ ] Freeze C module and record SHA-256 hashes.
- [ ] Run `python -m unittest discover -s tests -v`; archive stdout/stderr.
- [ ] Run `python experiments/run_fixed_vs_cnp.py`; archive raw CSV and JSON.
- [ ] Do not change score weights without updating report and rerunning experiments.

## Integration with A/B/D
- [ ] A supplies actual HTN `decision_result['subtasks']`.
- [ ] C registers every subtask with unique `subtask_id`.
- [ ] B supplies real Bid records; validate task and subtask IDs.
- [ ] C emits Assignment; B confirms actual receipt.
- [ ] B returns ExecutionResult; C updates feedback.
- [ ] D rejection occurs before `env.step`; verify no environment step on REJECT.
- [ ] Failed execution and safety rejection are recorded separately.
- [ ] Save full main.py log with timestamps, IDs, status transitions, and errors.

## Evidence hygiene
- [ ] Label synthetic benchmark data as synthetic.
- [ ] Label actual B/SMPL run data as real only when source logs exist.
- [ ] Report numerator, denominator, sample count, initial conditions, and version.
- [ ] Remove unsupported claims from slides/report.
