"""Scheduling Agent with parent task/subtask identity preservation."""
from __future__ import annotations
from typing import Any, Iterable, Mapping
from copy import deepcopy
from l3_system.algorithms.cnp_protocol import CNPError, CNPProtocol


class SchedulingAgent:
    def __init__(self, cnp: CNPProtocol | None = None):
        self.cnp = cnp or CNPProtocol()
        self.tasks: dict[str, dict[str, Any]] = {}
        self.assignments: dict[str, dict[str, Any]] = {}
        self.feedback_log: list[dict[str, Any]] = []
        self.assignment_history: dict[str, list[dict[str, Any]]] = {}
        self.failed_agents: dict[str, set[str]] = {}
        self.active_agent_tasks: dict[str, str] = {}
        self.retry_log: list[dict[str, Any]] = []

    @staticmethod
    def _key(task: Mapping[str, Any]) -> str:
        return str(task.get("subtask_id") or task["task_id"])

    def register_task(self, task: Mapping[str, Any]) -> dict[str, Any]:
        record = self.cnp.announce(task)
        key = self._key(record)
        record["status"] = str(task.get("status", "pending"))
        if key in self.tasks:
            if self.tasks[key] != record:
                raise CNPError(f"Task key {key!r} already registered with different data")
            return dict(self.tasks[key])
        self.tasks[key] = dict(record)
        return dict(record)

    def register_tasks(self, tasks: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Receive A's HTN subtask list; require unique subtask identity."""
        task_list = list(tasks)
        seen_subtasks: set[str] = set()
        for task in task_list:
            if task.get("subtask_id") is None or not str(task.get("subtask_id")):
                raise CNPError(
                    "A HTN subtask must include a non-empty subtask_id"
                )
            subtask_id = str(task["subtask_id"])
            if subtask_id in seen_subtasks:
                raise CNPError(f"Duplicate subtask_id in input: {subtask_id!r}")
            seen_subtasks.add(subtask_id)
        return [self.register_task(task) for task in task_list]

    def _completed_ids(self):
        completed = {self._key(t) for t in self.tasks.values() if t["status"] == "completed"}
        parents = {t["task_id"] for t in self.tasks.values()}
        completed.update(parent for parent in parents
                         if all(t["status"] == "completed" for t in self.tasks.values()
                                if t["task_id"] == parent))
        return completed

    def ready_tasks(self) -> list[dict[str, Any]]:
        ready = []
        completed_ids = self._completed_ids()
        for task in self.tasks.values():
            if task["status"] != "pending":
                continue
            if all(dep in completed_ids for dep in task.get("dependencies", [])):
                ready.append(deepcopy(task))
        return sorted(ready, key=lambda t: (-int(t.get("priority", 0)), self._key(t)))

    def schedule(self, task: Mapping[str, Any],
                 bids: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        key = self._key(task)
        if key in self.tasks:
            # Re-announcement may reuse A's original pending task payload even
            # though C internally moved the task to "announced" after no bids.
            incoming = self.cnp.announce(task)
            existing = dict(self.tasks[key])
            existing.pop("status", None)
            incoming.pop("status", None)
            if existing != incoming:
                raise CNPError(
                    f"Task key {key!r} already registered with different task data"
                )
            record = self.tasks[key]
        else:
            registered = self.register_task(task)
            key = self._key(registered)
            record = self.tasks[key]
        if key in self.assignments:
            raise CNPError(f"Subtask {key!r} has already been awarded")
        if record["status"] not in {"pending", "announced"}:
            raise CNPError(f"Task {key!r} cannot be scheduled from status {record['status']!r}")

        completed_ids = self._completed_ids()
        for dep in record.get("dependencies", []):
            if dep not in completed_ids:
                return {
                    "task_id": record["task_id"],
                    **({"subtask_id": record["subtask_id"]} if record.get("subtask_id") else {}),
                    "winner_id": None, "score": None,
                    "assignment_status": "blocked",
                    "reason": f"dependency_not_completed:{dep}",
                }

        record["status"] = "announced"
        # An execution agent may hold only one active assignment at a time.
        raw_bids = list(bids)
        eligible_bids = []
        for raw in raw_bids:
            agent_id = str(raw.get("agent_id", ""))
            active_key = self.active_agent_tasks.get(agent_id)
            if active_key is not None and active_key != key:
                continue
            if agent_id in self.failed_agents.get(key, set()):
                continue
            eligible_bids.append(raw)
        collected = self.cnp.collect_bids(record, eligible_bids)
        ranked = self.cnp.evaluate(collected)
        assignment = self.cnp.award(record, ranked)
        if assignment["assignment_status"] == "awarded":
            record["status"] = "awarded"
            # Store only the contract fields as canonical assignment.
            self.assignments[key] = {
                k: v for k, v in assignment.items()
            }
            self.assignment_history.setdefault(key, []).append(dict(assignment))
            self.active_agent_tasks[assignment["winner_id"]] = key
        return {
            **assignment,
            "announcement": self.cnp.announce(record),
            "ranked_bids": [x.to_dict() for x in ranked],
        }

    def schedule_tasks(
        self,
        tasks: Iterable[Mapping[str, Any]],
        bids_by_subtask: Mapping[str, Iterable[Mapping[str, Any]]],
    ) -> list[dict[str, Any]]:
        """Register A's list, then schedule each currently-ready subtask."""
        self.register_tasks(tasks)
        results = []
        for task in self.ready_tasks():
            key = self._key(task)
            results.append(self.schedule(task, bids_by_subtask.get(key, [])))
        return results

    def assignment_for_execution(self, assignment: Mapping[str, Any]) -> dict[str, Any]:
        """Return the plain dict contract to pass to B's agreed adapter."""
        required = ("task_id", "winner_id", "score", "assignment_status")
        missing = [k for k in required if k not in assignment]
        if missing:
            raise CNPError(f"Assignment missing fields: {', '.join(missing)}")
        if assignment["assignment_status"] != "awarded" or not assignment["winner_id"]:
            raise CNPError("Only awarded assignments can be sent to ExecutionAgent")
        # Keep parent and subtask identifiers unchanged.
        return {
            k: assignment[k] for k in
            ("task_id", "subtask_id", "winner_id", "score", "assignment_status")
            if k in assignment
        }

    def update_feedback(self, feedback: Mapping[str, Any]) -> dict[str, Any]:
        task_id = str(feedback.get("task_id", ""))
        subtask_id = feedback.get("subtask_id")
        key = str(subtask_id) if subtask_id is not None else task_id
        if key not in self.tasks:
            raise CNPError(f"Unknown task/subtask {key!r}")
        if self.tasks[key]["task_id"] != task_id:
            raise CNPError("Feedback parent task_id mismatch")
        normalized = self.cnp.feedback(
            task_id, str(feedback.get("execution_status", "")),
            feedback.get("failure_reason"),
            str(subtask_id) if subtask_id is not None else None,
            feedback.get("exclude_failed_agent"),
        )
        previous = self.tasks[key]["status"]
        if previous in {"completed", "failed", "blocked"}:
            # A duplicate terminal message cannot release a worker's newer lock.
            if previous == normalized["execution_status"] and self.feedback_log:
                old = next((f for f in reversed(self.feedback_log)
                            if str(f.get("subtask_id") or f["task_id"]) == key), None)
                if old == normalized:
                    return deepcopy(old)
            raise CNPError("Conflicting feedback for a terminal task")
        self.tasks[key]["status"] = normalized["execution_status"]
        current_assignment = self.assignments.get(key)
        if normalized["execution_status"] in {"completed", "failed"}:
            if current_assignment:
                winner = str(current_assignment["winner_id"])
                # Terminal feedback releases the winner's active-task lock.
                if self.active_agent_tasks.get(winner) == key:
                    self.active_agent_tasks.pop(winner, None)
                if normalized["execution_status"] == "failed":
                    # Safety rejection is an action-level decision, not proof
                    # that the execution agent itself is unavailable.
                    exclude = normalized["exclude_failed_agent"]
                    if exclude:
                        self.failed_agents.setdefault(key, set()).add(winner)
        self.feedback_log.append(normalized)
        return normalized

    def retry_failed_task(
        self,
        task_id: str,
        subtask_id: str,
        bids: Iterable[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Re-open a failed subtask and award it to a different eligible bidder.

        The failed winner is excluded automatically. If no eligible bid remains,
        the task stays failed and the caller receives an unassigned result.
        """
        key = str(subtask_id)
        if key not in self.tasks:
            raise CNPError(f"Unknown task/subtask {key!r}")
        record = self.tasks[key]
        if record["task_id"] != str(task_id):
            raise CNPError("task_id does not match registered subtask")
        if record["status"] != "failed":
            raise CNPError("Only failed tasks can be retried")
        previous_winner = (
            self.assignments.get(key, {}).get("winner_id")
            if self.assignments.get(key) else None
        )
        # Preserve the failed assignment in history; release the active lock.
        self.assignments.pop(key, None)
        record["status"] = "pending"
        result = self.schedule(record, bids)
        retry_entry = {
            "task_id": str(task_id),
            "subtask_id": key,
            "previous_winner": previous_winner,
            "new_winner": result.get("winner_id"),
            "assignment_status": result.get("assignment_status"),
            "reason": "execution_failure_retry",
        }
        self.retry_log.append(retry_entry)
        if result.get("assignment_status") != "awarded":
            record["status"] = "failed"
        return result

    def get_task_status(self, task_id: str, subtask_id: str | None = None) -> str:
        key = str(subtask_id) if subtask_id is not None else str(task_id)
        if key not in self.tasks:
            raise CNPError(f"Unknown task/subtask {key!r}")
        if self.tasks[key]["task_id"] != str(task_id):
            raise CNPError("Parent task_id mismatch")
        return str(self.tasks[key]["status"])

    def get_assignment(self, task_id: str, subtask_id: str | None = None):
        """Return the latest recorded Assignment, including completed history."""
        key = str(subtask_id) if subtask_id is not None else str(task_id)
        if key in self.tasks and self.tasks[key]["task_id"] != str(task_id):
            raise CNPError("Parent task_id mismatch")
        value = self.assignments.get(key)
        return deepcopy(value) if value else None

    def get_active_assignment(
        self, task_id: str, subtask_id: str | None = None
    ) -> dict[str, Any] | None:
        """Return the current Assignment only while its winner lock is active."""
        key = str(subtask_id) if subtask_id is not None else str(task_id)
        if key in self.tasks and self.tasks[key]["task_id"] != str(task_id):
            raise CNPError("Parent task_id mismatch")
        value = self.assignments.get(key)
        if not value:
            return None
        winner = str(value["winner_id"])
        if self.active_agent_tasks.get(winner) != key:
            return None
        return deepcopy(value)

    def get_assignment_history(
        self, task_id: str, subtask_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Return all awarded Assignments for a subtask, oldest first."""
        key = str(subtask_id) if subtask_id is not None else str(task_id)
        if key in self.tasks and self.tasks[key]["task_id"] != str(task_id):
            raise CNPError("Parent task_id mismatch")
        return deepcopy(self.assignment_history.get(key, []))
