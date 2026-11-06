"""Simplified CNP message models and deterministic bid evaluation."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Iterable, Mapping


class CNPError(ValueError):
    pass


@dataclass(frozen=True)
class Bid:
    task_id: str
    agent_id: str
    estimated_cost: float
    capability_match: float
    load: float
    subtask_id: str | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "Bid":
        required = ("task_id", "agent_id", "estimated_cost",
                    "capability_match", "load")
        missing = [key for key in required if key not in data]
        if missing:
            raise CNPError(f"Bid missing fields: {', '.join(missing)}")
        try:
            bid = cls(
                task_id=str(data["task_id"]),
                agent_id=str(data["agent_id"]),
                estimated_cost=float(data["estimated_cost"]),
                capability_match=float(data["capability_match"]),
                load=float(data["load"]),
                subtask_id=(str(data["subtask_id"])
                            if data.get("subtask_id") is not None else None),
            )
        except (TypeError, ValueError) as exc:
            raise CNPError(f"Invalid bid values: {exc}") from exc
        if not bid.task_id or not bid.agent_id:
            raise CNPError("task_id and agent_id must be non-empty")
        if bid.subtask_id == "":
            raise CNPError("subtask_id must be non-empty when provided")
        if bid.estimated_cost < 0:
            raise CNPError("estimated_cost must be >= 0")
        if not 0 <= bid.capability_match <= 1:
            raise CNPError("capability_match must be within [0, 1]")
        if not 0 <= bid.load <= 1:
            raise CNPError("load must be within [0, 1]")
        return bid

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScoredBid:
    bid: Bid
    score: float

    def to_dict(self) -> dict[str, Any]:
        result = self.bid.to_dict()
        result["score"] = round(self.score, 6)
        return result


class CNPProtocol:
    """Announcement -> bid collection -> evaluation -> award -> feedback."""

    def __init__(self, weights: Mapping[str, float] | None = None):
        self.weights = dict(weights or {
            "capability": 0.50, "cost": 0.30, "load": 0.20
        })
        if set(self.weights) != {"capability", "cost", "load"}:
            raise CNPError("weights must contain capability, cost and load")
        if any(float(v) < 0 for v in self.weights.values()):
            raise CNPError("weights must be non-negative")
        total = sum(float(v) for v in self.weights.values())
        if total <= 0:
            raise CNPError("at least one weight must be positive")
        self.weights = {k: float(v) / total for k, v in self.weights.items()}

    @staticmethod
    def task_key(task: Mapping[str, Any]) -> str:
        return str(task.get("subtask_id") or task["task_id"])

    @classmethod
    def announce(cls, task: Mapping[str, Any]) -> dict[str, Any]:
        if "task_id" not in task or "task_type" not in task:
            raise CNPError("Task requires task_id and task_type")
        if not str(task["task_id"]) or not str(task["task_type"]):
            raise CNPError("task_id and task_type must be non-empty")
        result = {
            "task_id": str(task["task_id"]),
            "task_type": str(task["task_type"]),
            "priority": int(task.get("priority", 0)),
            "dependencies": list(task.get("dependencies", [])),
            "constraints": dict(task.get("constraints", {})),
        }
        if task.get("subtask_id") is not None:
            result["subtask_id"] = str(task["subtask_id"])
        if task.get("status") is not None:
            result["status"] = str(task["status"])
        return result

    @staticmethod
    def collect_bids(task: Mapping[str, Any] | str,
                     bids: Iterable[Mapping[str, Any]]) -> list[Bid]:
        task_data = {"task_id": task} if isinstance(task, str) else task
        collected, seen = [], set()
        for raw in bids:
            bid = Bid.from_mapping(raw)
            if bid.task_id != str(task_data["task_id"]):
                raise CNPError("Bid task_id does not match announced task")
            expected_subtask = task_data.get("subtask_id")
            if bid.subtask_id != (str(expected_subtask) if expected_subtask is not None else None):
                raise CNPError("Bid subtask_id does not match announced task")
            if bid.agent_id in seen:
                raise CNPError(f"Duplicate bid from agent {bid.agent_id!r}")
            seen.add(bid.agent_id)
            collected.append(bid)
        return collected

    def score_bid(self, bid: Bid) -> float:
        # Lower estimated cost and lower utilization are preferred.
        cost_suitability = 1.0 / (1.0 + bid.estimated_cost)
        load_suitability = 1.0 - bid.load
        score = (
            self.weights["capability"] * bid.capability_match
            + self.weights["cost"] * cost_suitability
            + self.weights["load"] * load_suitability
        )
        return round(score, 6)

    def evaluate(self, bids: Iterable[Bid]) -> list[ScoredBid]:
        ranked = [ScoredBid(b, self.score_bid(b)) for b in bids]
        return sorted(ranked, key=lambda x: (-x.score, x.bid.agent_id))

    @staticmethod
    def award(task: Mapping[str, Any] | str,
              ranked: Iterable[ScoredBid]) -> dict[str, Any]:
        task_data = {"task_id": task} if isinstance(task, str) else task
        result = {
            "task_id": str(task_data["task_id"]),
            "winner_id": None,
            "score": None,
            "assignment_status": "unassigned",
        }
        if task_data.get("subtask_id") is not None:
            result["subtask_id"] = str(task_data["subtask_id"])
        ranked = list(ranked)
        if not ranked:
            result["reason"] = "no_valid_bids"
            return result
        result.update({
            "winner_id": ranked[0].bid.agent_id,
            "score": ranked[0].score,
            "assignment_status": "awarded",
        })
        return result

    @staticmethod
    def feedback(task_id: str, execution_status: str,
                 failure_reason: str | None = None,
                 subtask_id: str | None = None,
                 exclude_failed_agent: bool | None = None) -> dict[str, Any]:
        if execution_status not in {"running", "completed", "failed"}:
            raise CNPError("execution_status must be running, completed or failed")
        if execution_status == "failed" and not failure_reason:
            raise CNPError("failure_reason is required when execution failed")
        if exclude_failed_agent is not None and not isinstance(exclude_failed_agent, bool):
            raise CNPError("exclude_failed_agent must be bool when provided")
        # Default policy: do not penalize an agent for a safety-rule rejection;
        # exclude it for ordinary execution failures unless caller overrides.
        if exclude_failed_agent is None:
            exclude_failed_agent = (
                execution_status == "failed"
                and failure_reason != "safety_rejected"
            )
        result = {"task_id": str(task_id),
                  "execution_status": execution_status,
                  "failure_reason": failure_reason,
                  "exclude_failed_agent": exclude_failed_agent}
        if subtask_id is not None:
            result["subtask_id"] = str(subtask_id)
        return result
