"""A-side envelope for requesting bids through an injected C transport."""
from copy import deepcopy
from math import isfinite
from uuid import uuid4


class BidRoute:
    """Supply bids to the local orchestrator without owning workers or scheduling."""

    def __init__(self, exchange, *, environment, timeout_seconds=2):
        if not callable(exchange) or not isinstance(environment, str) or not environment.strip():
            raise ValueError("A bid route requires a transport exchange and environment")
        if type(timeout_seconds) not in (int, float) or not isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("A bid route requires a positive finite timeout")
        self.exchange = exchange
        self.environment = environment
        self.timeout_seconds = timeout_seconds

    def __call__(self, task):
        key = {"schema_version": "a-bid-route/1", "environment_id": self.environment,
               "correlation_id": str(uuid4()), "task_id": task["task_id"],
               "subtask_id": task["subtask_id"]}
        reply = self.exchange({**key, "task": deepcopy(task)}, timeout_seconds=self.timeout_seconds)
        if not isinstance(reply, dict) or any(reply.get(k) != v for k, v in key.items()):
            raise ValueError("Bid reply does not belong to this request")
        bids = reply.get("bids")
        if not isinstance(bids, list):
            raise ValueError("Bid reply must contain a list")
        seen = set()
        for bid in bids:
            if (not isinstance(bid, dict) or bid.get("task_id") != key["task_id"]
                    or bid.get("subtask_id") != key["subtask_id"]):
                raise ValueError("Bid does not belong to the announced subtask")
            worker = bid.get("agent_id")
            if not isinstance(worker, str) or not worker.strip() or worker in seen:
                raise ValueError("Bid reply contains a missing or duplicate worker")
            seen.add(worker)
        return deepcopy(bids)
