"""Produce bids from current logical resource state."""
from dataclasses import dataclass
import math

from l3_system.algorithms.cnp_protocol import CNPError


@dataclass
class Worker:
    worker_id: str
    capabilities: frozenset[str]
    environments: frozenset[str]
    cost_factor: float
    heartbeat: float
    load: float = 0.0
    available: bool = True


class WorkerRegistry:
    """Load and cost are supplied by the resource model at each decision."""

    def __init__(self, heartbeat_timeout):
        if not math.isfinite(heartbeat_timeout) or heartbeat_timeout <= 0:
            raise CNPError("Heartbeat timeout must be positive and finite")
        self.heartbeat_timeout = heartbeat_timeout
        self.workers = {}

    def register(self, worker):
        if not worker.worker_id or worker.worker_id in self.workers:
            raise CNPError("Worker identity must be unique and nonempty")
        if not worker.capabilities or not worker.environments:
            raise CNPError("Worker capabilities and environments are required")
        if not math.isfinite(worker.cost_factor) or worker.cost_factor <= 0:
            raise CNPError("Worker cost factor must be positive and finite")
        self.update(worker, heartbeat=worker.heartbeat, load=worker.load,
                    available=worker.available)
        self.workers[worker.worker_id] = worker

    @staticmethod
    def update(worker, *, heartbeat, load, available=True):
        if not math.isfinite(heartbeat) or heartbeat < worker.heartbeat:
            raise CNPError("Invalid or stale worker heartbeat")
        if not math.isfinite(load) or not 0 <= load <= 1:
            raise CNPError("Worker load must be in [0, 1]")
        worker.heartbeat = heartbeat
        worker.load = load
        worker.available = available

    def bids(self, task, *, now, environment, base_cost, occupied):
        if not math.isfinite(now) or not math.isfinite(base_cost) or base_cost < 0:
            raise CNPError("Invalid bidding time or base cost")
        bids = []
        for worker in self.workers.values():
            if (not worker.available or worker.worker_id in occupied
                    or not 0 <= now - worker.heartbeat < self.heartbeat_timeout
                    or task["task_type"] not in worker.capabilities
                    or environment not in worker.environments):
                continue
            bid = {"task_id": task["task_id"], "agent_id": worker.worker_id,
                   "estimated_cost": base_cost * worker.cost_factor,
                   "capability_match": 1.0, "load": worker.load}
            if task.get("subtask_id") is not None:
                bid["subtask_id"] = task["subtask_id"]
            bids.append(bid)
        return bids
