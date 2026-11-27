"""B adapter contract for the frozen two-unit C07 synthetic environment."""

from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from deliverables.C07.train import Fixture, load_actor

RECORD = Path(__file__).resolve().parents[2] / "evidence/c_20261127/learning"


class TrainingFixtureAdapter:
    source_dir = "C07 coupled dimensionless fixture"

    def __init__(self):
        self.state_version = self.step_count = self.step_calls = 0
        self.done = False
        self.fixture = None
        # Expose the exact local coordinates used by the trained actors.
        self.observation_space = SimpleNamespace(shape=(2,), low=np.full(2, -2.),
                                                 high=np.full(2, 2.), dtype=np.dtype(float))
        self.action_space = SimpleNamespace(shape=(2,), low=np.full(2, -1.),
                                            high=np.full(2, 1.), dtype=np.dtype(float))

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def reset(self, seed):
        self.fixture = Fixture(np.random.default_rng(seed))
        self.state_version += 1
        self.step_count = 0
        self.done = False
        return self.get_observation()

    def require_ready(self):
        if self.fixture is None or self.done:
            raise RuntimeError("Fixture needs an explicit reset")

    def get_observation(self):
        return self.fixture.state.tolist()

    def coerce_candidate(self, action):
        value = np.asarray(action, dtype=float)
        if value.shape != (2,) or not np.isfinite(value).all():
            raise ValueError("Expected two finite components")
        return value

    def step(self, action):
        self.require_ready()
        result = self.fixture.submit(self.coerce_candidate(action))
        if result["rejected"]:
            raise ValueError("Action outside fixture range")
        self.state_version += 1
        self.step_count += 1
        self.step_calls += 1
        self.done = result["terminated"] or result["truncated"]
        return {"success": True, "observation": self.get_observation(),
                "reward": float(result["reward"]), "action": result["executed"].tolist(),
                "done": self.done, "terminated": result["terminated"],
                "truncated": result["truncated"], "info": {}, "failure_reason": None,
                "step_calls": self.step_calls, "step_count": self.step_count,
                "state_version": self.state_version}


def frozen_policy():
    candidate = json.loads((RECORD / "learning.json").read_text())["selected"]
    weights = RECORD / candidate["weight_file"]
    if sha256(weights.read_bytes()).hexdigest() != candidate["weight_sha256"]:
        raise ValueError("Frozen actor digest mismatch")
    actors = [load_actor(weights, index) for index in range(2)]

    def propose(observation, _space, _attempt):
        return [actor(observation[index]) for index, actor in enumerate(actors)]

    return propose, candidate["weight_sha256"]
