"""Two-actor MADDPG demonstration on a dimensionless coupled control fixture."""
import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path
import platform

import numpy as np

TRAIN = (7101, 7102, 7103)
VALIDATION = tuple(range(7201, 7207))
EVALUATION = tuple(range(7301, 7307))
STEPS = 20
EPISODES = 140
BATCH = 64
CAPACITY = 4096
GAMMA = 0.95
TAU = 0.02
HIDDEN = 24
ACTOR_RATES = (0.002, 0.005)
CRITIC_RATE = 0.01
NOISE = 0.12

DYNAMICS = np.array([[0.80, 0.14], [0.12, 0.82]])
ACTION = np.array([[0.32, 0.07], [0.05, 0.30]])


class Fixture:
    def __init__(self, rng):
        self.steps = 0
        self.state = rng.uniform(-0.9, 0.9, size=2)

    def submit(self, requested):
        action = np.asarray(requested)
        # The local fixture is the only owner of state transitions.
        if action.shape != (2,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1):
            return {"rejected": True, "requested": action.tolist(), "executed": None}
        self.state = DYNAMICS @ self.state + ACTION @ action
        self.steps += 1
        reward = -(self.state @ self.state + 0.06 * (action @ action))
        return {"rejected": False, "requested": action.tolist(), "executed": action.copy(),
                "state": self.state.copy(), "reward": reward,
                "terminated": False, "truncated": self.steps == STEPS}


class Network:
    def __init__(self, rng, inputs, *, bounded):
        self.bounded = bounded
        self.params = [rng.normal(0, .3, (inputs, HIDDEN)), np.zeros(HIDDEN),
                       rng.normal(0, .2, (HIDDEN, 1)), np.zeros(1)]

    def forward(self, inputs):
        w1, b1, w2, b2 = self.params
        hidden = np.tanh(inputs @ w1 + b1)
        raw = hidden @ w2 + b2
        return (np.tanh(raw) if self.bounded else raw), hidden

    def backward(self, inputs, upstream, rate=0):
        output, hidden = self.forward(inputs)
        grad = upstream * (1 - output ** 2) if self.bounded else upstream
        w1, _, w2, _ = self.params
        grad_hidden = (grad @ w2.T) * (1 - hidden ** 2)
        grad_inputs = grad_hidden @ w1.T
        if rate:
            updates = (inputs.T @ grad_hidden, grad_hidden.sum(axis=0),
                       hidden.T @ grad, grad.sum(axis=0))
            for param, update in zip(self.params, updates):
                param -= rate * update
        return grad_inputs

    def update_from(self, source, rate):
        for target, current in zip(self.params, source.params):
            target += rate * (current - target)


class MADDPG:
    def __init__(self, rng, actor_rate):
        self.actor_rate = actor_rate
        self.actors = [Network(rng, 1, bounded=True) for _ in range(2)]
        self.critics = [Network(rng, 4, bounded=False) for _ in range(2)]
        self.targets = [Network(rng, 1, bounded=True) for _ in range(2)]
        self.target_critics = [Network(rng, 4, bounded=False) for _ in range(2)]
        self.pairs = list(zip(self.targets + self.target_critics, self.actors + self.critics))
        for target, network in self.pairs:
            target.update_from(network, 1)

    @staticmethod
    def joint(actors, states):
        return np.column_stack([a.forward(states[:, i:i+1])[0][:, 0]
                                for i, a in enumerate(actors)])

    def act(self, local):
        return self.joint(self.actors, local.reshape(1, 2))[0]

    def train(self, rows):
        states, actions, rewards, successors, terminal = np.split(rows, [2, 4, 5, 7], axis=1)
        current = np.column_stack((states, actions))
        next_joint = np.column_stack((successors, self.joint(self.targets, successors)))
        policy = self.joint(self.actors, states)
        errors = []
        for i, (critic, target, actor) in enumerate(zip(self.critics, self.target_critics, self.actors)):
            predicted, _ = critic.forward(current)
            target_value, _ = target.forward(next_joint)
            error = predicted - (rewards + GAMMA * (1 - terminal) * target_value)
            errors.append(float(np.mean(error ** 2)))
            critic.backward(current, 2 * error / len(rows), CRITIC_RATE)

            critic_input = np.column_stack((states, policy))
            action_grad = critic.backward(critic_input, -np.ones_like(error) / len(rows))
            actor.backward(states[:, i:i+1], action_grad[:, 2+i:3+i], self.actor_rate)
        if not np.isfinite(errors).all():
            raise RuntimeError("Nonfinite critic loss")
        for target, network in self.pairs:
            if not all(np.isfinite(param).all() for param in network.params):
                raise RuntimeError("Nonfinite actor or critic weights")
            target.update_from(network, TAU)
        return errors


def evaluate(policy, seeds):
    rows = []
    for seed in seeds:
        fixture = Fixture(np.random.default_rng(seed))
        total = 0.0
        for _ in range(STEPS):
            record = fixture.submit(policy(fixture.state.copy()))
            if record["rejected"]:
                raise RuntimeError("Evaluation action refused")
            total += record["reward"]
        rows.append({"seed": seed, "return": float(total)})
    return rows


def short_train(seed, rate, trace):
    rng = np.random.default_rng(seed)
    learner = MADDPG(rng, rate)
    initial = [actor.params[0].copy() for actor in learner.actors]
    replay = np.empty((CAPACITY, 8))
    count = 0
    updates = 0
    logs = []
    rejected = []
    for episode in range(EPISODES):
        fixture = Fixture(rng)
        total = 0.
        losses = []
        for step in range(STEPS):
            state = fixture.state.copy()
            requested = np.clip(learner.act(state) + rng.normal(0, NOISE, 2), -1, 1)
            result = fixture.submit(requested)
            trace.append({"episode": episode + 1, "step": step + 1,
                          "state": state.tolist(), "requested": result["requested"],
                          "executed": None if result["rejected"] else result["executed"].tolist(),
                          "rejected": result["rejected"], "terminated": result.get("terminated"),
                          "truncated": result.get("truncated"), "reward": result.get("reward")})
            if result["rejected"]:
                rejected.append({"episode": episode, "requested": result["requested"]})
                break
            replay[count % CAPACITY] = np.r_[state, result["executed"], result["reward"],
                                             result["state"], result["terminated"]]
            count += 1
            total += result["reward"]
            if count >= BATCH:
                batch = rng.choice(min(count, CAPACITY), size=BATCH, replace=False)
                losses.append(learner.train(replay[batch]))
                updates += 1
        logs.append({"episode": episode + 1, "return": float(total),
                     "mean_critic_mse": np.mean(losses, axis=0).tolist() if losses else None,
                     "truncated": fixture.steps == STEPS,
                     "replay_entries_cumulative": count})
    changes = [float(np.linalg.norm(actor.params[0] - before))
               for actor, before in zip(learner.actors, initial)]
    assert all(change > 0 for change in changes)
    return learner, {"seed": seed, "actor_rate": rate, "updates": updates, "logs": logs,
                     "actor_parameter_delta": changes, "rejected": rejected}


def save(actors, path):
    weights = {f"actor{i}_{j}": array.tolist() for i, actor in enumerate(actors)
               for j, array in enumerate(actor.params)}
    path.write_text(json.dumps(weights, allow_nan=False, separators=(",", ":")) + "\n")
    return sha256(path.read_bytes()).hexdigest()


def load_actor(path, actor_index):
    """Inference API takes an actor's single local observation."""
    weights = json.loads(path.read_text())
    keys = [f"actor{actor_index}_{i}" for i in range(4)]
    w1, b1, w2, b2 = [np.asarray(weights[k], dtype=float) for k in keys]
    if [w1.shape, b1.shape, w2.shape, b2.shape] != [(1, HIDDEN), (HIDDEN,), (HIDDEN, 1), (1,)]:
        raise ValueError("Invalid actor parameter shapes")
    if not all(np.isfinite(x).all() for x in (w1, b1, w2, b2)):
        raise ValueError("Invalid weights")

    def act(local_observation):
        observation = float(local_observation)
        if not np.isfinite(observation):
            raise ValueError("Nonfinite local observation")
        return float(np.tanh(np.tanh(np.array([[observation]]) @ w1 + b1) @ w2 + b2)[0, 0])

    return act


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    candidates = []
    trace = []
    for rate in ACTOR_RATES:
        for seed in TRAIN:
            transitions = []
            learner, record = short_train(seed, rate, transitions)
            trace.extend({"seed": seed, "actor_rate": rate, **row} for row in transitions)
            path = output / f"actor_{seed}_{rate}.json"
            record.update(weight_file=path.name, weight_sha256=save(learner.actors, path),
                          validation=evaluate(learner.act, VALIDATION))
            candidates.append(record)
    average = lambda rows: float(np.mean([row["return"] for row in rows]))
    rates = {rate: average([row for candidate in candidates if candidate["actor_rate"] == rate
                            for row in candidate["validation"]]) for rate in ACTOR_RATES}
    best_rate = max(rates, key=rates.get)
    selected = max((c for c in candidates if c["actor_rate"] == best_rate),
                   key=lambda c: average(c["validation"]))
    loaded = [load_actor(output / selected["weight_file"], i) for i in range(2)]

    def loaded_policy(state):
        return np.array([actor(state[i]) for i, actor in enumerate(loaded)])

    loaded_rows = evaluate(loaded_policy, VALIDATION)
    assert np.allclose([r["return"] for r in loaded_rows],
                       [r["return"] for r in selected["validation"]], rtol=0, atol=1e-12)
    fixture = Fixture(np.random.default_rng(0))
    veto = fixture.submit([1.01, 0.0])
    assert veto["rejected"] and fixture.steps == 0
    evaluation = {
        "fixed_zero": evaluate(lambda state: np.zeros_like(state), EVALUATION),
        "single_shared_p": evaluate(lambda state: -0.75 * state, EVALUATION),
        "maddpg": evaluate(loaded_policy, EVALUATION),
    }
    trace_path = output / "transitions.jsonl.gz"
    payload = "".join(json.dumps(row, allow_nan=False) + "\n" for row in trace)
    trace_path.write_bytes(gzip.compress(payload.encode(), mtime=0))
    report = {"transition_trace_sha256": sha256(trace_path.read_bytes()).hexdigest(), "config": {"train_seeds": TRAIN, "validation_seeds": VALIDATION,
                         "evaluation_seeds": EVALUATION, "episodes_per_candidate": EPISODES,
                         "steps_per_episode": STEPS, "actor_rates": ACTOR_RATES,
                         "critic_rate": CRITIC_RATE, "gamma": GAMMA, "tau": TAU, "noise_std": NOISE,
                         "batch": BATCH, "replay_capacity": CAPACITY, "hidden_units": HIDDEN},
              "python": platform.python_version(), "numpy": np.__version__,
              "hardware": {"machine": platform.machine(), "backend": "NumPy CPU"},
              "candidates": candidates, "validation_mean_by_rate": rates,
              "selected": {k: selected[k] for k in ("seed", "actor_rate", "weight_file", "weight_sha256")},
              "reload_return_atol": 1e-12, "veto": veto, "veto_steps": fixture.steps,
              "evaluation": evaluation,
              "evaluation_mean": {key: average(rows) for key, rows in evaluation.items()}}
    (output / "learning.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    run(parser.parse_args().out)
