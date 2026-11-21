"""Four-glass MDP, tabular Q-learning and SARSA. No negative rewards.

Only NumPy is required. Valid actions are constrained by physics, never by a
hand-written solution. BFS is a separate evaluation oracle, not a teacher.
"""
from collections import deque
from dataclasses import dataclass
from itertools import product
from time import perf_counter
import numpy as np


@dataclass(frozen=True)
class Config:
    capacity: int = 6
    total: int = 12
    gamma: float = 0.9
    alpha: float = 0.5
    epsilon_start: float = 0.6
    epsilon_end: float = 0.05
    episodes: int = 200000
    max_steps: int = 30
    eval_every: int = 5000

    def __post_init__(self):
        if not isinstance(self.capacity, int) or self.capacity < 1:
            raise ValueError("capacity must be a positive integer")
        if not isinstance(self.total, int) or not 0 < self.total < 4*self.capacity:
            raise ValueError("total must be an integer between 0 and 4*capacity, exclusive")
        if self.total % 4:
            raise ValueError("total must be divisible by four for integer-volume equality")
        if not 0 < self.gamma < 1 or not 0 < self.alpha <= 1:
            raise ValueError("require 0 < gamma < 1 and 0 < alpha <= 1")
        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ValueError("invalid epsilon schedule")
        if min(self.episodes, self.max_steps, self.eval_every) < 1:
            raise ValueError("episode and step counts must be positive")


class GlassEnv:
    """A step is one complete lift, including transfers to several recipients.

    State: (v0,v1,v2,v3), integer volumes with fixed total.
    Action: source i and three amounts, recipients in increasing index order.
    Global IDs: i*((C+1)**3-1) + encoded_amounts-1.
    A zero transfer is not a lift and therefore has no action ID.
    """
    def __init__(self, config=None):
        self.config = config or Config()
        c = self.config.capacity
        self.target = self.config.total // 4
        self.goal = (self.target,)*4
        self.states = tuple(s for s in product(range(c+1), repeat=4)
                            if sum(s) == self.config.total)
        self.state_index = {s: i for i, s in enumerate(self.states)}
        self.goal_id = self.state_index[self.goal]
        self.starts = np.array([i for i in range(len(self.states)) if i != self.goal_id])
        self.n_actions = 4*((c+1)**3-1)
        self.actions, self.next_states = [], []
        for state in self.states:
            ids, successors = [], []
            if state != self.goal:
                for source in range(4):
                    recipients = [j for j in range(4) if j != source]
                    ranges = [range(min(c-state[j], state[source])+1) for j in recipients]
                    for amounts in product(*ranges):
                        moved = sum(amounts)
                        if not 0 < moved <= state[source]:
                            continue
                        nxt = list(state)
                        nxt[source] -= moved
                        for j, amount in zip(recipients, amounts):
                            nxt[j] += amount
                        ids.append(self.encode_action(source, amounts))
                        successors.append(self.state_index[tuple(nxt)])
            self.actions.append(np.asarray(ids, dtype=np.int32))
            self.next_states.append(np.asarray(successors, dtype=np.int32))
        self.state = self.goal
        self.steps = 0

    def encode_action(self, source, amounts):
        b = self.config.capacity+1
        if source not in range(4) or len(amounts) != 3:
            raise ValueError("expected source 0..3 and three transfer amounts")
        if any(not isinstance(x, (int, np.integer)) or not 0 <= x < b for x in amounts):
            raise ValueError("amounts must be integers 0..capacity")
        encoded = amounts[0]*b*b + amounts[1]*b + amounts[2]
        if encoded == 0:
            raise ValueError("empty transfer is not an action")
        return int(source*(b**3-1)+encoded-1)

    def decode_action(self, action):
        if not isinstance(action, (int, np.integer)) or not 0 <= action < self.n_actions:
            raise ValueError("action ID is out of range")
        b = self.config.capacity+1
        source, encoded = divmod(int(action), b**3-1)
        encoded += 1
        first, rest = divmod(encoded, b*b)
        second, third = divmod(rest, b)
        return source, (first, second, third)

    def reset(self, state=None, seed=None):
        rng = np.random.default_rng(seed)
        self.state = self.states[int(rng.choice(self.starts))] if state is None else tuple(state)
        if self.state not in self.state_index:
            raise ValueError("invalid volumes, capacity, or total")
        self.steps = 0
        return self.state, {"already_solved": self.state == self.goal}

    def step(self, action):
        sid = self.state_index[self.state]
        if sid == self.goal_id:
            raise RuntimeError("episode has ended; call reset")
        if self.steps >= self.config.max_steps:
            raise RuntimeError("time limit reached; call reset")
        matches = np.flatnonzero(self.actions[sid] == action)
        if not len(matches):
            raise ValueError("physically impossible action; use the legal action mask")
        nxt = int(self.next_states[sid][matches[0]])
        self.state = self.states[nxt]
        self.steps += 1
        terminated = nxt == self.goal_id
        truncated = self.steps >= self.config.max_steps and not terminated
        return self.state, float(terminated), terminated, truncated, {"lifts": self.steps}

    def describe(self, action):
        source, amounts = self.decode_action(action)
        recipients = [j for j in range(4) if j != source]
        transfers = ", ".join(f"в стакан {j+1}: {v}" for j,v in zip(recipients, amounts) if v)
        return f"Поднять стакан {source+1}; {transfers}"


def shortest_distances(env):
    """Exact reverse BFS on every physically legal edge. Evaluation only."""
    reverse = [[] for _ in env.states]
    for sid, successors in enumerate(env.next_states):
        for nxt in successors:
            reverse[int(nxt)].append(sid)
    distances = np.full(len(env.states), -1, dtype=int)
    distances[env.goal_id] = 0
    queue = deque([env.goal_id])
    while queue:
        nxt = queue.popleft()
        for sid in reverse[nxt]:
            if distances[sid] == -1:
                distances[sid] = distances[nxt]+1
                queue.append(sid)
    return distances


def new_q(env):
    # Ragged storage: only legal state-action pairs are represented.
    return [np.zeros(len(a), dtype=np.float64) for a in env.actions]


def epsilon_greedy(values, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(len(values)))
    best = np.flatnonzero(values == values.max())
    return int(best[int(rng.integers(len(best)))])


def rollout(env, q, state, max_steps=None):
    sid = env.state_index[tuple(state)]
    limit = max_steps or env.config.max_steps
    trajectory = []
    for step in range(limit):
        if sid == env.goal_id:
            break
        # Fixed first-index tie rule gives reproducible evaluation with epsilon=0.
        local_action = int(np.argmax(q[sid]))
        nxt = int(env.next_states[sid][local_action])
        action = int(env.actions[sid][local_action])
        trajectory.append({"before": env.states[sid], "action": action,
                           "after": env.states[nxt], "reward": float(nxt == env.goal_id)})
        sid = nxt
    return trajectory, sid == env.goal_id


def evaluate(env, q, distances=None):
    distances = shortest_distances(env) if distances is None else distances
    success, optimal, steps, returns = [], [], [], []
    for sid in env.starts:
        path, solved = rollout(env, q, env.states[int(sid)])
        count = len(path)
        success.append(solved)
        optimal.append(solved and count == distances[sid])
        steps.append(count)  # Includes capped failures; never hide them.
        returns.append(env.config.gamma**(count-1) if solved else 0.0)
    return {"success_rate": float(np.mean(success)), "optimal_rate": float(np.mean(optimal)),
            "mean_lifts_capped": float(np.mean(steps)), "discounted_return": float(np.mean(returns))}


def train(env, method="q_learning", seed=0, progress=None):
    if method not in ("q_learning", "sarsa"):
        raise ValueError("method must be q_learning or sarsa")
    cfg = env.config
    q = new_q(env)
    # Equal starting states for paired seeds; exploration gets its own RNG.
    starts = np.random.default_rng(seed).choice(env.starts, size=cfg.episodes)
    rng = np.random.default_rng(seed+100_000)
    distances = shortest_distances(env)
    history = {"return": [], "raw_reward": [], "lifts": [], "epsilon": [], "checkpoints": []}
    history["checkpoints"].append({"episode": 0, **evaluate(env, q, distances)})
    started = perf_counter()
    for episode, start in enumerate(starts):
        fraction = min(episode/max(1, int(cfg.episodes*0.8)-1), 1.0)
        epsilon = cfg.epsilon_start + fraction*(cfg.epsilon_end-cfg.epsilon_start)
        sid = int(start)
        action = epsilon_greedy(q[sid], epsilon, rng)
        solved = False
        for step in range(cfg.max_steps):
            nxt = int(env.next_states[sid][action])
            solved = nxt == env.goal_id
            reward = float(solved)
            if solved:
                target = reward
                next_action = None
            elif method == "q_learning":
                target = reward+cfg.gamma*float(q[nxt].max())
                next_action = None
            else:
                next_action = epsilon_greedy(q[nxt], epsilon, rng)
                target = reward+cfg.gamma*q[nxt][next_action]
            q[sid][action] += cfg.alpha*(target-q[sid][action])
            if solved:
                break
            sid = nxt
            action = (epsilon_greedy(q[sid], epsilon, rng)
                      if next_action is None else next_action)
            # At time-limit truncation bootstrap is retained: this is not a goal.
        lifts = step+1
        history["return"].append(cfg.gamma**(lifts-1) if solved else 0.0)
        history["raw_reward"].append(float(solved))
        history["lifts"].append(lifts)
        history["epsilon"].append(epsilon)
        if (episode+1) % cfg.eval_every == 0 or episode+1 == cfg.episodes:
            checkpoint = {"episode": episode+1, **evaluate(env, q, distances)}
            history["checkpoints"].append(checkpoint)
            if progress:
                progress(method, seed, checkpoint)
    history["seconds"] = perf_counter()-started
    return q, history


def random_baseline(env, seed=999, repeats=10):
    rng = np.random.default_rng(seed)
    rows = []
    distances = shortest_distances(env)
    for _ in range(repeats):
        for start in env.starts:
            sid = int(start)
            for step in range(env.config.max_steps):
                sid = int(rng.choice(env.next_states[sid]))
                if sid == env.goal_id:
                    break
            solved = sid == env.goal_id
            rows.append((solved, solved and step+1 == distances[start], step+1,
                         env.config.gamma**step if solved else 0.0))
    values = np.mean(rows, axis=0)
    return dict(zip(("success_rate", "optimal_rate", "mean_lifts_capped", "discounted_return"),
                    map(float, values)))
