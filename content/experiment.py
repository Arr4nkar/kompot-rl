"""Reproducible experiments and presentation of measured results."""
from dataclasses import asdict
from pathlib import Path
import csv
import json
import numpy as np
from kompot_rl import Config, GlassEnv, train, shortest_distances, random_baseline, rollout

NAMES = {"q_learning": "Q-learning", "sarsa": "SARSA"}
COLORS = {"q_learning": "#3465d9", "sarsa": "#b94477"}


def run_experiment(config=None, seeds=(0, 1, 2, 3, 4), verbose=True):
    cfg = config or Config()
    env = GlassEnv(cfg)
    distances = shortest_distances(env)
    oracle = {"success_rate": 1.0, "optimal_rate": 1.0,
              "mean_lifts_capped": float(distances[env.starts].mean()),
              "discounted_return": float(np.mean(cfg.gamma**(distances[env.starts]-1)))}
    result = {"config": asdict(cfg), "seeds": list(seeds), "states": len(env.states),
              "legal_pairs": sum(map(len, env.actions)), "oracle": oracle,
              "random": random_baseline(env), "runs": {}}
    policies = {}
    for method in NAMES:
        result["runs"][method] = []
        for seed in seeds:
            q, hist = train(env, method, seed=seed)
            # Store block averages, not a fabricated interpolation of checkpoints.
            block = max(1, cfg.episodes//200)
            curve = []
            for start in range(0, cfg.episodes, block):
                stop = min(start+block, cfg.episodes)
                curve.append({"episode": stop,
                              "discounted_return": float(np.mean(hist["return"][start:stop])),
                              "raw_reward": float(np.mean(hist["raw_reward"][start:stop])),
                              "lifts": float(np.mean(hist["lifts"][start:stop]))})
            checkpoints = hist["checkpoints"]
            auc = float(np.trapezoid([r["discounted_return"] for r in checkpoints],
                                    [r["episode"] for r in checkpoints])/cfg.episodes)
            result["runs"][method].append({"seed": seed, "seconds": hist["seconds"],
                                           "auc_return": auc, "training_curve": curve,
                                           "checkpoints": checkpoints})
            if seed == seeds[0]:
                policies[method] = [int(env.actions[s][int(np.argmax(q[s]))])
                                    if s != env.goal_id else None for s in range(len(env.states))]
            if verbose:
                final = checkpoints[-1]
                print(f"{NAMES[method]}, seed={seed}: success={final['success_rate']:.1%}, "
                      f"optimal={final['optimal_rate']:.1%}, lifts={final['mean_lifts_capped']:.3f}",
                      flush=True)
    result["policies"] = policies
    return result


def summary_rows(result):
    rows = []
    for method, runs in result["runs"].items():
        row = {"method": NAMES[method]}
        for metric in ("success_rate", "optimal_rate", "mean_lifts_capped", "discounted_return"):
            vals = [r["checkpoints"][-1][metric] for r in runs]
            row[metric] = float(np.mean(vals))
            row[metric+"_std"] = float(np.std(vals, ddof=1)) if len(vals)>1 else 0.0
        for metric in ("auc_return", "seconds"):
            vals = [r[metric] for r in runs]
            row[metric] = float(np.mean(vals))
            row[metric+"_std"] = float(np.std(vals, ddof=1)) if len(vals)>1 else 0.0
        rows.append(row)
    return rows


def plot_results(result):
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for method, runs in result["runs"].items():
        for ax, field, collection in (
            (axes[0,0], "discounted_return", "training_curve"),
            (axes[0,1], "raw_reward", "training_curve"),
            (axes[1,0], "optimal_rate", "checkpoints"),
            (axes[1,1], "mean_lifts_capped", "checkpoints")):
            x = np.array([r["episode"] for r in runs[0][collection]])
            y = np.array([[r[field] for r in run[collection]] for run in runs])
            mean = y.mean(axis=0)
            std = y.std(axis=0, ddof=1) if len(runs)>1 else np.zeros_like(mean)
            ax.plot(x, mean, color=COLORS[method], label=NAMES[method], linewidth=2)
            ax.fill_between(x, mean-std, mean+std, color=COLORS[method], alpha=.13)
    axes[0,0].axhline(result["oracle"]["discounted_return"], color="#485660", linestyle="--", label="Точный оптимум")
    axes[1,0].axhline(1.0, color="#485660", linestyle="--")
    axes[1,1].axhline(result["oracle"]["mean_lifts_capped"], color="#485660", linestyle="--", label="Точный оптимум")
    specs = [("Награда при обучении", "Дисконтированный доход G₀"),
             ("Достижение равенства при обучении", "Сумма наград: 0 или 1 (среднее)"),
             ("Оптимальность жадной стратегии", "Доля решений с минимумом поднятий"),
             ("Число поднятий жадной стратегии", "Среднее; неуспех учитывается как 30")]
    for ax, (title, ylabel) in zip(axes.flat, specs):
        ax.set_title(title, loc="left", fontweight="bold", pad=12)
        ax.set_xlabel("Эпизоды обучения")
        ax.set_ylabel(ylabel.replace("30", str(result["config"]["max_steps"])))
        ax.grid(alpha=.15)
        ax.legend(loc="best", frameon=False)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    axes[0,0].set_ylim(-.02, 1.04)
    axes[0,1].set_ylim(-.02, 1.04)
    axes[1,0].set_ylim(-.02, 1.04)
    fig.suptitle(f"Четыре стакана · {len(result['seeds'])} независимых запусков · среднее ± стандартное отклонение", fontsize=14)
    return fig


def policy_q(env, policy):
    # Convert the stored greedy policy to values sufficient for rollout only.
    q = [np.zeros(len(a)) for a in env.actions]
    for sid, action in enumerate(policy):
        if action is not None:
            q[sid][int(np.flatnonzero(env.actions[sid] == action)[0])] = 1.0
    return q


def plot_trajectory(env, path, start):
    import matplotlib.pyplot as plt
    states = [tuple(start)]+[tuple(step["after"]) for step in path]
    fig, axes = plt.subplots(1, len(states), figsize=(3.0*len(states), 3.3), squeeze=False,
                             constrained_layout=True)
    for i, (ax, state) in enumerate(zip(axes[0], states)):
        ax.bar(range(1,5), [env.config.capacity]*4, color="#f0edf4", width=.65)
        ax.bar(range(1,5), state, color="#b94477", width=.65)
        ax.axhline(env.target, linestyle="--", color="#3465d9", linewidth=1)
        for j, value in enumerate(state):
            ax.text(j+1, value+.16, str(value), ha="center", fontsize=12)
        ax.set_ylim(0, env.config.capacity+.8)
        ax.set_xticks(range(1,5), ["1", "2", "3", "4"])
        ax.set_xlabel("Стакан")
        ax.set_title("Начало" if i == 0 else f"Поднятие {i}", loc="left")
        ax.spines[["top", "right"]].set_visible(False)
    return fig


def save_results(result, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory/"results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = summary_rows(result)
    with (directory/"summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Persist each seed's checkpoint metrics for independent auditing.
    with (directory/"checkpoints.csv").open("w", newline="", encoding="utf-8") as f:
        names = ["method", "seed", "episode", "success_rate", "optimal_rate", "mean_lifts_capped", "discounted_return"]
        writer = csv.DictWriter(f, fieldnames=names)
        writer.writeheader()
        for method, runs in result["runs"].items():
            for run in runs:
                for checkpoint in run["checkpoints"]:
                    writer.writerow({"method": method, "seed": run["seed"], **checkpoint})


if __name__ == "__main__":
    import argparse
    import matplotlib
    matplotlib.use("Agg")
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=Config().episodes)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--output", default="results")
    args = parser.parse_args()
    cfg = Config(episodes=args.episodes, eval_every=max(1,args.episodes//40))
    result = run_experiment(cfg, tuple(range(args.seeds)))
    save_results(result, args.output)
    figure = plot_results(result)
    figure.savefig(Path(args.output)/"learning_curves.png", dpi=160)
    print(json.dumps(summary_rows(result), ensure_ascii=False, indent=2))
