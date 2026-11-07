"""Plot existing ml_logger reward curves without launching the simulator."""

import argparse
import math
from pathlib import Path
import pickle


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRIC = "train/episode/rew_total/mean"
RUN_LABELS = {"ppo": "PPO", "trsppo": "TRS-PPO"}
LEGACY_RUN_NAMES = {"ppo": "train_go2", "trsppo": "train_trsppo_go2"}


def latest_runs(log_root, robot="go2"):
    runs = []
    for name in RUN_LABELS:
        candidates = sorted((log_root / robot / name).glob("*/metrics.pkl"))
        if not candidates and robot == "go2":
            candidates = sorted(log_root.glob(f"*/{LEGACY_RUN_NAMES[name]}/*/metrics.pkl"))
        if candidates:
            runs.append(candidates[-1].parent)
    return runs


def read_curve(metrics_path, metric):
    """Read appended pickle records, including logs that are still being written."""
    rows = []
    with metrics_path.open("rb") as stream:
        while True:
            try:
                record = pickle.load(stream)
            except (EOFError, pickle.UnpicklingError):
                # A running logger may not have finished its final record yet.
                break
            rows.extend(record if isinstance(record, list) else [record])
    points = []
    for row in rows:
        if "iterations" in row and metric in row:
            iteration, value = float(row["iterations"]), float(row[metric])
            if math.isfinite(iteration) and math.isfinite(value):
                points.append((iteration, value))
    if not points:
        raise ValueError(f"No finite values for {metric!r} in {metrics_path}")
    return sorted(points)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path,
                        help="Run directories or metrics.pkl files; defaults to the latest PPO and TRS-PPO runs.")
    parser.add_argument("--log-root", type=Path, default=REPO_ROOT / "runs/gait-conditioned-agility")
    parser.add_argument("--robot", default="go2", help="Robot whose latest PPO/TRS-PPO runs to plot.")
    parser.add_argument("--metric", default=DEFAULT_METRIC)
    parser.add_argument("--smooth", type=int, default=11,
                        help="Trailing moving-average window in logged points; use 1 for raw values.")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "tmp/training_rewards.png")
    args = parser.parse_args(argv)
    if args.smooth < 1:
        parser.error("--smooth must be at least 1")
    runs = args.runs if args.runs is not None else latest_runs(args.log_root, args.robot)
    if not runs:
        parser.error(f"No PPO/TRS-PPO metrics found for {args.robot!r} under {args.log_root}; provide --runs explicitly")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
    for run in runs:
        metrics_path = run / "metrics.pkl" if run.is_dir() else run
        try:
            points = read_curve(metrics_path, args.metric)
        except (OSError, ValueError) as error:
            parser.error(str(error))
        iterations, values = np.asarray(points).T
        algorithm = metrics_path.parent.parent.name
        if algorithm in RUN_LABELS:
            label = f"{metrics_path.parent.parent.parent.name} / {RUN_LABELS[algorithm]}"
        elif algorithm in LEGACY_RUN_NAMES.values():
            name = next(name for name, legacy in LEGACY_RUN_NAMES.items() if legacy == algorithm)
            label = f"go2 / {RUN_LABELS[name]}"
        else:
            label = metrics_path.parent.name
        if args.smooth > 1:
            sums = np.convolve(values, np.ones(args.smooth), mode="full")[:len(values)]
            counts = np.minimum(np.arange(len(values)) + 1, args.smooth)
            line, = ax.plot(iterations, sums / counts, linewidth=2, label=label)
            ax.plot(iterations, values, color=line.get_color(), linewidth=0.7, alpha=0.25)
        else:
            ax.plot(iterations, values, linewidth=1.5, label=label)
        print(f"{label}: {metrics_path}")
        print(f"  {len(points)} logged points; last iteration {iterations[-1]:g}; last value {values[-1]:.6g}")

    ax.set_title("Training rewards" if args.metric == DEFAULT_METRIC else args.metric)
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Mean total reward (logged)" if args.metric == DEFAULT_METRIC else args.metric)
    ax.grid(alpha=0.25)
    ax.legend()
    if args.smooth > 1:
        fig.text(0.5, 0.015, f"Trailing mean over {args.smooth} logged points; faint lines show raw values.",
                 ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(args.output))
    plt.close(fig)
    print(f"Saved {args.output.resolve()}")


if __name__ == "__main__":
    main()
