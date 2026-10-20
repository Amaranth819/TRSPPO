"""Plot evaluation-return learning curves from PPO/TRSPPO TensorBoard runs."""

import argparse
from collections import namedtuple
from pathlib import Path
import warnings

import numpy as np


REWARD_TAG = 'eval/episode_reward'
ALGORITHM_STYLES = {
    'ppo': ('PPO', '#1f77b4'),
    'trsppo': ('TRSPPO', '#ff7f0e'),
}
RunCurve = namedtuple('RunCurve', 'path steps rewards')
CurveStats = namedtuple('CurveStats', 'steps mean std counts')


def read_run(run_dir):
    """Read all reward scalars, retaining the latest wall-time value per step."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    run_dir = Path(run_dir)
    tb_dir = run_dir / 'tb'
    if not any(tb_dir.glob('events.out.tfevents.*')):
        warnings.warn('Skipping {}: no TensorBoard event files'.format(run_dir))
        return None
    # Disable TensorBoard's scalar reservoir sampling so no logged points are lost.
    accumulator = EventAccumulator(str(tb_dir), size_guidance={'scalars': 0})
    accumulator.Reload()
    if REWARD_TAG not in accumulator.Tags().get('scalars', []):
        warnings.warn('Skipping {}: missing {}'.format(run_dir, REWARD_TAG))
        return None
    rewards_by_step = {}
    for event in sorted(accumulator.Scalars(REWARD_TAG), key=lambda event: event.wall_time):
        if np.isfinite(event.value):
            rewards_by_step[event.step] = event.value
        else:
            warnings.warn('Ignoring non-finite reward in {} at step {}'.format(run_dir, event.step))
    if not rewards_by_step:
        warnings.warn('Skipping {}: no finite rewards'.format(run_dir))
        return None
    steps = np.array(sorted(rewards_by_step), dtype=np.int64)
    rewards = np.array([rewards_by_step[step] for step in steps], dtype=float)
    return RunCurve(run_dir, steps, rewards)


def load_runs(log_root, domain, task):
    """Find every run in logs/{domain}_{task}/{ppo,trsppo}/{run}/tb/."""
    task_dir = Path(log_root) / '{}_{}'.format(domain, task)
    if not task_dir.is_dir():
        raise ValueError('Task log directory does not exist: {}'.format(task_dir))
    curves = {}
    for algorithm in ALGORITHM_STYLES:
        algorithm_dir = task_dir / algorithm
        runs = []
        if algorithm_dir.is_dir():
            for run_dir in sorted(algorithm_dir.iterdir()):
                if run_dir.is_dir():
                    curve = read_run(run_dir)
                    if curve is not None:
                        runs.append(curve)
        if runs:
            curves[algorithm] = runs
        else:
            warnings.warn('No usable {} runs in {}'.format(algorithm, task_dir))
    if not curves:
        raise ValueError('No PPO/TRSPPO {} data found in {}'.format(REWARD_TAG, task_dir))
    return curves


def aggregate_runs(runs, step_interval):
    """Mean and population std on a step grid, using only runs covering each step.

    Exact logged values are used when present; otherwise interpolate linearly
    between neighboring evaluations. Never extrapolate beyond a run's endpoints.
    """
    if step_interval <= 0:
        raise ValueError('step_interval must be positive')
    if not runs:
        raise ValueError('At least one run is required')
    last_step = max(run.steps[-1] for run in runs)
    steps = np.arange(0, last_step + 1, step_interval, dtype=np.int64)
    rewards = np.full((len(runs), len(steps)), np.nan)
    for index, run in enumerate(runs):
        available = (steps >= run.steps[0]) & (steps <= run.steps[-1])
        rewards[index, available] = np.interp(steps[available], run.steps, run.rewards)
    counts = np.sum(np.isfinite(rewards), axis=0)
    available = counts > 0
    if not available.any():
        raise ValueError('No evaluations overlap the sampling grid; reduce --step_interval')
    rewards = rewards[:, available]
    return CurveStats(steps[available], np.nanmean(rewards, axis=0),
                      np.nanstd(rewards, axis=0, ddof=0), counts[available])


def plot_curves(curves, domain, task, step_interval, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    try:
        for algorithm, (label, color) in ALGORITHM_STYLES.items():
            if algorithm not in curves:
                continue
            runs = curves[algorithm]
            stats = aggregate_runs(runs, step_interval)
            ax.plot(stats.steps, stats.mean, color=color, linewidth=2, linestyle='-',
                    marker='o' if len(stats.steps) == 1 else None,
                    label='{} ({} run{})'.format(label, len(runs), '' if len(runs) == 1 else 's'))
            ax.fill_between(stats.steps, stats.mean - stats.std, stats.mean + stats.std,
                            color=color, alpha=0.2, linewidth=0)
            print('{}: {} runs, {} plotted points, {}-{} runs contributing per point'.format(
                label, len(runs), len(stats.steps), stats.counts.min(), stats.counts.max()))
            for run in runs:
                print('  {}: {} evaluations, steps {}-{}'.format(
                    run.path, len(run.steps), run.steps[0], run.steps[-1]))
        ax.set_title('{} / {}'.format(domain, task))
        ax.set_xlabel('Training steps')
        ax.set_ylabel('Evaluation episode return')
        ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        ax.grid(True, alpha=0.25)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(frameon=False)
        fig.tight_layout()
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output), dpi=200, bbox_inches='tight')
    finally:
        plt.close(fig)
    print('Saved {}'.format(output))
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log_root', type=Path, default=Path('logs'),
                        help='root containing {domain}_{task}/{algorithm}/{run}/tb (default: logs)')
    parser.add_argument('--domain', required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--step_interval', type=int, default=10000,
                        help='sample every X training steps (default: 10000); interpolate missing steps')
    parser.add_argument('--output', type=Path,
                        help='output image or PDF path (default: plots/{domain}_{task}.png)')
    args = parser.parse_args(argv)
    if args.step_interval <= 0:
        parser.error('--step_interval must be positive')
    output = args.output or Path('plots') / '{}_{}.png'.format(args.domain, args.task)
    try:
        curves = load_runs(args.log_root, args.domain, args.task)
        plot_curves(curves, args.domain, args.task, args.step_interval, output)
    except ImportError as error:
        parser.error('{}; install plotting dependencies with: python -m pip install matplotlib tensorboard'.format(error))
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
