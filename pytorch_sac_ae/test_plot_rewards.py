"""Test aggregation and TensorBoard input without requiring a simulator."""

from pathlib import Path
import tempfile
import unittest
import warnings

import numpy as np
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter

from plot_rewards import REWARD_TAG, RunCurve, aggregate_runs, load_runs, read_run


def curve(steps, rewards):
    return RunCurve(Path('run'), np.array(steps), np.array(rewards, dtype=float))


class PlotRewardsTests(unittest.TestCase):
    def write_events(self, run_dir, points, tag=REWARD_TAG):
        writer = EventFileWriter(str(run_dir / 'tb'))
        try:
            for step, value, wall_time in points:
                writer.add_event(Event(step=step, wall_time=wall_time, summary=Summary(
                    value=[Summary.Value(tag=tag, simple_value=value)])))
            writer.flush()
        finally:
            writer.close()

    def test_mean_and_std_across_runs_then_step_sampling(self):
        runs = [curve([0, 10, 20], [1, 3, 5]), curve([0, 10, 20], [3, 7, 9])]
        stats = aggregate_runs(runs, 20)
        np.testing.assert_array_equal(stats.steps, [0, 20])
        np.testing.assert_allclose(stats.mean, [2, 7])
        np.testing.assert_allclose(stats.std, [1, 2])
        np.testing.assert_array_equal(stats.counts, [2, 2])

    def test_unequal_lengths_and_start_times_are_not_extrapolated(self):
        runs = [curve([0, 10], [0, 10]), curve([10, 20, 30], [30, 40, 50])]
        stats = aggregate_runs(runs, 10)
        np.testing.assert_array_equal(stats.steps, [0, 10, 20, 30])
        np.testing.assert_allclose(stats.mean, [0, 20, 40, 50])
        np.testing.assert_allclose(stats.std, [0, 10, 0, 0])
        np.testing.assert_array_equal(stats.counts, [1, 2, 1, 1])

    def test_different_logging_frequencies_interpolate_on_common_grid(self):
        runs = [curve([0, 20], [0, 20]), curve([0, 10, 20], [0, 30, 40])]
        stats = aggregate_runs(runs, 10)
        np.testing.assert_allclose(stats.mean, [0, 20, 30])
        np.testing.assert_allclose(stats.std, [0, 10, 10])
        np.testing.assert_array_equal(stats.counts, [2, 2, 2])

    def test_single_run_has_zero_std_and_no_extra_final_point(self):
        stats = aggregate_runs([curve([0, 10, 25], [0, 20, 50])], 10)
        np.testing.assert_array_equal(stats.steps, [0, 10, 20])
        np.testing.assert_allclose(stats.mean, [0, 20, 40])
        np.testing.assert_array_equal(stats.std, [0, 0, 0])
        with self.assertRaisesRegex(ValueError, 'positive'):
            aggregate_runs([curve([0], [1])], 0)
        with self.assertRaisesRegex(ValueError, 'At least one'):
            aggregate_runs([], 10)
        with self.assertRaisesRegex(ValueError, 'sampling grid'):
            aggregate_runs([curve([1, 2], [1, 2])], 10)

    def test_tensorboard_duplicates_keep_latest_wall_time(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            self.write_events(run_dir, [(0, 1, 1), (10, 30, 3), (10, 20, 2), (20, 40, 4)])
            run = read_run(run_dir)
            np.testing.assert_array_equal(run.steps, [0, 10, 20])
            np.testing.assert_array_equal(run.rewards, [1, 30, 40])

    def test_discovery_includes_every_run_and_ignores_other_algorithms(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for algorithm, index in (('ppo', '1'), ('ppo', '2'), ('trsppo', 'seed2'), ('sac_ae', '1')):
                self.write_events(root / 'cheetah_run' / algorithm / index, [(0, 1, 1), (10, 2, 2)])
            self.write_events(root / 'cheetah_run' / 'ppo' / 'empty', [(0, 1, 1)], tag='train/episode_reward')
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                runs = load_runs(root, 'cheetah', 'run')
            self.assertTrue(any('missing eval/episode_reward' in str(item.message) for item in caught))
            self.assertEqual(set(runs), {'ppo', 'trsppo'})
            self.assertEqual([run.path.name for run in runs['ppo']], ['1', '2'])
            self.assertEqual([run.path.name for run in runs['trsppo']], ['seed2'])


if __name__ == '__main__':
    unittest.main()
