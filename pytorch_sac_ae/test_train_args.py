"""Run from the repo root: python -m unittest discover -s pytorch_sac_ae -p test_train_args.py -v."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from train import parse_args


class TrainArgsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.config = Path(self.directory.name) / 'args.json'

    def write_config(self, values):
        self.config.write_text(json.dumps(values))
        return ['--cfg', str(self.config)]

    def assert_parse_error(self, argv, message):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as error:
                parse_args(argv)
        self.assertEqual(error.exception.code, 2)
        self.assertIn(message, stderr.getvalue())

    def test_no_config_preserves_defaults_and_command_line(self):
        args = parse_args([])
        self.assertIsNone(args.cfg)
        self.assertEqual((args.agent, args.domain_name, args.task_name), ('sac_ae', 'cheetah', 'run'))
        self.assertEqual((args.actor_lr, args.hidden_dim, args.seed), (.001, 256, 1))
        self.assertEqual(args.work_dir, 'logs/cheetah_run/sac_ae/1')
        self.assertEqual(args.tr_value_coef, 0.)
        self.assertTrue(args.save_video)
        self.assertFalse(args.gym_env or args.prioritized_replay or args.time_rev)
        args = parse_args(['--agent', 'ppo', '--seed', '7', '--no_save_video'])
        self.assertEqual((args.agent, args.seed, args.save_video), ('ppo', 7, False))
        self.assertEqual(args.work_dir, 'logs/cheetah_run/ppo/7')

    def test_partial_config_and_cli_precedence_in_either_order(self):
        config = self.write_config({'agent': 'ppo', 'actor_lr': .0003, 'seed': 5, 'hidden_dim': 64})
        args = parse_args(config)
        self.assertEqual((args.agent, args.actor_lr, args.seed, args.hidden_dim), ('ppo', .0003, 5, 64))
        self.assertEqual(args.critic_lr, .001)
        overrides = ['--seed', '9', '--hidden_dim', '128']
        for argv in (config + overrides, overrides + config):
            with self.subTest(argv=argv):
                args = parse_args(argv)
                self.assertEqual((args.seed, args.hidden_dim, args.actor_lr), (9, 128, .0003))
        # Config defaults must not leak into later parse calls.
        self.assertEqual(parse_args([]).hidden_dim, 256)

    def test_boolean_config_values_and_overrides(self):
        for enabled in (True, False):
            with self.subTest(enabled=enabled):
                values = {name: enabled for name in (
                    'save_tb', 'save_video', 'save_model', 'save_buffer',
                    'gym_env', 'prioritized_replay', 'time_rev')}
                config = self.write_config(values)
                args = parse_args(config)
                for name in values:
                    self.assertIs(getattr(args, name), enabled)
                prefix = '--no_' if enabled else '--'
                overrides = [prefix + name for name in values if name != 'time_rev']
                overrides += ['--time_rev', 'false' if enabled else 'true']
                args = parse_args(config + overrides)
                for name in values:
                    self.assertIs(getattr(args, name), not enabled)

    def test_float_options_accept_json_numbers_even_with_integer_defaults(self):
        args = parse_args(self.write_config({
            'actor_log_std_min': -7.5, 'actor_log_std_max': 1.5, 'actor_lr': 1}))
        self.assertEqual((args.actor_log_std_min, args.actor_log_std_max, args.actor_lr), (-7.5, 1.5, 1))

    def test_all_archived_configs_preserve_settings_and_derive_output_paths(self):
        root = Path(__file__).resolve().parent / 'args'
        configs = sorted(root.glob('*/*/*.json'))
        self.assertTrue(configs)
        for config in configs:
            with self.subTest(config=str(config)):
                saved = json.loads(config.read_text())
                args = parse_args(['--cfg', str(config)])
                for name, value in saved.items():
                    if name not in ('cfg', 'work_dir'):
                        self.assertEqual(getattr(args, name), value)
                expected = Path('logs') / '{}_{}'.format(saved['domain_name'], saved['task_name'])
                agent_dir = ('trsppo_policy_0.1_value_0' if saved['agent'] == 'trsppo'
                             else saved['agent'])
                expected = expected / agent_dir / str(saved['seed'])
                self.assertEqual(Path(args.work_dir), expected)

    def test_work_dir_uses_final_settings_and_ignores_archived_path(self):
        config = self.write_config({
            'agent': 'trsppo', 'domain_name': 'cheetah', 'task_name': 'bidirectional',
            'seed': 1, 'work_dir': './logs/cheetah_bidirectional/trsppo/2'})
        self.assertEqual(parse_args(config).work_dir,
                         'logs/cheetah_bidirectional/trsppo_policy_0.1_value_0/1')
        for agent in ('sac_ae', 'ppo', 'trsppo'):
            overrides = ['--agent', agent, '--domain_name', 'pendulum',
                         '--task_name', 'two_poles', '--seed', '3', '--tr_policy_coef', '0.25',
                         '--tr_value_coef', '0.05']
            for argv in (config + overrides, overrides + config):
                with self.subTest(agent=agent, argv=argv):
                    args = parse_args(argv)
                    agent_dir = 'trsppo_policy_0.25_value_0.05' if agent == 'trsppo' else agent
                    self.assertEqual(args.work_dir, 'logs/pendulum_two_poles/{}/3'.format(agent_dir))
        self.assert_parse_error(['--work_dir', '/tmp/custom'], 'unrecognized arguments: --work_dir')

    def test_value_symmetry_config_and_cli_overrides_round_trip(self):
        config = self.write_config({'agent': 'trsppo', 'tr_policy_coef': 0.2, 'tr_value_coef': 0.05})
        args = parse_args(config)
        self.assertEqual(args.work_dir, 'logs/cheetah_run/trsppo_policy_0.2_value_0.05/1')
        for argv in (config + ['--tr_value_coef', '0'], ['--tr_value_coef', '0'] + config):
            self.assertEqual(parse_args(argv).work_dir, 'logs/cheetah_run/trsppo_policy_0.2_value_0/1')
        self.assertEqual(vars(parse_args(self.write_config(vars(args)))), vars(args))
        for coefficient in ('-1', 'nan', 'inf'):
            self.assert_parse_error(['--agent', 'trsppo', '--tr_value_coef', coefficient],
                                    '--tr_value_coef must be finite and nonnegative')
        self.assert_parse_error(self.write_config({'agent': 'trsppo', 'tr_value_coef': -0.1}),
                                '--tr_value_coef must be finite and nonnegative')

    def test_saved_merged_args_can_be_reloaded(self):
        args = parse_args(self.write_config({'agent': 'ppo', 'seed': 5}) + ['--seed', '8'])
        saved = vars(args).copy()
        saved['cfg'] = '/missing/original/config.json'
        reloaded = parse_args(self.write_config(saved))
        self.assertEqual(vars(reloaded), vars(args))

    def test_invalid_files_and_config_fields_report_parser_errors(self):
        config = ['--cfg', str(self.config)]
        self.assert_parse_error(config, 'cannot load --cfg')
        self.config.write_text('{')
        self.assert_parse_error(config, 'cannot load --cfg')
        for values, message in (
                ([], 'JSON object'),
                ({'actor_lrr': .001}, 'unknown --cfg arguments: actor_lrr'),
                ({'agent': 'unknown'}, 'agent must be one of'),
                ({'seed': 1.5}, 'seed must have type int'),
                ({'batch_size': True}, 'batch_size must have type int'),
                ({'actor_lr': '.001'}, 'actor_lr must have type float'),
                ({'save_video': 'false'}, 'save_video must have type bool'),
                ({'domain_name': None}, 'domain_name must have type str')):
            with self.subTest(values=values):
                self.assert_parse_error(self.write_config(values), message)

    def test_training_constraints_apply_after_cli_overrides(self):
        config = self.write_config({'agent': 'ppo', 'rollout_steps': 0, 'time_rev': True})
        args = parse_args(config + ['--rollout_steps', '128', '--time_rev', 'false'])
        self.assertEqual(args.rollout_steps, 128)
        self.assertFalse(args.time_rev)
        self.assert_parse_error(config + ['--time_rev', 'false'], '--rollout_steps must be positive')
        config = self.write_config({'agent': 'trsppo', 'domain_name': 'pendulum', 'task_name': 'swingup'})
        self.assert_parse_error(config, 'requires task_name')
        args = parse_args(config + ['--task_name', 'two_poles'])
        self.assertEqual((args.domain_name, args.task_name), ('pendulum', 'two_poles'))


if __name__ == '__main__':
    unittest.main()
