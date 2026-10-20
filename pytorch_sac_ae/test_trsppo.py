"""Run from the repo root: python -m unittest discover -s pytorch_sac_ae -p test_trsppo.py -v."""

import contextlib
import io
import math
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from ppo import PPOAgent, RolloutBuffer
from trsppo import TRSPPOAgent, TRSPPORolloutBuffer, reverse_action, reverse_observation


class TRSPPOTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        np.random.seed(7)

    def agent(self, cls=TRSPPOAgent, obs_dim=17, action_dim=6, **kwargs):
        return cls((obs_dim,), (action_dim,), 'cpu', hidden_dim=16, batch_size=4,
                   update_epochs=2, target_kl=0, **kwargs)

    def paired_buffers(self, agent, count=7, obs_dim=17, action_dim=6):
        plain = RolloutBuffer((obs_dim,), (action_dim,), count + 3)
        paired = TRSPPORolloutBuffer((obs_dim,), (action_dim,), count + 3)
        for i in range(count):
            obs = np.random.randn(obs_dim).astype(np.float32)
            next_obs = np.random.randn(obs_dim).astype(np.float32)
            _, raw_action, log_prob, value = agent.act(obs)
            transition = (obs, raw_action, log_prob, float(i + 1), value,
                          agent.value(next_obs), 1., i == 3)
            plain.add(*transition)
            paired.add(*transition, next_obs=next_obs)
        return plain, paired

    def test_mapping_preserves_configuration_and_reverses_every_velocity(self):
        for domain, obs_dim, action_dim, velocity_start, velocity_stop in (
                ('cheetah', 17, 6, 8, 17), ('acrobot', 6, 1, 4, 6),
                ('cartpole', 5, 1, 3, 5),
                ('walker', 24, 6, 15, 24), ('hopper', 15, 4, 6, 13)):
            with self.subTest(domain=domain):
                obs = torch.arange(1., 6 * obs_dim + 1).reshape(2, 3, obs_dim)
                original = obs.clone()
                transformed = reverse_observation(obs, domain)
                torch.testing.assert_close(transformed[..., :velocity_start], original[..., :velocity_start])
                velocities = slice(velocity_start, velocity_stop)
                torch.testing.assert_close(transformed[..., velocities], -original[..., velocities])
                torch.testing.assert_close(transformed[..., velocity_stop:], original[..., velocity_stop:])
                torch.testing.assert_close(reverse_observation(transformed, domain), original)
                torch.testing.assert_close(obs, original)
                self.assertNotEqual(obs.data_ptr(), transformed.data_ptr())
                action = torch.linspace(-1, 1, action_dim)
                torch.testing.assert_close(reverse_action(action, domain), action)
                self.assertNotEqual(reverse_action(action, domain).data_ptr(), action.data_ptr())
                with self.assertRaises(ValueError):
                    reverse_observation(torch.zeros(obs_dim + 1), domain)
                with self.assertRaises(ValueError):
                    reverse_action(torch.zeros(action_dim + 1), domain)
        # Existing callers retain the cheetah mapping without a new argument.
        obs = torch.arange(17.)
        torch.testing.assert_close(reverse_observation(obs), reverse_observation(obs, 'cheetah'))
        torch.testing.assert_close(reverse_action(torch.ones(6)), torch.ones(6))
        with self.assertRaises(ValueError):
            reverse_observation(torch.zeros(17), 'unknown')

    def test_mappings_match_simulator_velocity_reversal(self):
        from dm_control import suite
        from dmc2gym.wrappers import _flatten_obs
        for domain, task in (('acrobot', 'swingup'), ('cartpole', 'swingup'),
                             ('walker', 'stand'),
                             ('walker', 'walk'), ('walker', 'run'),
                             ('hopper', 'stand'), ('hopper', 'hop')):
            with self.subTest(domain=domain, task=task):
                env = suite.load(domain, task, task_kwargs={'random': 7})
                try:
                    for _ in range(3):
                        env.reset()
                        physics = env.physics
                        position = physics.data.qpos.copy()
                        velocity = np.linspace(-1.3, 2.1, physics.model.nv)
                        with physics.reset_context():
                            physics.data.qpos[:] = position
                            physics.data.qvel[:] = velocity
                            # Exercise cartpole's control penalty at nonzero force.
                            if domain == 'cartpole':
                                physics.data.ctrl[:] = .3
                        obs = _flatten_obs(env.task.get_observation(physics))
                        reward = env.task.get_reward(physics)
                        control = physics.data.ctrl.copy()
                        with physics.reset_context():
                            physics.data.qpos[:] = position
                            physics.data.qvel[:] = -velocity
                            physics.data.ctrl[:] = control
                        expected = _flatten_obs(env.task.get_observation(physics))
                        reversed_obs = reverse_observation(torch.as_tensor(obs), domain).numpy()
                        if domain == 'hopper':
                            # Compare mechanical features with the simulator.
                            # Preserved touch readings are an approximation,
                            # not necessarily the recomputed contact forces.
                            self.assertEqual(obs.shape, (15,))
                            self.assertEqual(env.action_spec().shape, (4,))
                            np.testing.assert_allclose(reversed_obs[:13], expected[:13], rtol=0, atol=1e-12)
                            np.testing.assert_array_equal(reversed_obs[13:], obs[13:])
                        else:
                            np.testing.assert_allclose(reversed_obs, expected, rtol=0, atol=1e-12)
                        if (domain, task) in (('acrobot', 'swingup'), ('cartpole', 'swingup'),
                                               ('walker', 'stand'),
                                               ('hopper', 'stand')):
                            self.assertAlmostEqual(env.task.get_reward(physics), reward, places=12)
                finally:
                    env.close()

    def test_loss_uses_next_state_detached_reference_and_action_dimension_mean(self):
        for domain, obs_dim, action_dim, velocity_start in (
                ('cheetah', 17, 6, 8), ('acrobot', 6, 1, 4),
                ('cartpole', 5, 1, 3),
                ('walker', 24, 6, 15), ('hopper', 15, 4, 6)):
            with self.subTest(domain=domain):
                agent = self.agent(obs_dim=obs_dim, action_dim=action_dim, domain_name=domain)
                agent.actor.mean = torch.nn.Linear(obs_dim, action_dim, bias=False)
                with torch.no_grad():
                    agent.actor.mean.weight.zero_()
                    for j in range(action_dim):
                        agent.actor.mean.weight[j, j] = .5
                        agent.actor.mean.weight[j, velocity_start + j] = .25
                        if domain == 'hopper':
                            agent.actor.mean.weight[j, 13 + j % 2] = .125
                obs = torch.arange(3. * obs_dim).reshape(3, obs_dim) / 100
                next_obs = torch.full((3, obs_dim), .7)
                # Masked endpoints must not be evaluated: 0 * NaN is still NaN.
                obs[1] = float('nan')
                next_obs[1] = float('nan')
                obs.requires_grad_()
                next_obs.requires_grad_()
                valid = torch.tensor([True, False, True])
                velocity = slice(velocity_start, velocity_start + action_dim)
                target = .5 * obs.detach()[valid, :action_dim] + .25 * obs.detach()[valid, velocity]
                prediction = (.5 * next_obs.detach()[valid, :action_dim] -
                              .25 * next_obs.detach()[valid, velocity])
                if domain == 'hopper':
                    touch_indices = [13, 14, 13, 14]
                    target += .125 * obs.detach()[valid][:, touch_indices]
                    prediction += .125 * next_obs.detach()[valid][:, touch_indices]
                target, prediction = target.tanh(), prediction.tanh()
                expected = (prediction - target).square().sum() / (2 * action_dim)
                loss = agent.actor_symmetry_loss(obs, next_obs, valid)
                torch.testing.assert_close(loss, expected)
                loss.backward()
                self.assertIsNone(obs.grad)
                self.assertGreater(next_obs.grad[valid].abs().sum().item(), 0)
                self.assertEqual(next_obs.grad[1].abs().sum().item(), 0)
                self.assertTrue(torch.isfinite(agent.actor.mean.weight.grad).all())
                self.assertGreater(agent.actor.mean.weight.grad.abs().sum().item(), 0)
                self.assertIsNone(agent.actor.log_std.grad)
                self.assertTrue(all(p.grad is None for p in agent.critic.parameters()))

    def test_pair_storage_copies_endpoints_and_masks_both_kinds_of_episode_end(self):
        buffer = TRSPPORolloutBuffer((17,), (6,), 3)
        for i, (discount, done) in enumerate(((1., False), (0., True), (1., True))):
            obs = np.full(17, i, dtype=np.float32)
            next_obs = obs + 10
            buffer.add(obs, np.zeros(6), 0., 1., 0., 2., discount, done, next_obs=next_obs)
            obs[:] = -99
            next_obs[:] = -99
        batch = buffer.batch(.9, .95, 'cpu')
        torch.testing.assert_close(batch['tr_valid'], torch.tensor([True, False, False]))
        order = torch.tensor([2, 0, 1])
        torch.testing.assert_close(batch['next_obs'][order] - batch['obs'][order], torch.full((3, 17), 10.))
        # Both episode ends are masked for TR, while GAE still bootstraps the timeout.
        self.assertAlmostEqual(batch['returns'][1].item(), 1.)
        self.assertAlmostEqual(batch['returns'][2].item(), 2.8, places=6)
        with tempfile.TemporaryDirectory() as directory:
            buffer.save(directory)
            with np.load(os.path.join(directory, 'rollout.npz')) as saved:
                np.testing.assert_array_equal(saved['next_obs'], buffer.next_obs)
                np.testing.assert_array_equal(saved['dones'], [0., 1., 1.])
        buffer.clear()
        self.assertEqual(buffer.size, 0)
        with self.assertRaises(ValueError):
            buffer.batch(.9, .95, 'cpu')

    def test_zero_coefficient_matches_vanilla_ppo_updates_exactly(self):
        for domain, obs_dim, action_dim in (('cheetah', 17, 6), ('acrobot', 6, 1),
                                           ('cartpole', 5, 1),
                                           ('walker', 24, 6), ('hopper', 15, 4)):
            with self.subTest(domain=domain):
                plain_agent = self.agent(PPOAgent, obs_dim=obs_dim, action_dim=action_dim, entropy_coef=.01)
                tr_agent = self.agent(obs_dim=obs_dim, action_dim=action_dim, domain_name=domain,
                                      tr_policy_coef=0, entropy_coef=.01)
                tr_agent.actor.load_state_dict(plain_agent.actor.state_dict())
                tr_agent.critic.load_state_dict(plain_agent.critic.state_dict())
                plain, paired = self.paired_buffers(plain_agent, obs_dim=obs_dim, action_dim=action_dim)
                torch.manual_seed(29)
                plain_metrics = plain_agent.update(plain)
                plain_rng = torch.get_rng_state()
                torch.manual_seed(29)
                tr_metrics = tr_agent.update(paired)
                torch.testing.assert_close(torch.get_rng_state(), plain_rng)
                for network in ('actor', 'critic'):
                    for name, value in getattr(plain_agent, network).state_dict().items():
                        torch.testing.assert_close(value, getattr(tr_agent, network).state_dict()[name], rtol=0, atol=0)
                for key, value in plain_metrics.items():
                    self.assertEqual(value, tr_metrics[key])
                self.assertGreater(tr_metrics['tr_policy_loss'], 0.)
                self.assertEqual(tr_metrics['tr_policy_loss_weighted'], 0.)

    def test_enabled_loss_changes_actor_but_not_critic_update(self):
        plain_agent = self.agent(PPOAgent)
        tr_agent = self.agent(tr_policy_coef=.1)
        tr_agent.actor.load_state_dict(plain_agent.actor.state_dict())
        tr_agent.critic.load_state_dict(plain_agent.critic.state_dict())
        plain, paired = self.paired_buffers(plain_agent)
        torch.manual_seed(31)
        plain_agent.update(plain)
        torch.manual_seed(31)
        metrics = tr_agent.update(paired)
        self.assertTrue(any(not torch.equal(a, b) for a, b in
                            zip(plain_agent.actor.parameters(), tr_agent.actor.parameters())))
        for a, b in zip(plain_agent.critic.parameters(), tr_agent.critic.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertEqual(metrics['updates'], 4)
        self.assertTrue(all(math.isfinite(x) for x in metrics.values()))
        self.assertAlmostEqual(metrics['tr_policy_loss_weighted'], .1 * metrics['tr_policy_loss'])
        self.assertAlmostEqual(metrics['actor_total_loss'],
                               metrics['actor_loss'] + metrics['tr_policy_loss_weighted'], places=6)
        self.assertAlmostEqual(metrics['tr_valid_pair_fraction'], 6. / 7, places=6)

    def test_all_masked_and_single_transition_rollouts_are_safe(self):
        for terminal in (False, True):
            agent = self.agent()
            _, buffer = self.paired_buffers(agent, count=1)
            buffer.dones[0] = terminal
            metrics = agent.update(buffer)
            self.assertTrue(all(math.isfinite(x) for x in metrics.values()))
            self.assertEqual(metrics['tr_valid_pair_fraction'], float(not terminal))
            if terminal:
                self.assertEqual(metrics['tr_policy_loss'], 0.)
                self.assertEqual(metrics['tr_policy_loss_weighted'], 0.)

    def test_checkpoint_restores_networks_and_optimizers(self):
        for domain, obs_dim, action_dim in (('cheetah', 17, 6), ('acrobot', 6, 1),
                                           ('cartpole', 5, 1),
                                           ('walker', 24, 6), ('hopper', 15, 4)):
            with self.subTest(domain=domain):
                settings = dict(obs_dim=obs_dim, action_dim=action_dim, domain_name=domain)
                agent = self.agent(**settings)
                _, buffer = self.paired_buffers(agent, obs_dim=obs_dim, action_dim=action_dim)
                agent.update(buffer)
                obs = buffer.obs[0]
                with tempfile.TemporaryDirectory() as directory:
                    agent.save(directory, 7)
                    self.assertTrue(os.path.isfile(os.path.join(directory, 'trsppo_7.pt')))
                    restored = self.agent(**settings)
                    restored.load(directory, 7)
                    np.testing.assert_array_equal(agent.select_action(obs), restored.select_action(obs))
                    self.assertEqual(agent.value(obs), restored.value(obs))
                    self.assertTrue(restored.actor_optimizer.state)
                    self.assertTrue(restored.critic_optimizer.state)

    def test_agent_rejects_invalid_mapping_dimensions_and_coefficients(self):
        for coefficient in (-1., float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                self.agent(tr_policy_coef=coefficient)
        with self.assertRaises(ValueError):
            TRSPPOAgent((18,), (6,), 'cpu')
        with self.assertRaises(ValueError):
            TRSPPOAgent((17,), (12,), 'cpu')

    def test_agent_rejects_wrong_domain_dimensions(self):
        for domain, obs_dim, action_dim in (('acrobot', 6, 1), ('cartpole', 5, 1),
                                           ('walker', 24, 6), ('hopper', 15, 4)):
            with self.subTest(domain=domain):
                with self.assertRaises(ValueError):
                    self.agent(obs_dim=obs_dim + 1, action_dim=action_dim, domain_name=domain)
                with self.assertRaises(ValueError):
                    self.agent(obs_dim=obs_dim, action_dim=action_dim + 1, domain_name=domain)
                with self.assertRaises(ValueError):
                    self.agent(domain_name=domain)
        with self.assertRaises(ValueError):
            self.agent(domain_name='unknown')

    def test_cli_registers_supported_tasks_and_passes_domain_to_agent(self):
        from train import parse_args, make_agent
        for domain, task, obs_dim, action_dim in (
                ('cheetah', 'run', 17, 6), ('cheetah', 'bidirectional', 17, 6),
                ('cartpole', 'swingup', 5, 1),
                ('acrobot', 'swingup', 6, 1), ('walker', 'stand', 24, 6),
                ('walker', 'walk', 24, 6), ('walker', 'run', 24, 6),
                ('hopper', 'stand', 15, 4), ('hopper', 'hop', 15, 4)):
            with self.subTest(domain=domain, task=task):
                argv = ['train.py', '--agent', 'trsppo', '--domain_name', domain, '--task_name', task]
                with patch('sys.argv', argv):
                    args = parse_args()
                self.assertEqual(args.tr_policy_coef, .1)
                self.assertEqual(args.rollout_steps, 1000)
                self.assertFalse(hasattr(args, 'tr_warmup_updates'))
                args.hidden_dim = 16
                agent = make_agent((obs_dim,), (action_dim,), args, 'cpu')
                self.assertIsInstance(agent, TRSPPOAgent)
                self.assertEqual(agent.domain_name, domain)
        for extra in (['--domain_name', 'walker', '--task_name', 'unknown'],
                      ['--domain_name', 'hopper', '--task_name', 'run'],
                      ['--domain_name', 'cartpole', '--task_name', 'two_poles'],
                      ['--domain_name', 'cartpole', '--task_name', 'three_poles'],
                      ['--domain_name', 'acrobot', '--task_name', 'swingup_sparse'],
                      ['--domain_name', 'reacher', '--task_name', 'easy'],
                      ['--task_name', 'unknown'], ['--tr_policy_coef', '-1'],
                      ['--tr_policy_coef', 'nan'], ['--time_rev', 'True']):
            with patch('sys.argv', ['train.py', '--agent', 'trsppo'] + extra):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        parse_args()


if __name__ == '__main__':
    unittest.main()
