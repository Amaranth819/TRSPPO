"""Actor-only time-reversal consistency for PPO on supported state observations.

For a real transition (o, a, r, next_o), regularize the deterministic action at
T(next_o) toward the detached current-policy action at o. This is an inductive
bias, not a guarantee that the environment dynamics are reversible.
"""

from collections import namedtuple
import math
import os

import numpy as np
import torch
from torch.nn import functional as F

from ppo import PPOAgent, RolloutBuffer


# Offsets follow dmc2gym's flattened observation order. Explicit velocity bounds
# preserve trailing sensor features, including hopper's two touch readings.
TimeReversalSpec = namedtuple('TimeReversalSpec',
                             ('obs_dim', 'action_dim', 'velocity_start',
                              'velocity_stop', 'tasks'))
TIME_REVERSAL_SPECS = {
    'cheetah': TimeReversalSpec(17, 6, 8, 17, ('run', 'bidirectional')),
    'acrobot': TimeReversalSpec(6, 1, 4, 6, ('swingup',)),
    # Single pole: [cart position, cos(theta), sin(theta), cart/pole velocities].
    'cartpole': TimeReversalSpec(5, 1, 3, 5, ('swingup',)),
    'walker': TimeReversalSpec(24, 6, 15, 24, ('stand', 'walk', 'run')),
    'hopper': TimeReversalSpec(15, 4, 6, 13, ('stand', 'hop')),
}


def get_time_reversal_spec(domain_name):
    """Return a domain's explicit mapping rather than guessing from its shape."""
    if domain_name not in TIME_REVERSAL_SPECS:
        raise ValueError('No TRSPPO time-reversal mapping for domain {!r}. Supported domains: {}.'
                         .format(domain_name, ', '.join(TIME_REVERSAL_SPECS)))
    return TIME_REVERSAL_SPECS[domain_name]


def reverse_observation(obs, domain_name='cheetah'):
    """Negate generalized velocities, preserving configuration and sensor features.

    Preserving hopper's touch readings is an approximation: the simulator's
    dissipative contact forces can change when velocities are reversed.
    """
    spec = get_time_reversal_spec(domain_name)
    if obs.ndim == 0 or obs.shape[-1] != spec.obs_dim:
        raise ValueError('{} time reversal requires {}-dimensional state observations.'
                         .format(domain_name, spec.obs_dim))
    reversed_obs = obs.clone()
    velocities = slice(spec.velocity_start, spec.velocity_stop)
    reversed_obs[..., velocities] = -reversed_obs[..., velocities]
    return reversed_obs


def reverse_action(action, domain_name='cheetah'):
    """Motor force/torque controls are even under ideal mechanical time reversal."""
    spec = get_time_reversal_spec(domain_name)
    if action.ndim == 0 or action.shape[-1] != spec.action_dim:
        raise ValueError('{} time reversal requires {}-dimensional motor controls.'
                         .format(domain_name, spec.action_dim))
    return action.clone()


class TRSPPORolloutBuffer(RolloutBuffer):
    """Store real transition endpoints alongside the standard PPO rollout."""

    def __init__(self, obs_shape, action_shape, capacity):
        super().__init__(obs_shape, action_shape, capacity)
        self.next_obs = np.empty((capacity, *obs_shape), dtype=np.float32)

    def add(self, obs, raw_action, log_prob, reward, value, next_value,
            discount, done, *, next_obs):
        index = self.size
        super().add(obs, raw_action, log_prob, reward, value, next_value, discount, done)
        self.next_obs[index] = next_obs

    def batch(self, discount, gae_lambda, device):
        batch = super().batch(discount, gae_lambda, device)
        batch['next_obs'] = torch.as_tensor(self.next_obs[:self.size], device=device)
        # Conservatively omit both terminal and timeout transitions from TR loss.
        # They still participate in the ordinary PPO/GAE update.
        batch['tr_valid'] = torch.as_tensor(self.dones[:self.size] == 0, device=device)
        return batch

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        names = ('obs', 'next_obs', 'raw_actions', 'log_probs', 'rewards',
                 'values', 'next_values', 'discounts', 'dones')
        np.savez_compressed(os.path.join(directory, 'rollout.npz'),
                            **{name: getattr(self, name)[:self.size] for name in names})


class TRSPPOAgent(PPOAgent):
    """PPO with a detached-reference actor loss and an ordinary value critic."""

    checkpoint_prefix = 'trsppo'

    def __init__(self, obs_shape, action_shape, device, tr_policy_coef=0.1,
                 domain_name='cheetah', **kwargs):
        spec = get_time_reversal_spec(domain_name)
        if tuple(obs_shape) != (spec.obs_dim,) or tuple(action_shape) != (spec.action_dim,):
            raise ValueError('TRSPPO {} requires state observations ({},) and actions ({},).'
                             .format(domain_name, spec.obs_dim, spec.action_dim))
        if not math.isfinite(tr_policy_coef) or tr_policy_coef < 0:
            raise ValueError('tr_policy_coef must be finite and nonnegative.')
        super().__init__(obs_shape, action_shape, device, **kwargs)
        self.tr_policy_coef = tr_policy_coef
        self.domain_name = domain_name

    def actor_symmetry_loss(self, obs, next_obs, valid):
        """Mean squared error across valid transitions and all action components.

        The reference is evaluated by the current actor for each minibatch, with
        no gradient through that branch. Only the reversed-next-state prediction
        receives this loss's gradient; neither critic nor action std is involved.
        """
        if not valid.any():
            return obs.new_zeros(())
        # Select before evaluating the networks so masked endpoints cannot affect
        # the loss even if they contain invalid or automatically reset observations.
        with torch.no_grad():
            target = reverse_action(self.actor(obs[valid]).mean.tanh(), self.domain_name)
        prediction = self.actor(reverse_observation(next_obs[valid], self.domain_name)).mean.tanh()
        return F.mse_loss(prediction, target)

    def _auxiliary_actor_loss(self, batch, indices):
        obs = batch['obs'][indices]
        next_obs = batch['next_obs'][indices]
        valid = batch['tr_valid'][indices]
        if self.tr_policy_coef == 0:
            # Keep a diagnostic symmetry error without changing PPO gradients or
            # consuming random numbers when running the zero-coefficient ablation.
            with torch.no_grad():
                symmetry_loss = self.actor_symmetry_loss(obs, next_obs, valid)
            weighted_loss = obs.new_zeros(())
        else:
            symmetry_loss = self.actor_symmetry_loss(obs, next_obs, valid)
            weighted_loss = self.tr_policy_coef * symmetry_loss
        return weighted_loss, {
            'tr_policy_loss': symmetry_loss.item(),
            'tr_policy_loss_weighted': weighted_loss.item(),
            'tr_valid_pair_fraction': valid.float().mean().item(),
        }
