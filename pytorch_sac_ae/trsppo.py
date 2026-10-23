"""Time-reversal consistency for PPO on supported state observations.

For a real transition (o, a, r, next_o), regularize the deterministic action at
T(next_o) toward the detached current-policy action at o. This is an inductive
bias, not a guarantee that the environment dynamics are reversible.
An optional critic penalty compares V(T(next_o)) with detached V(o), using the
same valid transition pairs as the actor.
"""

from collections import namedtuple
import math
import os

import numpy as np
import torch
from torch.nn import functional as F

from ppo import PPOAgent, RolloutBuffer


# Offsets follow dmc2gym's flattened observation order. Separate velocity ranges
# preserve interleaved actuator/sensor features as well as trailing touch readings.
TimeReversalSpec = namedtuple('TimeReversalSpec',
                             ('obs_dim', 'action_dim', 'velocity_ranges', 'tasks'))
TIME_REVERSAL_SPECS = {
    'cheetah': TimeReversalSpec(17, 6, ((8, 17),), ('run', 'bidirectional')),
    'acrobot': TimeReversalSpec(6, 1, ((4, 6),), ('swingup',)),
    # Single pole: [cart position, cos(theta), sin(theta), cart/pole velocities].
    'cartpole': TimeReversalSpec(5, 1, ((3, 5),), ('swingup',)),
    # Joint angles, target displacement, then the two joint velocities.
    'reacher': TimeReversalSpec(6, 2, ((4, 6),), ('hard',)),
    'walker': TimeReversalSpec(24, 6, ((15, 24),), ('stand', 'walk', 'run')),
    'hopper': TimeReversalSpec(15, 4, ((6, 13),), ('stand', 'hop')),
    'finger': TimeReversalSpec(9, 2, ((4, 7),), ('spin',)),
}
# These domains require an explicit task because their dimensions vary by task.
TASK_TIME_REVERSAL_SPECS = {
    'pendulum': {
        'two_poles': TimeReversalSpec(6, 2, ((4, 6),), ('two_poles',)),
        'three_poles': TimeReversalSpec(9, 3, ((6, 9),), ('three_poles',)),
    },
    'quadruped': {
        # Joint velocities, torso linear velocity, and gyro angular velocity.
        # Keep activations, accelerometer and force/torque readings unchanged.
        'walk': TimeReversalSpec(78, 12, ((16, 32), (44, 47), (51, 54)), ('walk',)),
    },
}


def get_time_reversal_spec(domain_name, task_name=None):
    """Select an explicit mapping, requiring a task when dimensions vary."""
    if domain_name in TASK_TIME_REVERSAL_SPECS:
        task_specs = TASK_TIME_REVERSAL_SPECS[domain_name]
        if task_name not in task_specs:
            raise ValueError('TRSPPO {} requires task_name to be one of: {}.'
                             .format(domain_name, ', '.join(task_specs)))
        return task_specs[task_name]
    if domain_name not in TIME_REVERSAL_SPECS:
        domains = sorted(set(TIME_REVERSAL_SPECS) | set(TASK_TIME_REVERSAL_SPECS))
        raise ValueError('No TRSPPO time-reversal mapping for domain {!r}. Supported domains: {}.'
                         .format(domain_name, ', '.join(domains)))
    spec = TIME_REVERSAL_SPECS[domain_name]
    if task_name is not None and task_name not in spec.tasks:
        raise ValueError('TRSPPO supports these {} tasks: {}.'
                         .format(domain_name, ', '.join(spec.tasks)))
    return spec


def reverse_observation(obs, domain_name='cheetah', task_name=None):
    """Negate velocity features, preserving configuration and actuator states.

    Preserving hopper/finger touch readings and quadruped accelerometer/force
    readings is an approximation: dissipative dynamics can change these sensors
    when velocities are reversed. Quadruped's gyro is a velocity and is negated.
    """
    spec = get_time_reversal_spec(domain_name, task_name)
    if obs.ndim == 0 or obs.shape[-1] != spec.obs_dim:
        raise ValueError('{} time reversal requires {}-dimensional state observations.'
                         .format(domain_name, spec.obs_dim))
    reversed_obs = obs.clone()
    for start, stop in spec.velocity_ranges:
        reversed_obs[..., start:stop] = -reversed_obs[..., start:stop]
    return reversed_obs


def reverse_action(action, domain_name='cheetah', task_name=None):
    """Preserve action signs and ordering.

    Force/torque controls are even under ideal mechanical time reversal.
    For quadruped's filtered position targets, identity is only a regularization
    assumption; it does not reverse the actuator filter dynamics.
    """
    spec = get_time_reversal_spec(domain_name, task_name)
    if action.ndim == 0 or action.shape[-1] != spec.action_dim:
        raise ValueError('{} time reversal requires {}-dimensional controls.'
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
    """PPO with detached-reference actor and optional critic consistency losses."""

    checkpoint_prefix = 'trsppo'

    def __init__(self, obs_shape, action_shape, device, tr_policy_coef=0.1,
                 domain_name='cheetah', task_name=None, tr_value_coef=0.0, **kwargs):
        spec = get_time_reversal_spec(domain_name, task_name)
        if tuple(obs_shape) != (spec.obs_dim,) or tuple(action_shape) != (spec.action_dim,):
            environment = domain_name if task_name is None else domain_name + '/' + task_name
            raise ValueError('TRSPPO {} requires state observations ({},) and actions ({},).'
                             .format(environment, spec.obs_dim, spec.action_dim))
        for name, coefficient in (('tr_policy_coef', tr_policy_coef), ('tr_value_coef', tr_value_coef)):
            if not math.isfinite(coefficient) or coefficient < 0:
                raise ValueError('{} must be finite and nonnegative.'.format(name))
        super().__init__(obs_shape, action_shape, device, **kwargs)
        self.tr_policy_coef = tr_policy_coef
        self.tr_value_coef = tr_value_coef
        self.domain_name = domain_name
        self.task_name = task_name

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
            target = reverse_action(self.actor(obs[valid]).mean.tanh(),
                                    self.domain_name, self.task_name)
        reversed_obs = reverse_observation(next_obs[valid], self.domain_name, self.task_name)
        prediction = self.actor(reversed_obs).mean.tanh()
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

    def critic_symmetry_loss(self, next_obs, reference_values, valid):
        """Match reversed-next-state values to detached current-state predictions.

        Reuse the current critic predictions from the ordinary value loss and
        apply the same valid-transition mask as the actor before evaluation.
        """
        if not valid.any():
            return next_obs.new_zeros(())
        reversed_obs = reverse_observation(next_obs[valid], self.domain_name, self.task_name)
        reversed_values = self.critic(reversed_obs).squeeze(-1)
        return F.mse_loss(reversed_values, reference_values.detach()[valid])

    def _auxiliary_critic_loss(self, batch, indices, predicted_values):
        if self.tr_value_coef == 0:
            # Skip even diagnostic evaluations to preserve the baseline's cost.
            return predicted_values.new_zeros(()), {
                'tr_value_loss': 0.0,
                'tr_value_loss_weighted': 0.0,
            }
        symmetry_loss = self.critic_symmetry_loss(
            batch['next_obs'][indices], predicted_values, batch['tr_valid'][indices])
        weighted_loss = self.tr_value_coef * symmetry_loss
        return weighted_loss, {
            'tr_value_loss': symmetry_loss.item(),
            'tr_value_loss_weighted': weighted_loss.item(),
        }
