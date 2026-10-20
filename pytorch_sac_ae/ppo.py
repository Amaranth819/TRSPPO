"""PPO with separate MLP policy/value networks for continuous state control.

References: https://arxiv.org/abs/1707.06347 (PPO),
https://arxiv.org/abs/1506.02438 (generalized advantage estimation).
"""

import math
import os

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal


def compute_gae(rewards, values, next_values, discounts, dones,
                discount=0.99, gae_lambda=0.95):
    """Bootstrap at timeouts, but never carry advantages across a reset.

    `discounts` are environment discounts: zero at true termination, one at
    time limits. `dones` marks either kind of episode boundary. The final
    next_value also bootstraps a rollout that ends in the middle of an episode.
    """
    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + discount * discounts[t] * next_values[t] - values[t]
        gae = delta + discount * gae_lambda * discounts[t] * (1 - dones[t]) * gae
        advantages[t] = gae
    return advantages, advantages + values


class RolloutBuffer:
    """A single on-policy rollout, discarded after each PPO update."""

    def __init__(self, obs_shape, action_shape, capacity):
        self.capacity = capacity
        self.obs = np.empty((capacity, *obs_shape), dtype=np.float32)
        # Keep pre-tanh actions to avoid unstable inverse-tanh near +/-1.
        self.raw_actions = np.empty((capacity, *action_shape), dtype=np.float32)
        for name in ('log_probs', 'rewards', 'values', 'next_values', 'discounts', 'dones'):
            setattr(self, name, np.empty(capacity, dtype=np.float32))
        self.size = 0

    def add(self, obs, raw_action, log_prob, reward, value, next_value, discount, done):
        if self.size >= self.capacity:
            raise ValueError('Rollout is full; update and clear it before adding data.')
        i = self.size
        self.obs[i] = obs
        self.raw_actions[i] = raw_action
        self.log_probs[i] = log_prob
        self.rewards[i] = reward
        self.values[i] = value
        self.next_values[i] = next_value
        self.discounts[i] = discount
        self.dones[i] = done
        self.size += 1

    def batch(self, discount, gae_lambda, device):
        if not self.size:
            raise ValueError('Cannot update PPO from an empty rollout.')
        n = self.size
        advantages, returns = compute_gae(
            self.rewards[:n], self.values[:n], self.next_values[:n],
            self.discounts[:n], self.dones[:n], discount, gae_lambda)
        arrays = dict(obs=self.obs[:n], raw_actions=self.raw_actions[:n],
                      log_probs=self.log_probs[:n], advantages=advantages,
                      returns=returns, values=self.values[:n])
        return {k: torch.as_tensor(v, device=device) for k, v in arrays.items()}

    def clear(self):
        self.size = 0

    def save(self, directory):
        """Keep the latest rollout for inspection, not off-policy replay."""
        os.makedirs(directory, exist_ok=True)
        names = ('obs', 'raw_actions', 'log_probs', 'rewards', 'values',
                 'next_values', 'discounts', 'dones')
        np.savez_compressed(os.path.join(directory, 'rollout.npz'),
                            **{k: getattr(self, k)[:self.size] for k in names})


def _mlp(input_dim, hidden_dim, output_dim, output_gain):
    net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(),
                        nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                        nn.Linear(hidden_dim, output_dim))
    for layer in net:
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, gain=math.sqrt(2))
            nn.init.zeros_(layer.bias)
    nn.init.orthogonal_(net[-1].weight, gain=output_gain)
    return net


class GaussianActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, log_std_min, log_std_max):
        super().__init__()
        self.mean = _mlp(obs_dim, hidden_dim, action_dim, output_gain=0.01)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, obs):
        mean = self.mean(obs)
        log_std = self.log_std.clamp(self.log_std_min, self.log_std_max)
        return Normal(mean, log_std.exp().expand_as(mean))

    @staticmethod
    def log_prob(distribution, raw_action):
        # Stable log(1 - tanh(x)^2), including saturated actions.
        log_jacobian = 2 * (math.log(2) - raw_action - F.softplus(-2 * raw_action))
        return (distribution.log_prob(raw_action) - log_jacobian).sum(-1)


class PPOAgent:
    checkpoint_prefix = 'ppo'

    def __init__(self, obs_shape, action_shape, device, hidden_dim=256,
                 discount=0.99, actor_lr=1e-3, critic_lr=1e-3,
                 actor_beta=0.9, critic_beta=0.9,
                 actor_log_std_min=-10, actor_log_std_max=2,
                 batch_size=256, gae_lambda=0.95, clip_ratio=0.2,
                 update_epochs=10, entropy_coef=0.0, value_coef=0.5,
                 max_grad_norm=0.5, target_kl=0.03):
        if len(obs_shape) != 1 or len(action_shape) != 1:
            raise ValueError('PPO requires flat state observations and continuous actions.')
        self.device = torch.device(device)
        self.discount = discount
        self.batch_size = batch_size
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.update_epochs = update_epochs
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.actor = GaussianActor(obs_shape[0], action_shape[0], hidden_dim,
                                   actor_log_std_min, actor_log_std_max).to(self.device)
        self.critic = _mlp(obs_shape[0], hidden_dim, 1, output_gain=1).to(self.device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999), eps=1e-5)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999), eps=1e-5)
        self.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def _obs_tensor(self, obs):
        return torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

    @torch.no_grad()
    def act(self, obs):
        obs = self._obs_tensor(obs)
        distribution = self.actor(obs)
        raw_action = distribution.sample()
        log_prob = self.actor.log_prob(distribution, raw_action)
        value = self.critic(obs).squeeze(-1)
        return (raw_action.tanh()[0].cpu().numpy(), raw_action[0].cpu().numpy(),
                log_prob.item(), value.item())

    def sample_action(self, obs):
        return self.act(obs)[0]

    @torch.no_grad()
    def select_action(self, obs):
        return self.actor(self._obs_tensor(obs)).mean.tanh()[0].cpu().numpy()

    @torch.no_grad()
    def value(self, obs):
        return self.critic(self._obs_tensor(obs)).item()

    def _auxiliary_actor_loss(self, batch, indices):
        """Hook for actor regularizers; vanilla PPO has no additional loss."""
        return batch['obs'].new_zeros(()), {}

    def update(self, rollout, logger=None, step=0):
        batch = rollout.batch(self.discount, self.gae_lambda, self.device)
        advantages = batch['advantages']
        if rollout.size > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        totals = dict(actor_loss=0.0, critic_loss=0.0, entropy=0.0,
                      approx_kl=0.0, clip_fraction=0.0)
        samples = updates = 0
        early_stop = False
        for _ in range(self.update_epochs):
            indices = torch.randperm(rollout.size, device=self.device)
            for start in range(0, rollout.size, self.batch_size):
                idx = indices[start:start + self.batch_size]
                distribution = self.actor(batch['obs'][idx])
                log_prob = self.actor.log_prob(distribution, batch['raw_actions'][idx])
                log_ratio = log_prob - batch['log_probs'][idx]
                ratio = log_ratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                if self.target_kl > 0 and approx_kl.item() > self.target_kl:
                    early_stop = True
                    break
                surrogate = ratio * advantages[idx]
                clipped = ratio.clamp(1 - self.clip_ratio, 1 + self.clip_ratio) * advantages[idx]
                actor_loss = -torch.min(surrogate, clipped).mean()
                # Monte Carlo entropy of the squashed policy, with a pathwise
                # gradient when entropy regularization is enabled.
                entropy = -self.actor.log_prob(distribution, distribution.rsample()).mean()
                predicted_values = self.critic(batch['obs'][idx]).squeeze(-1)
                critic_loss = F.mse_loss(predicted_values, batch['returns'][idx])

                auxiliary_loss, auxiliary_metrics = self._auxiliary_actor_loss(batch, idx)
                actor_total_loss = actor_loss - self.entropy_coef * entropy + auxiliary_loss
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_total_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                self.critic_optimizer.zero_grad(set_to_none=True)
                (self.value_coef * critic_loss).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                count = len(idx)
                metrics = dict(actor_loss=actor_loss.item(), critic_loss=critic_loss.item(),
                               entropy=entropy.item(), approx_kl=approx_kl.item(),
                               clip_fraction=((ratio - 1).abs() > self.clip_ratio).float().mean().item())
                if auxiliary_metrics:
                    metrics.update(auxiliary_metrics)
                    metrics['actor_total_loss'] = actor_total_loss.item()
                for key, value in metrics.items():
                    totals[key] = totals.get(key, 0.0) + count * value
                samples += count
                updates += 1
            if early_stop:
                break
        metrics = {key: value / max(samples, 1) for key, value in totals.items()}
        with torch.no_grad():
            predictions = self.critic(batch['obs']).squeeze(-1)
            variance = batch['returns'].var(unbiased=False)
            explained_variance = (1 - (batch['returns'] - predictions).var(unbiased=False) /
                                  variance).item() if variance.item() > 1e-8 else 0.0
        metrics.update(explained_variance=explained_variance, updates=updates,
                       early_stop=float(early_stop), rollout_steps=rollout.size,
                       batch_reward=float(rollout.rewards[:rollout.size].mean()))
        if logger is not None:
            for key, value in metrics.items():
                logger.log('train/' + key, value, step)
        return metrics

    def save(self, model_dir, step):
        os.makedirs(model_dir, exist_ok=True)
        torch.save(dict(step=step, actor=self.actor.state_dict(), critic=self.critic.state_dict(),
                        actor_optimizer=self.actor_optimizer.state_dict(),
                        critic_optimizer=self.critic_optimizer.state_dict()),
                   os.path.join(model_dir, '{}_{}.pt'.format(self.checkpoint_prefix, step)))

    def load(self, model_dir, step):
        checkpoint = torch.load(os.path.join(model_dir, '{}_{}.pt'.format(self.checkpoint_prefix, step)),
                                map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
