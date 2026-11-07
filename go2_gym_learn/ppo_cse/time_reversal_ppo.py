"""Opt-in temporal time-reversal PPO, leaving the original PPO implementation intact.

Adapted from symm_rl_isaaclab's TimeReversalPPO, with the WTW adaptation network,
privileged critic, 58D/70D frames, and two previous-action slots.
"""

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .ppo import PPO, PPO_Args
from .time_reversal import TemporalPairs, observation_layout


@dataclass(frozen=True)
class TimeReversalConfig:
    actor_loss_coef: float = 0.1
    value_loss_coef: float = 0.0
    warmup_iterations: int = 0
    min_abs_command_velocity: float = 0.0
    clip_actions: float = 10.0
    command_scale: float = 2.0

    def __post_init__(self):
        for name in ("actor_loss_coef", "value_loss_coef", "min_abs_command_velocity"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError("%s must be finite and nonnegative" % name)
        if not isinstance(self.warmup_iterations, int) or self.warmup_iterations < 0:
            raise ValueError("warmup_iterations must be a nonnegative integer")
        for name in ("clip_actions", "command_scale"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError("%s must be finite and positive" % name)


class TimeReversalPPO(PPO):
    def __init__(self, actor_critic, device="cpu", config=None):
        super().__init__(actor_critic, device=device)
        self.config = config if config is not None else TimeReversalConfig()
        self.update_count = 0
        self.last_tr_metrics = {}

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, privileged_obs_shape,
                     obs_history_shape, action_shape):
        if (len(actor_obs_shape) != 1 or actor_obs_shape[0] not in (58, 70) or tuple(privileged_obs_shape) != (2,)
                or tuple(action_shape) != (12,) or len(obs_history_shape) != 1
                or obs_history_shape[0] <= 0 or obs_history_shape[0] % actor_obs_shape[0]):
            raise ValueError("TimeReversalPPO requires 58D/70D frames, 2 privileged values, and 12 actions")
        self.frame_dim = actor_obs_shape[0]
        observation_layout(self.frame_dim)
        if num_transitions_per_env < obs_history_shape[0] // self.frame_dim + 2:
            raise ValueError("TimeReversalPPO needs num_steps_per_env >= history_length + 2")
        super().init_storage(num_envs, num_transitions_per_env, actor_obs_shape, privileged_obs_shape,
                             obs_history_shape, action_shape)

    def _action_mean(self, history):
        # Keep this differentiable and avoid act_student's CPU/NumPy side effect.
        # Recompute friction/restitution estimates from each history without
        # transforming their signs; actor TR gradients also train the adapter.
        latent = self.actor_critic.adaptation_module(history)
        return self.actor_critic.actor_body(torch.cat((history, latent), dim=-1))

    def _clamp_std(self):
        with torch.no_grad():
            if not torch.isfinite(self.actor_critic.std).all():
                raise FloatingPointError("TR-PPO action standard deviation became nonfinite")
            self.actor_critic.std.clamp_(min=1.e-6)

    def act(self, obs, privileged_obs, obs_history):
        self._clamp_std()
        return super().act(obs, privileged_obs, obs_history)

    def consistency_losses(self, batch):
        """Regularize the full student, with gradients only through the reversed branch."""
        reversed_history, reference_history, reversed_privileged, reference_privileged = batch
        zero = reversed_history.new_zeros(())
        if reversed_history.shape[0] == 0:
            return zero, zero
        actor_loss, value_loss = zero, zero
        if self.config.actor_loss_coef:
            prediction = self._action_mean(reversed_history)
            with torch.no_grad():
                target = self._action_mean(reference_history)
            actor_loss = F.mse_loss(prediction, target)
        if self.config.value_loss_coef:
            prediction = self.actor_critic.evaluate(reversed_history, reversed_privileged)
            with torch.no_grad():
                target = self.actor_critic.evaluate(reference_history, reference_privileged)
            value_loss = F.mse_loss(prediction, target)
        return actor_loss, value_loss

    def training_state(self):
        return {"update_count": self.update_count, "config": asdict(self.config),
                "learning_rate": self.learning_rate, "optimizer": self.optimizer.state_dict(),
                "adaptation_optimizer": self.adaptation_module_optimizer.state_dict()}

    def load_training_state(self, state):
        if state["config"] != asdict(self.config):
            raise ValueError("TR-PPO checkpoint configuration differs from this run")
        self.update_count = int(state["update_count"])
        self.learning_rate = state["learning_rate"]
        self.optimizer.load_state_dict(state["optimizer"])
        self.adaptation_module_optimizer.load_state_dict(state["adaptation_optimizer"])

    def update(self):
        self._clamp_std()
        active = (self.update_count >= self.config.warmup_iterations
                  and (self.config.actor_loss_coef > 0 or self.config.value_loss_coef > 0))
        self.last_tr_metrics = {"tr_actor_loss": 0.0, "tr_value_loss": 0.0,
                                "tr_valid_pair_fraction": 0.0, "tr_active": float(active)}
        if not active:
            result = super().update()
            self._clamp_std()
            self.update_count += 1
            return result

        pairs = TemporalPairs(self.storage, self.config.clip_actions,
                              self.config.min_abs_command_velocity, self.config.command_scale,
                              frame_dim=self.frame_dim)
        self.last_tr_metrics["tr_valid_pair_fraction"] = pairs.mask.float().mean().item()
        pair_batches = pairs.batches(PPO_Args.num_mini_batches, PPO_Args.num_learning_epochs)
        generator = self.storage.mini_batch_generator(PPO_Args.num_mini_batches, PPO_Args.num_learning_epochs)
        mean_value_loss = mean_surrogate_loss = 0.0
        mean_adaptation_loss = mean_adaptation_test_loss = 0.0
        # This update deliberately mirrors PPO.update locally. The original PPO
        # loss, optimizer, and adaptation code are not modified by opting in.
        for batch in generator:
            (_, _, privileged, history, actions, old_values, advantages, returns,
             old_log_prob, old_mu, old_sigma, masks, _) = batch
            self.actor_critic.act(history, masks=masks)
            log_prob = self.actor_critic.get_actions_log_prob(actions)
            values = self.actor_critic.evaluate(history, privileged, masks=masks)
            mu = self.actor_critic.action_mean
            sigma = self.actor_critic.action_std
            entropy = self.actor_critic.entropy

            if PPO_Args.desired_kl is not None and PPO_Args.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(torch.log(sigma / old_sigma + 1.e-5)
                                   + (old_sigma.square() + (old_mu - mu).square())
                                   / (2.0 * sigma.square()) - 0.5, dim=-1).mean()
                    if kl > PPO_Args.desired_kl * 2.0:
                        self.learning_rate = max(1.e-5, self.learning_rate / 1.5)
                    elif 0.0 < kl < PPO_Args.desired_kl / 2.0:
                        self.learning_rate = min(1.e-2, self.learning_rate * 1.5)
                    for group in self.optimizer.param_groups:
                        group["lr"] = self.learning_rate

            ratio = torch.exp(log_prob - old_log_prob.squeeze(-1))
            advantage = advantages.squeeze(-1)
            surrogate_loss = torch.max(-advantage * ratio, -advantage * ratio.clamp(
                1.0 - PPO_Args.clip_param, 1.0 + PPO_Args.clip_param)).mean()
            if PPO_Args.use_clipped_value_loss:
                clipped = old_values + (values - old_values).clamp(-PPO_Args.clip_param, PPO_Args.clip_param)
                value_loss = torch.max((values - returns).square(), (clipped - returns).square()).mean()
            else:
                value_loss = F.mse_loss(values, returns)

            actor_tr, value_tr = self.consistency_losses(next(pair_batches))
            loss = (surrogate_loss + PPO_Args.value_loss_coef * value_loss
                    - PPO_Args.entropy_coef * entropy.mean()
                    + self.config.actor_loss_coef * actor_tr + self.config.value_loss_coef * value_tr)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), PPO_Args.max_grad_norm)
            self.optimizer.step()
            self._clamp_std()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            self.last_tr_metrics["tr_actor_loss"] += actor_tr.item()
            self.last_tr_metrics["tr_value_loss"] += value_tr.item()

            num_train = int(privileged.shape[0] // 5 * 4)
            if num_train == 0:
                raise ValueError("TR-PPO adaptation mini-batches need at least 5 samples")
            for _ in range(PPO_Args.num_adaptation_module_substeps):
                prediction = self.actor_critic.adaptation_module(history)
                selection = torch.arange(prediction.shape[1], device=prediction.device)
                if PPO_Args.selective_adaptation_module_loss:
                    selection = 0
                adaptation_loss = F.mse_loss(prediction[:num_train, selection], privileged[:num_train, selection])
                test_loss = F.mse_loss(prediction[num_train:, selection], privileged[num_train:, selection])
                self.adaptation_module_optimizer.zero_grad()
                adaptation_loss.backward()
                self.adaptation_module_optimizer.step()
                mean_adaptation_loss += adaptation_loss.item()
                mean_adaptation_test_loss += test_loss.item()

        updates = PPO_Args.num_learning_epochs * PPO_Args.num_mini_batches
        adaptation_updates = updates * PPO_Args.num_adaptation_module_substeps
        self.last_tr_metrics["tr_actor_loss"] /= updates
        self.last_tr_metrics["tr_value_loss"] /= updates
        self.storage.clear()
        self.update_count += 1
        return (mean_value_loss / updates, mean_surrogate_loss / updates,
                mean_adaptation_loss / adaptation_updates, 0.0, 0.0,
                mean_adaptation_test_loss / adaptation_updates, 0.0, 0.0)
