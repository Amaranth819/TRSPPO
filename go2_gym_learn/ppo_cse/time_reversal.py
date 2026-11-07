"""Time reversal for the shared 70D and legacy 58D observation layouts.

Histories are frame-major, oldest first. Position actions are even under the
chosen reversal convention, but their temporal indices must be shifted.
"""

from dataclasses import dataclass

import torch


FRAME_DIM = 70
ACTION_DIM = 12


@dataclass(frozen=True)
class ObservationLayout:
    command_count: int

    @property
    def commands(self):
        return slice(3, 3 + self.command_count)

    @property
    def joint_velocities(self):
        return slice(15 + self.command_count, 27 + self.command_count)

    @property
    def previous_action(self):
        return slice(27 + self.command_count, 39 + self.command_count)

    @property
    def second_previous_action(self):
        return slice(39 + self.command_count, 51 + self.command_count)


def observation_layout(frame_dim):
    if frame_dim not in (58, 70):
        raise ValueError("TimeReversalPPO requires 58D or 70D observation frames")
    return ObservationLayout(command_count=frame_dim - 55)


def validate_environment(cfg):
    """Reject layouts/gaits whose reversal rules have not been implemented."""
    layout = observation_layout(cfg.env.num_observations)
    required = {
        "num_actions": ACTION_DIM,
        "num_privileged_obs": 2,
        "observe_command": True,
        "observe_two_prev_actions": True,
        "observe_gait_commands": True,
        "observe_clock_inputs": True,
        "observe_vel": False,
        "observe_only_ang_vel": False,
        "observe_only_lin_vel": False,
        "observe_yaw": False,
        "observe_contact_states": False,
        "observe_timing_parameter": False,
        "priv_observe_friction": True,
        "priv_observe_restitution": True,
    }
    for name, expected in required.items():
        if getattr(cfg.env, name) != expected:
            raise ValueError("TimeReversalPPO requires env.%s=%r" % (name, expected))
    if cfg.commands.num_commands != 15 or cfg.terrain.measure_heights:
        raise ValueError("TimeReversalPPO requires 15 commands and no height observations")
    indices = getattr(cfg.commands, "cmd_indices", None)
    observed = range(cfg.commands.num_commands) if indices is None else indices
    if tuple(observed) != tuple(range(layout.command_count)):
        raise ValueError("TimeReversalPPO requires commands in their original [vx, vy, yaw, ...] order")
    if cfg.obs_scales.gait_phase_cmd != 1.0:
        raise ValueError("TimeReversalPPO requires gait_phase_cmd observation scale 1.0")
    for name in ("gait_duration_cmd_range", "limit_gait_duration"):
        if tuple(getattr(cfg.commands, name)) != (0.5, 0.5):
            raise ValueError("TimeReversalPPO currently supports a fixed 0.5 gait duration")


def reverse_history(history, current_actions, next_actions, frame_dim=FRAME_DIM):
    """Reverse histories using outgoing actions at the newest frame and after it.

    At original state s_j the two action slots contain a_(j-1), a_(j-2).
    At its reversed counterpart they must contain a_j, a_(j+1). Inputs
    current_actions/next_actions must use the environment's observation clipping.
    The sine clocks are even for phase reflection phi'=(0.5-phi) mod 1.
    """
    layout = observation_layout(frame_dim)
    if history.ndim != 2 or history.shape[-1] == 0 or history.shape[-1] % frame_dim:
        raise ValueError(f"Expected flattened histories of complete {frame_dim}D frames")
    expected = (history.shape[0], ACTION_DIM)
    if tuple(current_actions.shape) != expected or tuple(next_actions.shape) != expected:
        raise ValueError("Current and next actions must have shape (batch, 12)")
    frames = history.reshape(history.shape[0], history.shape[1] // frame_dim, frame_dim)
    result = frames.flip(1).clone()
    result[..., 3:6].neg_()  # commanded linear and yaw velocities
    result[..., layout.joint_velocities].neg_()
    if layout.command_count == 15:
        result[..., 8:11] = torch.remainder(-result[..., 8:11], 1.0)

    # Outgoing action sequence, in original time: a_(t-H+1), ..., a_t, a_(t+1).
    outgoing = torch.cat((frames[:, 1:, layout.previous_action], current_actions[:, None], next_actions[:, None]), dim=1)
    result[..., layout.previous_action] = outgoing[:, :-1].flip(1)
    result[..., layout.second_previous_action] = outgoing[:, 1:].flip(1)
    return result.reshape_as(history)


class TemporalPairs:
    """Index rollout pairs without allocating another full rollout of histories."""

    def __init__(self, storage, clip_actions, min_abs_command_velocity=0.0, command_scale=2.0,
                 frame_dim=FRAME_DIM):
        self.storage = storage
        self.clip_actions = clip_actions
        self.frame_dim = frame_dim
        layout = observation_layout(frame_dim)
        histories = storage.observation_histories
        steps, envs, width = histories.shape
        if width == 0 or width % frame_dim:
            raise ValueError(f"Expected flattened histories of complete {frame_dim}D frames")
        self.history_length = width // frame_dim
        h = self.history_length
        if storage.step != steps:
            raise ValueError("Time-reversal pairing requires a complete rollout")
        if steps < h + 2:
            raise ValueError("TimeReversalPPO needs num_steps_per_env >= history_length + 2")
        self.num_envs = envs
        self.pair_count = (steps - h - 1) * envs
        frames = histories.view(steps, envs, h, frame_dim)
        reference = frames[:steps-h-1]
        current = frames[h:-1]
        # Zero padding marks incomplete histories; the TR-specific wrapper clears
        # padding on every reset, including resets before this rollout started.
        full = reference[:, :, 0].ne(0).any(-1) & current[:, :, 0].ne(0).any(-1)
        # Include done[t]: a_(t+1) must belong to the same episode as s_t.
        crosses_reset = storage.dones[:-1, :, 0].bool().unfold(0, h + 1, 1).any(-1)
        command = reference[:, :, -1, layout.commands]
        reference_constant = torch.isclose(reference[..., layout.commands], command[:, :, None], rtol=0, atol=1e-6).all(-1).all(-1)
        current_constant = torch.isclose(current[..., layout.commands], command[:, :, None], rtol=0, atol=1e-6).all(-1).all(-1)
        next_constant = torch.isclose(frames[h+1:, :, -1, layout.commands], command, rtol=0, atol=1e-6).all(-1)
        privileged_constant = torch.isclose(
            storage.privileged_observations[:steps-h-1], storage.privileged_observations[h:-1],
            rtol=0, atol=1e-6,
        ).all(-1)
        moving = command[..., 0].abs() / command_scale >= min_abs_command_velocity
        self.mask = (full & ~crosses_reset & reference_constant & current_constant & next_constant
                     & privileged_constant & moving).reshape(-1)

    def batches(self, num_mini_batches, num_epochs):
        if self.pair_count < num_mini_batches:
            raise ValueError("Fewer temporal pairs than PPO mini-batches")
        for _ in range(num_epochs):
            indices = torch.randperm(self.pair_count, device=self.mask.device)
            # Integer slicing covers the remainder and also works with older torch.
            for i in range(num_mini_batches):
                batch = indices[i*self.pair_count//num_mini_batches:(i+1)*self.pair_count//num_mini_batches]
                batch = batch[self.mask[batch]]
                ref_step = batch // self.num_envs
                env = batch % self.num_envs
                step = ref_step + self.history_length
                history = self.storage.observation_histories[step, env]
                actions = self.storage.actions[step, env].clamp(-self.clip_actions, self.clip_actions)
                next_actions = self.storage.actions[step + 1, env].clamp(-self.clip_actions, self.clip_actions)
                reversed_history = reverse_history(history, actions, next_actions, frame_dim=self.frame_dim)
                yield (reversed_history, self.storage.observation_histories[ref_step, env],
                       self.storage.privileged_observations[step, env],
                       self.storage.privileged_observations[ref_step, env])
