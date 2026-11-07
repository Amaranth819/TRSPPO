"""Evaluation-only hooks; training environments and wrappers are unchanged."""

from isaacgym.torch_utils import quat_rotate_inverse
import numpy as np
import torch

from go2_gym.envs.go2.velocity_tracking import VelocityTrackingEasyEnv


def leg_name(name):
    name = name.lower()
    for leg, long_name in (("fl", "front_left"), ("fr", "front_right"),
                           ("rl", "rear_left"), ("rr", "rear_right")):
        if name.startswith(leg + "_") or long_name in name:
            return leg
    raise ValueError(f"Unknown leg in simulator body/joint name {name}")


def joint_order(names):
    lookup = {}
    for index, name in enumerate(names):
        role = "hip" if "hip" in name or "abad" in name else "thigh" if "thigh" in name else "calf"
        lookup[(leg_name(name), role)] = index
    keys = [(leg, role) for leg in ("fl", "fr", "rl", "rr") for role in ("hip", "thigh", "calf")]
    if len(names) != 12 or set(lookup) != set(keys):
        raise ValueError("Expected hip/thigh/calf joints for FL, FR, RL, RR")
    return [lookup[key] for key in keys]


class EvaluationEnv(VelocityTrackingEasyEnv):
    def __init__(self, command, **kwargs):
        self.fixed_command = list(command)
        self.evaluation_sample = None
        super().__init__(**kwargs)
        body_names = self.gym.get_asset_rigid_body_names(self.robot_asset)
        feet = [body_names[index] for index in self.feet_indices.cpu().tolist()]
        if [leg_name(name) for name in feet] != ["fl", "fr", "rl", "rr"]:
            raise ValueError(f"Saved gait clocks require FL, FR, RL, RR foot order, found {feet}")
        self.metric_joint_order = joint_order(self.dof_names)
        self.action_clamped = torch.zeros_like(self.actions, dtype=torch.bool)

    def create_sim(self):
        if self.headless:
            self.graphics_device_id = -1
        return super().create_sim()

    def _init_command_distribution(self, env_ids):
        # Fixed evaluation commands need no training curriculum or sampled bins.
        self.category_names = ["evaluation"]
        self.curricula = []
        self.env_command_bins = np.zeros(self.num_envs, dtype=np.int64)
        self.env_command_categories = np.zeros(self.num_envs, dtype=np.int64)

    def set_command(self, command):
        self.fixed_command = list(command)
        self._resample_commands(torch.arange(self.num_envs, device=self.device))

    def _resample_commands(self, env_ids):
        if len(env_ids):
            self.commands[env_ids] = torch.tensor(self.fixed_command, device=self.device)
            for values in getattr(self, "command_sums", {}).values():
                values[env_ids] = 0.

    def reset(self):
        # Initialize observations at the reset state without an unrecorded physics
        # step. This also keeps the requested commands/clocks in the FIRST input.
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.actions.zero_()
        self.action_clamped.zero_()
        for name in ("joint_pos_err_last", "joint_pos_err_last_last", "joint_vel_last", "joint_vel_last_last"):
            if hasattr(self, name):
                getattr(self, name).zero_()
        self.base_pos[:] = self.root_states[:, :3]
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.gait_indices[:] = -self.dt * self.commands[:, 4]
        self._step_contact_targets()
        self.compute_observations()
        self.obs_buf.clamp_(-self.cfg.normalization.clip_observations, self.cfg.normalization.clip_observations)
        self.evaluation_sample = None
        return self.obs_buf

    def step(self, actions):
        self.action_clamped = actions.abs() > self.cfg.normalization.clip_actions
        return super().step(actions)

    def check_termination(self):
        super().check_termination()
        # This hook runs after physics/clock updates and BEFORE reset_idx mutates
        # terminal positions, velocities, actions, or gait phase.
        q = self.base_quat
        heading = torch.atan2(2 * (q[:, 3] * q[:, 2] + q[:, 0] * q[:, 1]),
                              1 - 2 * (q[:, 1].square() + q[:, 2].square()))
        order = self.metric_joint_order
        samples = {
            "done": self.reset_buf.bool(),
            "position": self.root_states[:, :3],
            "heading": heading,
            "velocity": torch.stack((self.base_lin_vel[:, 0], self.base_lin_vel[:, 1], self.base_ang_vel[:, 2]), dim=-1),
            "torque": self.torques[:, order],
            "power": (self.torques * self.dof_vel)[:, order],
            "foot_force": self.contact_forces[:, self.feet_indices, 2],
            "phase": self.gait_indices,
            "action_clamped": self.action_clamped[:, order],
        }
        self.evaluation_sample = {key: value.detach().cpu().numpy().copy() for key, value in samples.items()}

    def metric_metadata(self):
        masses = [sum(prop.mass for prop in self.gym.get_actor_rigid_body_properties(env, actor))
                  for env, actor in zip(self.envs, self.actor_handles)]
        return {"mass_kg": masses,
                "effort_limits_nm": self.torque_limits[self.metric_joint_order].cpu().tolist(),
                "joint_names": [self.dof_names[index] for index in self.metric_joint_order],
                "foot_order": ["FL", "FR", "RL", "RR"]}
