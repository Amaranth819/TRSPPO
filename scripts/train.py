OBSERVATION_HISTORY_LENGTH = 20
ROBOT_HISTORY_LENGTHS = {robot: OBSERVATION_HISTORY_LENGTH for robot in ("go2", "x1", "x1_go2bend")}


def parse_args(argv=None):
    # Keep --help usable without importing Isaac Gym, torch, or params_proto.
    import argparse
    import math

    def positive_int(value):
        value = int(value)
        if value <= 0:
            raise argparse.ArgumentTypeError("must be positive")
        return value

    def nonnegative_float(value):
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise argparse.ArgumentTypeError("must be finite and nonnegative")
        return value

    parser = argparse.ArgumentParser(description="Train Go2 or X1 MLP policies with PPO or time-reversal PPO.")
    parser.add_argument("--robot", choices=tuple(ROBOT_HISTORY_LENGTHS), default="go2",
                        help="Select the robot model and training environment.")
    parser.add_argument("--algorithm", choices=("ppo", "trsppo"), default="ppo",
                        help="Use trsppo for actor time-reversal consistency.")
    parser.add_argument("--headless", action="store_true", help="Disable the simulator viewer.")
    parser.add_argument("--num-envs", type=positive_int, default=None)
    parser.add_argument("--num-steps-per-env", type=positive_int, default=None,
                        help="Rollout length: default 24 for both algorithms and all robots.")
    parser.add_argument("--iterations", dest="num_learning_iterations", type=positive_int, default=10000)
    parser.add_argument("--tensorboard", action="store_true",
                        help="Write TensorBoard scalars alongside the run's metrics.pkl.")
    parser.add_argument("--tensorboard-log-dir", default=None,
                        help="Override the TensorBoard directory (also enables logging).")
    parser.add_argument("--tr-actor-loss-coef", type=nonnegative_float, default=0.1)
    parser.add_argument("--tr-value-loss-coef", type=nonnegative_float, default=0.0,
                        help="Optional value consistency loss; disabled by default.")
    parser.add_argument("--tr-warmup-iterations", type=int, default=0)
    parser.add_argument("--tr-min-abs-command-velocity", type=nonnegative_float, default=0.0)
    args = parser.parse_args(argv)
    if args.tr_warmup_iterations < 0:
        parser.error("--tr-warmup-iterations must be nonnegative")
    history_length = ROBOT_HISTORY_LENGTHS[args.robot]
    if (args.algorithm == "trsppo" and args.num_steps_per_env is not None
            and args.num_steps_per_env < history_length + 2):
        parser.error(f"trsppo requires --num-steps-per-env >= {history_length + 2} "
                     f"for the {history_length}-frame history")
    return args


def train_go2(headless=True, algorithm="ppo", num_steps_per_env=None, num_envs=None,
              num_learning_iterations=10000, tr_actor_loss_coef=0.1, tr_value_loss_coef=0.0,
              tr_warmup_iterations=0, tr_min_abs_command_velocity=0.0,
              tensorboard=False, tensorboard_log_dir=None, robot="go2"):

    import isaacgym
    assert isaacgym
    import torch

    from go2_gym.envs.base.legged_robot_config import Cfg
    from go2_gym.envs.go2.go2_config import config_go2
    from go2_gym.envs.x1.x1_config import config_x1, config_x1_go2bend
    from go2_gym.envs.go2.velocity_tracking import VelocityTrackingEasyEnv

    from ml_logger import logger

    if tensorboard or tensorboard_log_dir is not None:
        # Check the optional dependency before creating the simulator.
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:
            raise RuntimeError(
                "TensorBoard logging requires the optional dependency. Install it with: "
                "python -m pip install -e '.[tensorboard]'"
            ) from exc

    from go2_gym_learn.ppo_cse import Runner
    from go2_gym.envs.wrappers.history_wrapper import HistoryWrapper
    from go2_gym_learn.ppo_cse.actor_critic import AC_Args
    from go2_gym_learn.ppo_cse.ppo import PPO_Args
    from go2_gym_learn.ppo_cse import RunnerArgs
    from go2_gym_learn.ppo_cse.algorithms import normalize_algorithm

    algorithm = normalize_algorithm(algorithm)
    if robot not in ROBOT_HISTORY_LENGTHS:
        raise ValueError(f"Unsupported robot: {robot}")
    RunnerArgs.algorithm_class_name = algorithm
    default_steps = max(24, ROBOT_HISTORY_LENGTHS[robot] + 2) if algorithm == "trsppo" else 24
    RunnerArgs.num_steps_per_env = num_steps_per_env if num_steps_per_env is not None else default_steps

    {"go2": config_go2, "x1": config_x1, "x1_go2bend": config_x1_go2bend}[robot](Cfg)
    # Share the full command vector and four-gait curriculum across all robots.
    Cfg.commands.cmd_indices = None
    Cfg.commands.gait_category_names = None

    Cfg.commands.num_lin_vel_bins = 30
    Cfg.commands.num_ang_vel_bins = 30
    Cfg.curriculum_thresholds.tracking_ang_vel = 0.7
    Cfg.curriculum_thresholds.tracking_lin_vel = 0.8
    Cfg.curriculum_thresholds.tracking_contacts_shaped_vel = 0.90
    Cfg.curriculum_thresholds.tracking_contacts_shaped_force = 0.90

    Cfg.commands.distributional_commands = True

    Cfg.domain_rand.lag_timesteps = 6
    Cfg.domain_rand.randomize_lag_timesteps = True
    Cfg.control.control_type = "actuator_net" if robot == "go2" else "P"

    Cfg.domain_rand.randomize_rigids_after_start = False
    Cfg.env.priv_observe_motion = False
    Cfg.env.priv_observe_gravity_transformed_motion = False
    Cfg.domain_rand.randomize_friction_indep = False
    Cfg.env.priv_observe_friction_indep = False
    Cfg.domain_rand.randomize_friction = True
    Cfg.env.priv_observe_friction = True
    Cfg.domain_rand.friction_range = [0.1, 3.0]
    Cfg.domain_rand.randomize_restitution = True
    Cfg.env.priv_observe_restitution = True
    Cfg.domain_rand.restitution_range = [0.0, 0.4]
    Cfg.domain_rand.randomize_base_mass = True
    Cfg.env.priv_observe_base_mass = False
    Cfg.domain_rand.added_mass_range = [-1.0, 3.0]
    Cfg.domain_rand.randomize_gravity = True
    Cfg.domain_rand.gravity_range = [-1.0, 1.0]
    Cfg.domain_rand.gravity_rand_interval_s = 8.0
    Cfg.domain_rand.gravity_impulse_duration = 0.99
    Cfg.env.priv_observe_gravity = False
    Cfg.domain_rand.randomize_com_displacement = False
    Cfg.domain_rand.com_displacement_range = [-0.15, 0.15]
    Cfg.env.priv_observe_com_displacement = False
    Cfg.domain_rand.randomize_ground_friction = True
    Cfg.env.priv_observe_ground_friction = False
    Cfg.env.priv_observe_ground_friction_per_foot = False
    Cfg.domain_rand.ground_friction_range = [0.0, 0.0]
    Cfg.domain_rand.randomize_motor_strength = True
    Cfg.domain_rand.motor_strength_range = [0.9, 1.1]
    Cfg.env.priv_observe_motor_strength = False
    Cfg.domain_rand.randomize_motor_offset = True
    Cfg.domain_rand.motor_offset_range = [-0.02, 0.02]
    Cfg.env.priv_observe_motor_offset = False
    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.randomize_Kp_factor = False
    Cfg.env.priv_observe_Kp_factor = False
    Cfg.domain_rand.randomize_Kd_factor = False
    Cfg.env.priv_observe_Kd_factor = False
    Cfg.env.priv_observe_body_velocity = False
    Cfg.env.priv_observe_body_height = False
    Cfg.env.priv_observe_desired_contact_states = False
    Cfg.env.priv_observe_contact_forces = False
    Cfg.env.priv_observe_foot_displacement = False
    Cfg.env.priv_observe_gravity_transformed_foot_displacement = False

    Cfg.env.num_privileged_obs = 2
    Cfg.env.num_observation_history = OBSERVATION_HISTORY_LENGTH
    Cfg.reward_scales.feet_contact_forces = 0.0

    Cfg.domain_rand.rand_interval_s = 4
    Cfg.commands.num_commands = 15
    Cfg.env.observe_two_prev_actions = True
    Cfg.env.observe_yaw = False
    Cfg.env.num_observations = 70
    Cfg.env.num_scalar_observations = 70
    Cfg.env.observe_gait_commands = True
    Cfg.env.observe_timing_parameter = False
    Cfg.env.observe_clock_inputs = True

    Cfg.domain_rand.tile_height_range = [-0.0, 0.0]
    Cfg.domain_rand.tile_height_curriculum = False
    Cfg.domain_rand.tile_height_update_interval = 1000000
    Cfg.domain_rand.tile_height_curriculum_step = 0.01
    Cfg.terrain.border_size = 0.0
    # Default value = "trimesh"
    Cfg.terrain.mesh_type = "trimesh"  # "heightfield" # none, plane, heightfield or trimesh
    Cfg.terrain.num_cols = 30
    Cfg.terrain.num_rows = 30
    Cfg.terrain.terrain_width = 1.0 # 5.0
    Cfg.terrain.terrain_length = 1.0 # 5.0
    Cfg.terrain.x_init_range = 0.2
    Cfg.terrain.y_init_range = 0.2
    Cfg.terrain.teleport_thresh = 0.3
    Cfg.terrain.teleport_robots = False
    Cfg.terrain.center_robots = True
    Cfg.terrain.center_span = 4
    Cfg.terrain.horizontal_scale = 0.10
    Cfg.rewards.use_terminal_foot_height = False
    Cfg.rewards.use_terminal_body_height = True
    Cfg.rewards.terminal_body_height = 0.05
    Cfg.rewards.use_terminal_roll_pitch = True
    Cfg.rewards.terminal_body_ori = 1.6

    Cfg.commands.resampling_time = 10

    Cfg.reward_scales.feet_slip = -0.04
    Cfg.reward_scales.action_smoothness_1 = -0.1
    Cfg.reward_scales.action_smoothness_2 = -0.1
    Cfg.reward_scales.dof_vel = -1e-4
    Cfg.reward_scales.dof_pos = -0.0
    Cfg.reward_scales.jump = 10.0
    Cfg.reward_scales.base_height = 0.0
    Cfg.rewards.base_height_target = 0.30
    Cfg.reward_scales.estimation_bonus = 0.0
    Cfg.reward_scales.raibert_heuristic = -10.0
    Cfg.reward_scales.feet_impact_vel = -0.0
    Cfg.reward_scales.feet_clearance = -0.0
    Cfg.reward_scales.feet_clearance_cmd = -0.0
    Cfg.reward_scales.feet_clearance_cmd_linear = -30.0
    Cfg.reward_scales.orientation = 0.0
    Cfg.reward_scales.orientation_control = -5.0
    Cfg.reward_scales.tracking_stance_width = -0.0
    Cfg.reward_scales.tracking_stance_length = -0.0
    Cfg.reward_scales.lin_vel_z = -0.02
    Cfg.reward_scales.ang_vel_xy = -0.001
    Cfg.reward_scales.feet_air_time = 0.0
    Cfg.reward_scales.hop_symmetry = 0.0
    Cfg.rewards.kappa_gait_probs = 0.07
    Cfg.rewards.gait_force_sigma = 100.
    Cfg.rewards.gait_vel_sigma = 10.
    Cfg.reward_scales.tracking_contacts_shaped_force = 4.0
    Cfg.reward_scales.tracking_contacts_shaped_vel = 4.0
    Cfg.reward_scales.collision = -5.0

    Cfg.rewards.reward_container_name = "CoRLRewards"
    Cfg.rewards.only_positive_rewards = False
    Cfg.rewards.only_positive_rewards_ji22_style = True
    Cfg.rewards.sigma_rew_neg = 0.02



    Cfg.commands.lin_vel_x = [-1.0, 1.0]
    Cfg.commands.lin_vel_y = [-0.6, 0.6]
    Cfg.commands.ang_vel_yaw = [-1.0, 1.0]
    Cfg.commands.body_height_cmd = [-0.25, 0.15]
    Cfg.commands.gait_frequency_cmd_range = [2.0, 4.0]
    Cfg.commands.gait_phase_cmd_range = [0.0, 1.0]
    Cfg.commands.gait_offset_cmd_range = [0.0, 1.0]
    Cfg.commands.gait_bound_cmd_range = [0.0, 1.0]
    Cfg.commands.gait_duration_cmd_range = [0.5, 0.5]
    Cfg.commands.footswing_height_range = [0.03, 0.35]
    Cfg.commands.body_pitch_range = [-0.4, 0.4]
    Cfg.commands.body_roll_range = [-0.0, 0.0]
    Cfg.commands.stance_width_range = [0.10, 0.45]
    Cfg.commands.stance_length_range = [0.35, 0.45]

    Cfg.commands.limit_vel_x = [-5.0, 5.0]
    Cfg.commands.limit_vel_y = [-0.6, 0.6]
    Cfg.commands.limit_vel_yaw = [-5.0, 5.0]
    Cfg.commands.limit_body_height = [-0.25, 0.15]
    Cfg.commands.limit_gait_frequency = [2.0, 4.0]
    Cfg.commands.limit_gait_phase = [0.0, 1.0]
    Cfg.commands.limit_gait_offset = [0.0, 1.0]
    Cfg.commands.limit_gait_bound = [0.0, 1.0]
    Cfg.commands.limit_gait_duration = [0.5, 0.5]
    Cfg.commands.limit_footswing_height = [0.03, 0.35]
    Cfg.commands.limit_body_pitch = [-0.4, 0.4]
    Cfg.commands.limit_body_roll = [-0.0, 0.0]
    Cfg.commands.limit_stance_width = [0.10, 0.45]
    Cfg.commands.limit_stance_length = [0.35, 0.45]

    Cfg.commands.num_bins_vel_x = 21
    Cfg.commands.num_bins_vel_y = 1
    Cfg.commands.num_bins_vel_yaw = 21
    Cfg.commands.num_bins_body_height = 1
    Cfg.commands.num_bins_gait_frequency = 1
    Cfg.commands.num_bins_gait_phase = 1
    Cfg.commands.num_bins_gait_offset = 1
    Cfg.commands.num_bins_gait_bound = 1
    Cfg.commands.num_bins_gait_duration = 1
    Cfg.commands.num_bins_footswing_height = 1
    Cfg.commands.num_bins_body_roll = 1
    Cfg.commands.num_bins_body_pitch = 1
    Cfg.commands.num_bins_stance_width = 1

    Cfg.normalization.friction_range = [0, 1]
    Cfg.normalization.ground_friction_range = [0, 1]
    Cfg.terrain.yaw_init_range = 3.14
    Cfg.normalization.clip_actions = 10.0

    Cfg.commands.exclusive_phase_offset = False
    Cfg.commands.pacing_offset = False
    Cfg.commands.binary_phases = True
    Cfg.commands.gaitwise_curricula = True

    time_reversal_config = None
    if algorithm == "trsppo":
        from go2_gym_learn.ppo_cse.time_reversal import validate_environment
        from go2_gym_learn.ppo_cse.time_reversal_ppo import TimeReversalConfig
        from go2_gym.envs.wrappers.time_reversal_history_wrapper import TimeReversalHistoryWrapper

        validate_environment(Cfg)
        if RunnerArgs.num_steps_per_env < Cfg.env.num_observation_history + 2:
            raise ValueError("TimeReversalPPO needs num_steps_per_env >= history_length + 2")
        time_reversal_config = TimeReversalConfig(
            actor_loss_coef=tr_actor_loss_coef, value_loss_coef=tr_value_loss_coef,
            warmup_iterations=tr_warmup_iterations,
            min_abs_command_velocity=tr_min_abs_command_velocity,
            clip_actions=Cfg.normalization.clip_actions, command_scale=Cfg.obs_scales.lin_vel,
        )
        HistoryWrapper = TimeReversalHistoryWrapper
    if num_envs is not None:
        Cfg.env.num_envs = num_envs
    env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=headless, cfg=Cfg)

    # log the experiment parameters
    logger.log_params(robot=robot, algorithm=algorithm,
                      AC_Args=vars(AC_Args), PPO_Args=vars(PPO_Args), RunnerArgs=vars(RunnerArgs),
                      Cfg=vars(Cfg))
    if time_reversal_config is not None:
        logger.log_params(TimeReversalConfig=vars(time_reversal_config))

    env = HistoryWrapper(env)
    gpu_id = 0
    runner = Runner(env, device=f"cuda:{gpu_id}", time_reversal_config=time_reversal_config)
    writer = None
    if tensorboard or tensorboard_log_dir is not None:
        from pathlib import Path

        log_dir = (Path(tensorboard_log_dir) if tensorboard_log_dir is not None
                   else Path(logger.root) / logger.prefix / "tensorboard")
        log_dir = log_dir.expanduser().resolve()
        writer = SummaryWriter(log_dir=str(log_dir), flush_secs=10)
        print(f"TensorBoard logs: {log_dir}")
    try:
        runner.learn(num_learning_iterations=num_learning_iterations, init_at_random_ep_len=True,
                     eval_freq=100, summary_writer=writer)
    finally:
        if writer is not None:
            writer.close()


if __name__ == '__main__':
    args = parse_args()
    # The legacy PPO_Args class registers its own CLI. Our arguments are already consumed.
    import sys
    sys.argv = [sys.argv[0]]
    from pathlib import Path
    from ml_logger import logger
    from go2_gym import MINI_GYM_ROOT_DIR

    is_trsppo = args.algorithm == "trsppo"
    logger.configure(logger.utcnow(f'gait-conditioned-agility/{args.robot}/{args.algorithm}/%Y-%m-%d_%H-%M-%S.%f'),
                     root=Path(f"{MINI_GYM_ROOT_DIR}/runs").resolve(), )
    logger.log_text("""
                charts: 
                - yKey: train/episode/rew_total/mean
                  xKey: iterations
                - yKey: train/episode/rew_tracking_lin_vel/mean
                  xKey: iterations
                - yKey: train/episode/rew_tracking_contacts_shaped_force/mean
                  xKey: iterations
                - yKey: train/episode/rew_action_smoothness_1/mean
                  xKey: iterations
                - yKey: train/episode/rew_action_smoothness_2/mean
                  xKey: iterations
                - yKey: train/episode/rew_tracking_contacts_shaped_vel/mean
                  xKey: iterations
                - yKey: train/episode/rew_orientation_control/mean
                  xKey: iterations
                - yKey: train/episode/rew_dof_pos/mean
                  xKey: iterations
                - yKey: train/episode/command_area_trot/mean
                  xKey: iterations
                - yKey: train/episode/max_terrain_height/mean
                  xKey: iterations
                - type: video
                  glob: "videos/*.mp4"
                - yKey: adaptation_loss/mean
                  xKey: iterations
                """, filename=".charts.yml", dedent=True)

    if is_trsppo:
        logger.log_text("""
                - yKey: tr_actor_loss/mean
                  xKey: iterations
                - yKey: tr_value_loss/mean
                  xKey: iterations
                - yKey: tr_valid_pair_fraction/mean
                  xKey: iterations
                - yKey: tr_active/mean
                  xKey: iterations
                """, filename=".charts.yml", dedent=True, overwrite=False)
    train_go2(**vars(args))
