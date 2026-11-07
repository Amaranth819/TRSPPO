"""Evaluate saved WTW policies with the Isaac Lab locomotion metric formulas."""

import argparse
import contextlib
import copy
from datetime import datetime, timezone
import hashlib
import io
import math
from pathlib import Path
import pickle
import random
import sys
import traceback
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ROBOTS = ("go2", "x1", "x1_go2bend")
ALGORITHMS = ("ppo", "trsppo")
GAITS = ("pronk", "trot", "pace", "bound")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=("all",) + ROBOTS, default="all")
    parser.add_argument("--algorithm", choices=("all",) + ALGORITHMS, default="all")
    parser.add_argument("--run", default="latest", help="Run name or directory; latest selects the newest saved policy.")
    parser.add_argument("--checkpoint", default="last", help="last, or an iteration number (e.g. 5999).")
    parser.add_argument("--log-root", type=Path, default=ROOT / "runs/gait-conditioned-agility")
    parser.add_argument("--gait", choices=("all",) + GAITS, default="all")
    parser.add_argument("--velocities", nargs="+", type=float, default=[-2., -1., -.5, .5, 1., 2.])
    parser.add_argument("--num-envs", type=int, default=100, help="Independent parallel trials per gait/speed.")
    parser.add_argument("--settle-steps", type=int, default=250)
    parser.add_argument("--measure-steps", type=int, default=1000)
    parser.add_argument("--frequency", type=float, default=3., help="Shared gait frequency in Hz; duty factor is 0.5.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--viewer", action="store_true", help="Show the simulator; otherwise run headless.")
    parser.add_argument("--output", type=Path, help="New output directory; existing directories are never overwritten.")
    parser.add_argument("--no-save-rollouts", action="store_true", help="Keep metrics/metadata without raw NPZ traces.")
    parser.add_argument("--dry-run", action="store_true", help="Check selected runs/configs/weights on CPU without simulation.")
    args = parser.parse_args(argv)
    if args.num_envs < 1 or args.settle_steps < 0 or args.measure_steps < 2:
        parser.error("num-envs must be positive, settle-steps nonnegative, and measure-steps at least 2")
    if not math.isfinite(args.frequency) or args.frequency <= 0:
        parser.error("frequency must be finite and positive")
    if not args.velocities or any(not math.isfinite(v) or v == 0 for v in args.velocities):
        parser.error("velocities must be finite and nonzero (relative tracking/progress metrics require motion)")
    if len(set(args.velocities)) != len(args.velocities):
        parser.error("velocities must not contain duplicates")
    if args.run != "latest" and (args.robot == "all" or args.algorithm == "all"):
        parser.error("an explicit --run requires one --robot and one --algorithm")
    if args.checkpoint != "last" and not args.checkpoint.isdecimal():
        parser.error("checkpoint must be last or a nonnegative iteration number")
    if args.seed < 0 or args.seed >= 2 ** 32:
        parser.error("seed must be in [0, 2**32)")
    return args


class CPUUnpickler(pickle.Unpickler):
    """Training parameter pickles may contain CUDA tensors in derived settings."""
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            import torch
            return lambda data: torch.load(io.BytesIO(data), map_location="cpu")
        return super().find_class(module, name)


def load_parameters(path):
    with Path(path).open("rb") as stream:
        return CPUUnpickler(stream).load()


def checkpoint_filename(value):
    return "ac_weights_last.pt" if value == "last" else "ac_weights_%06d.pt" % int(value)


def resolve_runs(args):
    """Resolve the full requested matrix before spending time on simulation."""
    selected, missing = [], []
    robots = ROBOTS if args.robot == "all" else (args.robot,)
    algorithms = ALGORITHMS if args.algorithm == "all" else (args.algorithm,)
    filename = checkpoint_filename(args.checkpoint)
    for robot in robots:
        for algorithm in algorithms:
            parent = args.log_root / robot / algorithm
            if args.run == "latest":
                candidates = [p.parent for p in parent.glob("*/parameters.pkl")
                              if (p.parent / "checkpoints" / filename).is_file()]
                run = max(candidates, key=lambda p: (p / "parameters.pkl").stat().st_mtime_ns) if candidates else None
            else:
                given = Path(args.run).expanduser()
                run = given if given.is_dir() else parent / given
            if run is None or not (run / "parameters.pkl").is_file() or not (run / "checkpoints" / filename).is_file():
                missing.append(f"{robot}/{algorithm}: parameters.pkl and checkpoints/{filename} under {run or parent}")
            else:
                selected.append(dict(robot=robot, algorithm=algorithm, run=run.resolve(),
                                     checkpoint=(run / "checkpoints" / filename).resolve()))
    if missing:
        raise ValueError("Missing requested trained policies:\n  " + "\n  ".join(missing)
                         + "\nSelect available runs with --robot/--algorithm. Deprecated logs are not selected automatically.")
    return selected


def validate_parameters(parameters, task, gaits):
    cfg = parameters["Cfg"]
    saved_robot = parameters.get("robot")
    if saved_robot is not None and saved_robot != task["robot"]:
        raise ValueError(f"Run was trained for {saved_robot}, not {task['robot']}")
    saved_algorithm = parameters.get("algorithm")
    if saved_algorithm is not None and saved_algorithm != task["algorithm"]:
        raise ValueError(f"Run algorithm is {saved_algorithm}, not {task['algorithm']}")
    model = cfg["asset"]["file"].replace("\\", "/")
    expected_asset = "/go2/" if task["robot"] == "go2" else "/dobot_quad_v2/"
    if expected_asset not in model:
        raise ValueError(f"Robot {task['robot']} does not match saved asset {model}")
    if task["robot"] != "go2":
        joints = cfg["init_state"]["default_joint_angles"]
        bent = joints["joint_rear_left_thigh_pitch"] * joints["joint_front_left_thigh_pitch"] > 0
        if bent != (task["robot"] == "x1_go2bend"):
            raise ValueError("Requested X1 variant does not match the checkpoint's saved reference pose")
    dims = cfg["env"]["num_observations"]
    if dims not in (58, 70) or cfg["env"]["num_actions"] != 12:
        raise ValueError("Evaluation supports the saved 58D/70D, 12-action policies")
    trained = cfg["commands"].get("gait_category_names")
    if dims == 58:
        trained = trained or ["trot"]
    if trained is not None and set(gaits).difference(trained):
        raise ValueError(f"Checkpoint trained on {trained}; cannot evaluate requested gaits {gaits}. "
                         "Use a new four-gait checkpoint, or select --gait trot for legacy runs.")
    dt = float(cfg["sim"]["dt"]) * int(cfg["control"]["decimation"])
    return dt


def make_config(parameters, args):
    # Saved settings keep the exact robot, controller, joint pose, history, and lag.
    sections = copy.deepcopy(parameters["Cfg"])
    cfg = SimpleNamespace(**{name: SimpleNamespace(**values) for name, values in sections.items()
                             if isinstance(values, dict)})
    cfg.commands.cmd_indices = getattr(cfg.commands, "cmd_indices", None)
    cfg.commands.gait_category_names = getattr(cfg.commands, "gait_category_names", None)
    cfg.commands.pacing_offset = getattr(cfg.commands, "pacing_offset", False)
    cfg.commands.command_curriculum = False
    cfg.commands.heading_command = False
    cfg.commands.resampling_time = 1.e6
    cfg.env.num_envs = args.num_envs
    cfg.env.record_video = False
    cfg.env.num_recording_envs = 0
    dt = cfg.sim.dt * cfg.control.decimation
    cfg.env.episode_length_s = (args.settle_steps + args.measure_steps + 10) * dt
    cfg.terrain.mesh_type = "plane"
    cfg.terrain.curriculum = False
    cfg.terrain.teleport_robots = False
    cfg.terrain.measure_heights = False
    cfg.terrain.yaw_init_range = 0.
    cfg.terrain.x_init_range = cfg.terrain.y_init_range = 0.
    cfg.terrain.static_friction = cfg.terrain.dynamic_friction = 1.
    cfg.terrain.restitution = 0.
    cfg.noise.add_noise = False
    for name in vars(cfg.domain_rand):
        if name.startswith("randomize_") and name != "randomize_lag_timesteps":
            setattr(cfg.domain_rand, name, False)
    cfg.domain_rand.push_robots = False
    # In WTW randomize_lag_timesteps enables a FIXED delay buffer. Preserve it.
    # Each robot has its own collision group. With self-collisions disabled and
    # no other objects, net foot forces are exclusively ground-contact forces.
    cfg.asset.self_collisions = 1
    cfg.sim.use_gpu_pipeline = args.device.startswith("cuda")
    cfg.sim.physx["use_gpu"] = args.device.startswith("cuda")
    return cfg


def load_policy(parameters, checkpoint, device):
    import torch
    from go2_gym_learn.ppo_cse.actor_critic import AC_Args, ActorCritic
    for key, value in parameters["AC_Args"].items():
        setattr(AC_Args, key, value)
    cfg = parameters["Cfg"]["env"]
    width = cfg["num_observations"] * cfg["num_observation_history"]
    with contextlib.redirect_stdout(io.StringIO()):
        actor = ActorCritic(cfg["num_observations"], cfg["num_privileged_obs"], width, cfg["num_actions"])
    actor.load_state_dict(torch.load(str(checkpoint), map_location="cpu"), strict=True)
    actor = actor.to(device).eval()

    def policy(observations):
        history = observations["obs_history"].to(device)
        latent = actor.adaptation_module(history)
        return actor.actor_body(torch.cat((history, latent), dim=-1))

    with torch.no_grad():
        probe = policy({"obs_history": torch.zeros(1, width, device=device)})
    if probe.shape != (1, 12) or not torch.isfinite(probe).all():
        raise ValueError("Checkpoint failed the deterministic 12-action inference check")
    return policy


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_task(task, parameters, args, output, summaries):
    import numpy as np
    import torch
    from go2_gym.envs.wrappers.history_wrapper import HistoryWrapper
    from go2_gym.envs.wrappers.time_reversal_history_wrapper import TimeReversalHistoryWrapper
    from go2_gym_learn.eval_metrics.environment import EvaluationEnv
    from go2_gym_learn.eval_metrics.protocol import command_vector, trial_metrics, summarize_trials, write_csv, write_json

    cfg = make_config(parameters, args)
    policy = load_policy(parameters, task["checkpoint"], args.device)
    gaits = GAITS if args.gait == "all" else (args.gait,)
    seed_everything(args.seed)
    base = EvaluationEnv(command_vector(gaits[0], args.velocities[0], args.frequency, cfg.commands.pacing_offset),
                         sim_device=args.device, headless=not args.viewer, cfg=cfg)
    wrapper = TimeReversalHistoryWrapper if task["algorithm"] == "trsppo" else HistoryWrapper
    env = wrapper(base)
    task_output = output / task["robot"] / task["algorithm"]
    try:
        physical = base.metric_metadata()
        write_json(task_output / "policy.json", {
            "run": str(task["run"]), "checkpoint": str(task["checkpoint"]),
            "checkpoint_sha256": sha256(task["checkpoint"]), "robot": task["robot"],
            "algorithm": task["algorithm"], "history_wrapper": wrapper.__name__,
            "observation_dim": cfg.env.num_observations, "history_length": cfg.env.num_observation_history,
            "control": vars(cfg.control), "fixed_delay_enabled": cfg.domain_rand.randomize_lag_timesteps,
            "lag_physics_steps": cfg.domain_rand.lag_timesteps, "step_dt": base.dt, **physical})
        for gait in gaits:
            for velocity_index, velocity in enumerate(args.velocities):
                # Same case seed across algorithms/robots; separate trial states within each case.
                case_seed = (args.seed + GAITS.index(gait) * len(args.velocities) + velocity_index) % (2 ** 32)
                seed_everything(case_seed)
                base.set_command(command_vector(gait, velocity, args.frequency, cfg.commands.pacing_offset))
                env.reset()
                obs = env.get_observations()
                traces = {}
                active = np.ones(args.num_envs, dtype=bool)
                steps = args.settle_steps + args.measure_steps
                print(f"[evaluate] {task['robot']}/{task['algorithm']} gait={gait} vx={velocity:g}, trials={args.num_envs}", flush=True)
                with torch.no_grad():
                    for _ in range(steps):
                        actions = policy(obs)
                        if not torch.isfinite(actions).all():
                            raise FloatingPointError("Policy produced nonfinite actions")
                        obs, _, _, _ = env.step(actions)
                        for name, value in base.evaluation_sample.items():
                            traces.setdefault(name, []).append(value)
                        active &= ~base.evaluation_sample["done"]
                        if not active.any():
                            break
                arrays = {key: np.stack(value) for key, value in traces.items()}
                rows = [trial_metrics(arrays, trial, gait, velocity, base.dt, args.settle_steps,
                                      physical["mass_kg"][trial], physical["effort_limits_nm"])
                        for trial in range(args.num_envs)]
                summary = dict(robot=task["robot"], algorithm=task["algorithm"], run=str(task["run"]),
                               checkpoint=str(task["checkpoint"]), gait=gait, velocity_mps=velocity,
                               seed=case_seed, **summarize_trials(rows))
                case_output = task_output / f"velocity_{velocity:+}" / gait
                write_json(case_output / "trial_metrics.json", rows)
                write_csv(case_output / "trial_metrics.csv", rows)
                write_json(case_output / "summary.json", summary)
                if not args.no_save_rollouts:
                    np.savez_compressed(str(case_output / "rollout.npz"), **arrays)
                summaries.append(summary)
                write_csv(output / "summary.csv", summaries)
                print(f"  measured={summary['measured_trials']}/{args.num_envs}, "
                      f"terminated={summary['terminated_trials']}, joint success={summary['joint_success_rate']:.1%}", flush=True)
    finally:
        base.close()


def main(argv=None):
    args = parse_args(argv)
    try:
        tasks = resolve_runs(args)
        # Isaac Gym must be imported before torch, including torch used to read metadata.
        if not args.dry_run:
            import isaacgym
            assert isaacgym
        import torch
        gaits = GAITS if args.gait == "all" else (args.gait,)
        prepared = []
        for task in tasks:
            parameters = load_parameters(task["run"] / "parameters.pkl")
            dt = validate_parameters(parameters, task, gaits)
            if args.frequency * dt >= .5:
                raise ValueError("Gait frequency must be below half the policy sampling frequency")
            load_policy(parameters, task["checkpoint"], "cpu")
            prepared.append((task, parameters))
            print(f"[plan] {task['robot']}/{task['algorithm']}: {task['checkpoint']}\n"
                  f"  gaits={','.join(gaits)} speeds={args.velocities}, {args.num_envs} trials each; "
                  f"settle={args.settle_steps * dt:g}s measure={args.measure_steps * dt:g}s", flush=True)
        if args.dry_run:
            print(f"Validated {len(tasks)} policies, {len(tasks) * len(gaits) * len(args.velocities)} gait/speed cases. No simulation started.")
            return 0
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; activate the training environment on a machine with a working NVIDIA driver")
        from go2_gym_learn.eval_metrics import reference_metrics
        from go2_gym_learn.eval_metrics.protocol import write_json
        output = args.output or ROOT / "evaluations" / datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S.%f")
        output.mkdir(parents=True, exist_ok=False)
        manifest = {
            "status": "running", "protocol": "wtw_four_gaits_isaaclab_metrics_v1",
            "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "metrics_sha256": sha256(reference_metrics.__file__),
            "evaluation_config": reference_metrics.evaluation_config(),
            "terrain": "flat plane", "self_collisions": False,
            "ground_force_source": "net foot forces, ground-only by collision isolation",
            "domain_randomization": False, "observation_noise": False,
            "initial_state_randomization": "saved WTW reset distribution, fixed case seeds, initial yaw zero",
            "failure_policy": "first episode only, terminal sample included, failed trials retained",
            "sampling": "final physics substep at policy frequency, before automatic reset",
            "phase_conversion": "reference common phase = unwrapped WTW phase + 0.5",
        }
        write_json(output / "manifest.json", manifest)
        summaries = []
        try:
            for task, parameters in prepared:
                run_task(task, parameters, args, output, summaries)
        except Exception as error:
            manifest.update(status="failed", error=str(error), traceback=traceback.format_exc())
            write_json(output / "manifest.json", manifest)
            traceback.print_exc()
            raise
        manifest.update(status="complete", completed_cases=len(summaries))
        write_json(output / "manifest.json", manifest)
        print(f"Evaluation saved to {output.resolve()}")
        return 0
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"Evaluation error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
