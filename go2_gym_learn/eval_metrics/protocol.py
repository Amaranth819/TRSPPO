"""Simulator-independent four-gait evaluation protocol and report generation."""

import csv
import json
import math
from pathlib import Path

import numpy as np

from . import reference_metrics as metrics


# Physical foot offsets in FL, FR, RL, RR order. WTW stance occupies [0, .5).
GAIT_OFFSETS = {
    "pronk": (0., 0., 0., 0.),
    "trot": (.5, 0., 0., .5),
    "pace": (.5, 0., .5, 0.),
    "bound": (.5, .5, 0., 0.),
}
GAIT_LIBRARY = [dict(name=name, family=name, index=i, phases=phases)
                for i, (name, phases) in enumerate(GAIT_OFFSETS.items())]


def gait_command(gait, pacing_offset=False):
    """Map physical gait names through the saved clock-generator convention."""
    commands = {"pronk": (0., 0., 0.), "trot": (.5, 0., 0.),
                "pace": (0., 0., .5), "bound": (0., .5, 0.)}
    phase, offset, bound = commands[gait]
    return (phase, bound, offset) if pacing_offset else (phase, offset, bound)


def command_vector(gait, velocity, frequency=3., pacing_offset=False):
    # vx, vy, yaw, height, frequency, phase/offset/bound, duration, swing height,
    # body pitch/roll, stance width/length, auxiliary reward coefficient.
    return [velocity, 0., 0., 0., frequency, *gait_command(gait, pacing_offset),
            .5, .08, 0., 0., .25, .40, 0.]


def measurement_bounds(common_phase, settle_steps):
    """Match the reference's complete-cycle trimming of the post-settle window."""
    start, stop = settle_steps, len(common_phase)
    first = math.ceil(float(common_phase[start]) - 1.e-8)
    last = math.floor(float(common_phase[-1]) + 1.e-8)
    cycles = max(0, last - first)
    if cycles:
        left = int(np.searchsorted(common_phase, first, side="left"))
        right = int(np.searchsorted(common_phase, last, side="left"))
        if start <= left < right <= stop:
            return left, right, cycles
    return start, stop, 0


def trial_metrics(data, trial, gait, velocity, dt, settle_steps, mass, effort_limits, config=None):
    """Analyze only a trial's first episode, including authentic terminal samples."""
    config = metrics.evaluation_config(config)
    done = np.asarray(data["done"][:, trial], dtype=bool)
    endings = np.flatnonzero(done)
    stop = int(endings[0]) + 1 if endings.size else len(done)
    terminated = bool(endings.size)
    row = {"trial": trial, "gait": gait, "velocity_mps": velocity,
           "terminated": terminated, "recorded_samples": stop,
           "termination_time_s": stop * dt if terminated else None,
           "status": "failed" if terminated else "valid", "reason": "",
           "velocity_only_success": False, "gait_only_success": False,
           "joint_success": False, "success": False}
    if stop <= settle_steps + 1:
        row.update(status="failed" if terminated else "short", measured_samples=0,
                   reason="Fewer than two post-settle samples in the first episode")
        return row
    arrays = {name: np.asarray(value[:stop, trial]) for name, value in data.items() if name != "done"}
    if any(not np.isfinite(value).all() for value in arrays.values()):
        row.update(status="invalid", measured_samples=0, reason="Nonfinite rollout samples")
        return row

    # The reference marks stance in [.5, 1); WTW marks stance in [0, .5).
    # A half-cycle common-phase shift preserves each foot's physical stance.
    phase = np.unwrap(arrays["phase"] * 2 * np.pi) / (2 * np.pi) + .5
    start, end, cycles = measurement_bounds(phase, settle_steps)
    selection = slice(start, end)
    commands = np.tile([velocity, 0., 0.], (stop, 1))
    positions = arrays["position"][selection, :2].copy()
    if start:
        positions[0] = arrays["position"][start - 1, :2]
    velocity_result = metrics.velocity_metrics(
        commands[selection], arrays["velocity"][selection], positions, dt,
        base_headings_rad=arrays["heading"][selection],
        desired_headings_rad=np.zeros(end - start), terminated=terminated,
        full_commands=commands, full_body_velocities=arrays["velocity"],
        measurement_start_index=start, config=config)
    origin = arrays["position"][start - 1 if start else 0, 0]
    progress = float(np.sign(velocity) * (arrays["position"][end - 1, 0] - origin))
    requested_distance = abs(velocity) * (end - start) * dt
    velocity_result.update(signed_directed_progress_m=progress,
                           progress_per_commanded_distance=progress / requested_distance,
                           lost_progress_m=max(0., requested_distance - progress))
    forces = np.maximum(arrays["foot_force"][selection], 0.)
    gait_result, contacts = metrics.gait_metrics(
        forces, phase[selection], GAIT_OFFSETS[gait], .5, dt, config,
        robot_mass_kg=float(mass), gait_library=GAIT_LIBRARY)
    gait_result["complete_cycles"] = cycles
    load_result = metrics.load_metrics(
        arrays["torque"][selection], arrays["power"][selection], effort_limits,
        forces, contacts, dt, complete_cycles=cycles, directed_progress_m=progress,
        action_clamped=arrays["action_clamped"][selection], config=config)
    planar_rmse = float(np.sqrt(np.mean(np.sum(
        (arrays["velocity"][selection, :2] - commands[selection, :2]) ** 2, axis=1))))
    row.update(measured_samples=end - start, measurement_start_s=start * dt,
               measurement_stop_s=end * dt, tracking_rmse_mps=planar_rmse,
               tracking_success=(planar_rmse <= .05 + .25 * abs(velocity)
                                 and velocity_result["yaw_rmse_radps"] <= config["tracking"]["yaw_rmse_limit_radps"]))
    for prefix, result in (("velocity", velocity_result), ("gait", gait_result), ("load", load_result)):
        row.update({prefix + "_" + key: value for key, value in result.items()})
    row.update(metrics.success_metrics(velocity_result, gait_result, planned_family=gait,
                                       terminated=terminated, config=config))
    return row


def summarize_trials(rows):
    """Keep every trial in success denominators; publish counts for metric means."""
    result = {"trials": len(rows), "terminated_trials": sum(row["terminated"] for row in rows),
              "measured_trials": sum(row.get("measured_samples", 0) > 0 for row in rows)}
    for name in ("velocity_only_success", "gait_only_success", "joint_success", "success"):
        result[name + "_rate"] = sum(bool(row.get(name, False)) for row in rows) / len(rows)
    for name in sorted(set().union(*(row.keys() for row in rows))):
        if not name.startswith(("velocity_", "gait_", "load_", "tracking_rmse")):
            continue
        values = [row[name] for row in rows if isinstance(row.get(name), (int, float))
                  and not isinstance(row[name], bool) and math.isfinite(row[name])]
        if values:
            result[name + "_mean"] = float(np.mean(values))
            result[name + "_std"] = float(np.std(values))
            result[name + "_count"] = len(values)
    return result


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, allow_nan=False) if isinstance(value, (list, dict))
                             else value for key, value in row.items()})
