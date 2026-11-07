# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Internal pure NumPy metrics for the symmetric locomotion leg-usage study.

The functions in this module deliberately have no Isaac Sim dependency.  This
keeps the protocol equations independently testable and makes archived
rollouts analyzable on machines that do not have the simulator installed.
"""

from __future__ import annotations

import copy
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

LEG_NAMES = ("fl", "fr", "rl", "rr")
JOINT_NAMES_PER_LEG = ("hip", "thigh", "calf")
EFFORT_LIMIT_SENTINEL_NM = 1.0e8

DEFAULT_EVALUATION_CONFIG: dict[str, Any] = {
    "contact": {
        "mode": "body_weight",
        "on_n": 20.0,
        "off_n": 10.0,
        "alpha_on": 0.12,
        "alpha_off": 0.06,
        "minimum_on_n": 10.0,
        "minimum_off_n": 5.0,
        "minimum_dwell_s": 0.02,
        "ground_filtered_required": True,
    },
    "gait": {
        "boundary_exclusion_cycles": 0.02,
        "event_match_margin_cycles": 0.20,
        "phase_sync_tolerance_cycles": 0.02,
        "simultaneous_tolerance_cycles": 0.04,
        "classifier_max_error_cycles": 0.12,
        "classifier_min_margin_cycles": 0.01,
        "minimum_complete_cycles": 2,
        "minimum_events_per_foot": 2,
    },
    "tracking": {
        "vx_relative_error_limit": 0.20,
        "relative_error_epsilon_mps": 1.0e-6,
        "yaw_rmse_limit_radps": 0.05,
        "rise_fraction": 0.90,
        "settling_relative_band": 0.10,
    },
    "load": {
        "joint_saturation_fraction": 0.95,
        "impact_window_s": 0.05,
        "concentration_epsilon": 1.0e-12,
    },
    "success": {
        "require_no_termination": True,
        "require_positive_progress": True,
        "maximum_vx_relative_rmse": 0.20,
        "maximum_yaw_rmse_radps": 0.05,
        "minimum_gait_agreement": 0.80,
        "require_correct_family": True,
        "minimum_complete_cycles": 2,
        "minimum_events_per_foot": 2,
    },
}


def evaluation_config(overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return a validated evaluation configuration with nested overrides."""
    result = copy.deepcopy(DEFAULT_EVALUATION_CONFIG)
    if overrides is not None:
        unknown_sections = set(overrides).difference(result)
        if unknown_sections:
            raise ValueError(f"Unknown evaluation config sections: {sorted(unknown_sections)}")
        for section, values in overrides.items():
            if not isinstance(values, Mapping):
                raise ValueError(f"Evaluation config section {section!r} must be a mapping.")
            unknown = set(values).difference(result[section])
            if unknown:
                raise ValueError(f"Unknown evaluation config values in {section!r}: {sorted(unknown)}")
            result[section].update(values)
    contact = result["contact"]
    if contact["mode"] not in {"absolute", "body_weight"}:
        raise ValueError("contact.mode must be 'absolute' or 'body_weight'.")
    numeric_nonnegative = (
        ("contact", "on_n"),
        ("contact", "off_n"),
        ("contact", "alpha_on"),
        ("contact", "alpha_off"),
        ("contact", "minimum_on_n"),
        ("contact", "minimum_off_n"),
        ("contact", "minimum_dwell_s"),
        ("gait", "boundary_exclusion_cycles"),
        ("gait", "event_match_margin_cycles"),
        ("gait", "phase_sync_tolerance_cycles"),
        ("gait", "simultaneous_tolerance_cycles"),
        ("gait", "classifier_max_error_cycles"),
        ("gait", "classifier_min_margin_cycles"),
        ("tracking", "vx_relative_error_limit"),
        ("tracking", "relative_error_epsilon_mps"),
        ("tracking", "yaw_rmse_limit_radps"),
        ("load", "impact_window_s"),
        ("load", "concentration_epsilon"),
        ("success", "maximum_vx_relative_rmse"),
        ("success", "maximum_yaw_rmse_radps"),
        ("success", "minimum_gait_agreement"),
    )
    for section, name in numeric_nonnegative:
        value = float(result[section][name])
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{section}.{name} must be finite and non-negative.")
        result[section][name] = value
    for section, name in (
        ("gait", "minimum_complete_cycles"),
        ("gait", "minimum_events_per_foot"),
        ("success", "minimum_complete_cycles"),
        ("success", "minimum_events_per_foot"),
    ):
        value = int(result[section][name])
        if value < 0:
            raise ValueError(f"{section}.{name} must be non-negative.")
        result[section][name] = value
    on_n, off_n = resolve_contact_thresholds(result, robot_mass_kg=None)
    if not on_n > off_n >= 0.0:
        raise ValueError("Contact thresholds must satisfy F_on > F_off >= 0 N.")
    if result["contact"]["mode"] == "body_weight" and not (
        float(result["contact"]["alpha_on"]) > float(result["contact"]["alpha_off"]) >= 0.0
        and float(result["contact"]["minimum_on_n"]) > float(result["contact"]["minimum_off_n"]) >= 0.0
    ):
        raise ValueError("Body-weight contact parameters must preserve strict on/off hysteresis ordering.")
    if float(result["tracking"]["relative_error_epsilon_mps"]) <= 0.0:
        raise ValueError("tracking.relative_error_epsilon_mps must be positive.")
    if not 0.0 < float(result["tracking"]["rise_fraction"]) <= 1.0:
        raise ValueError("tracking.rise_fraction must lie in (0, 1].")
    if not 0.0 <= float(result["tracking"]["settling_relative_band"]) < 1.0:
        raise ValueError("tracking.settling_relative_band must lie in [0, 1).")
    if not 0.0 < float(result["load"]["joint_saturation_fraction"]) <= 1.0:
        raise ValueError("load.joint_saturation_fraction must lie in (0, 1].")
    if float(result["load"]["concentration_epsilon"]) <= 0.0:
        raise ValueError("load.concentration_epsilon must be positive.")
    if not 0.0 <= float(result["success"]["minimum_gait_agreement"]) <= 1.0:
        raise ValueError("success.minimum_gait_agreement must lie in [0, 1].")
    return result


def resolve_contact_thresholds(config: Mapping[str, Any], robot_mass_kg: float | None) -> tuple[float, float]:
    """Resolve contact hysteresis thresholds [N]."""
    contact = config.get("contact", config)
    if str(contact["mode"]) == "absolute":
        return float(contact["on_n"]), float(contact["off_n"])
    # A missing mass is allowed only for static config validation.  The minima
    # are still enough to verify the strict ordering of the default protocol.
    quarter_weight = 0.0 if robot_mass_kg is None else float(robot_mass_kg) * 9.80665 / 4.0
    if robot_mass_kg is not None and (not math.isfinite(robot_mass_kg) or robot_mass_kg <= 0.0):
        raise ValueError("Robot mass must be finite and positive for body-weight contact thresholds.")
    on_n = max(float(contact["minimum_on_n"]), float(contact["alpha_on"]) * quarter_weight)
    off_n = max(float(contact["minimum_off_n"]), float(contact["alpha_off"]) * quarter_weight)
    return on_n, off_n


def hysteresis_contacts(
    vertical_force_n: np.ndarray,
    on_threshold_n: float,
    off_threshold_n: float,
    minimum_dwell_samples: int = 1,
) -> np.ndarray:
    """Classify ground contact using force hysteresis and transition dwell."""
    force = np.asarray(vertical_force_n, dtype=float)
    if force.ndim != 2 or force.shape[1] != 4:
        raise ValueError("vertical_force_n must have shape (T, 4).")
    if not np.all(np.isfinite(force)):
        raise ValueError("vertical_force_n contains non-finite samples.")
    if not on_threshold_n > off_threshold_n >= 0.0:
        raise ValueError("Contact thresholds must satisfy F_on > F_off >= 0 N.")
    dwell = int(minimum_dwell_samples)
    if dwell < 1:
        raise ValueError("minimum_dwell_samples must be at least one.")
    contacts = np.zeros(force.shape, dtype=bool)
    for leg in range(4):
        state = False
        candidate_state: bool | None = None
        candidate_start = 0
        for index, sample in enumerate(force[:, leg]):
            requested = (
                True
                if (not state and sample >= on_threshold_n)
                else False
                if (state and sample <= off_threshold_n)
                else state
            )
            if requested == state:
                candidate_state = None
            elif candidate_state != requested:
                candidate_state = requested
                candidate_start = index
            if candidate_state is not None and index - candidate_start + 1 >= dwell:
                state = candidate_state
                contacts[candidate_start : index + 1, leg] = state
                candidate_state = None
            contacts[index, leg] = state
    return contacts


def desired_stance_states(
    common_phase_cycles: np.ndarray,
    foot_phase_offsets: Sequence[float],
    duty_factor: float | np.ndarray,
    boundary_exclusion_cycles: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return desired stance, boundary-valid mask, and wrapped foot phases."""
    common = np.asarray(common_phase_cycles, dtype=float).reshape(-1)
    offsets = np.asarray(foot_phase_offsets, dtype=float)
    if offsets.shape != (4,):
        raise ValueError("foot_phase_offsets must contain FL, FR, RL, RR values.")
    duty = np.asarray(duty_factor, dtype=float)
    if duty.ndim == 0:
        duty = np.full(common.shape, float(duty))
    duty = duty.reshape(-1)
    if duty.shape != common.shape or not np.all((duty > 0.0) & (duty < 1.0)):
        raise ValueError("duty_factor must be in (0, 1) and match common phase samples.")
    phases = np.remainder(common[:, None] + offsets[None, :], 1.0)
    stance_start = 1.0 - duty[:, None]
    desired = phases >= stance_start
    distance_zero = np.minimum(phases, 1.0 - phases)
    distance_stance = np.abs(circular_difference(phases, stance_start))
    valid = (distance_zero > boundary_exclusion_cycles) & (distance_stance > boundary_exclusion_cycles)
    return desired, valid, phases


def _safe_divide(numerator: float, denominator: float) -> float | None:
    return None if denominator <= 0.0 else float(numerator / denominator)


def _classification_metrics(actual: np.ndarray, desired: np.ndarray, mask: np.ndarray) -> dict[str, float | None]:
    actual = actual[mask]
    desired = desired[mask]
    if actual.size == 0:
        return {
            key: None
            for key in ("agreement", "false_swing_contact", "missed_stance_contact", "precision", "recall", "f1")
        }
    tp = int(np.count_nonzero(actual & desired))
    fp = int(np.count_nonzero(actual & ~desired))
    fn = int(np.count_nonzero(~actual & desired))
    desired_swing = int(np.count_nonzero(~desired))
    desired_stance = int(np.count_nonzero(desired))
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = (
        None
        if precision is None or recall is None or precision + recall == 0.0
        else 2.0 * precision * recall / (precision + recall)
    )
    return {
        "agreement": float(np.mean(actual == desired)),
        "false_swing_contact": _safe_divide(fp, desired_swing),
        "missed_stance_contact": _safe_divide(fn, desired_stance),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def circular_difference(values: np.ndarray | float, reference: np.ndarray | float) -> np.ndarray:
    """Return signed shortest differences in cycle units in ``[-0.5, 0.5)``."""
    return np.remainder(np.asarray(values) - np.asarray(reference) + 0.5, 1.0) - 0.5


def _event_indices(states: np.ndarray, rising: bool) -> list[np.ndarray]:
    delta = np.diff(states.astype(np.int8), axis=0)
    target = 1 if rising else -1
    return [np.flatnonzero(delta[:, leg] == target) + 1 for leg in range(4)]


def _complete_cycle_range(common_phase: np.ndarray) -> tuple[int, int]:
    """Return complete common-cycle bounds for a sampled half-open trace."""
    differences = np.diff(np.asarray(common_phase, dtype=float))
    positive_differences = differences[differences > 0.0]
    if not positive_differences.size:
        boundary = math.ceil(float(common_phase[0]) - 1.0e-9)
        return boundary, boundary
    exclusive_stop = float(common_phase[-1]) + float(np.median(positive_differences))
    first_cycle = math.ceil(float(common_phase[0]) - 1.0e-9)
    last_cycle = math.floor(exclusive_stop + 1.0e-9)
    return first_cycle, max(first_cycle, last_cycle)


def _match_events(
    actual: np.ndarray,
    expected: np.ndarray,
    common_phase: np.ndarray,
    foot_phase: np.ndarray,
    target_phase: np.ndarray,
    margin_cycles: float,
) -> tuple[list[float], int, int]:
    available = list(int(value) for value in actual)
    errors: list[float] = []
    unmatched_expected = 0
    for expected_index in expected:
        expected_cycle = math.floor(float(common_phase[expected_index]) + 1.0e-9)
        same_cycle = [index for index in available if math.floor(float(common_phase[index]) + 1.0e-9) == expected_cycle]
        if not same_cycle:
            unmatched_expected += 1
            continue
        candidates = np.asarray(same_cycle, dtype=int)
        distances = np.abs(circular_difference(foot_phase[candidates], target_phase[candidates]))
        nearest_position = int(np.argmin(distances))
        nearest = int(candidates[nearest_position])
        # Also require proximity to the same unwrapped common-cycle transition.
        if float(distances[nearest_position]) <= margin_cycles:
            errors.append(float(circular_difference(foot_phase[nearest], target_phase[nearest])))
            available.remove(nearest)
        else:
            unmatched_expected += 1
    return errors, unmatched_expected, len(available)


def _same_phase_synchronization(
    events: list[np.ndarray],
    common_phase: np.ndarray,
    foot_phase_offsets: np.ndarray,
    simultaneous_tolerance_cycles: float,
    step_dt: float,
    period_s: float | None,
) -> dict[str, Any]:
    """Measure planned same-phase event synchronization within complete cycles."""
    differences_s: list[float] = []
    pair_records: list[dict[str, Any]] = []
    first_cycle, last_cycle = _complete_cycle_range(common_phase)
    complete_cycles = last_cycle - first_cycle
    for first in range(4):
        for second in range(first + 1, 4):
            if (
                abs(float(circular_difference(foot_phase_offsets[first], foot_phase_offsets[second])))
                > simultaneous_tolerance_cycles
            ):
                continue
            pair_differences_s: list[float] = []
            for cycle in range(first_cycle, last_cycle):
                first_events = events[first][
                    (common_phase[events[first]] >= cycle) & (common_phase[events[first]] < cycle + 1)
                ]
                second_events = events[second][
                    (common_phase[events[second]] >= cycle) & (common_phase[events[second]] < cycle + 1)
                ]
                if first_events.size == 1 and second_events.size == 1:
                    difference_s = abs(int(first_events[0]) - int(second_events[0])) * step_dt
                    differences_s.append(difference_s)
                    pair_differences_s.append(difference_s)
            pair_differences_cycles = (
                [] if period_s is None else [difference_s / period_s for difference_s in pair_differences_s]
            )
            pair_records.append(
                {
                    "first_foot": LEG_NAMES[first],
                    "second_foot": LEG_NAMES[second],
                    "eligible_complete_cycles": complete_cycles,
                    "matched_complete_cycles": len(pair_differences_s),
                    "unmatched_complete_cycles": complete_cycles - len(pair_differences_s),
                    "coverage_fraction": _safe_divide(len(pair_differences_s), complete_cycles),
                    "mean_s": float(np.mean(pair_differences_s)) if pair_differences_s else None,
                    "p95_s": float(np.percentile(pair_differences_s, 95)) if pair_differences_s else None,
                    "mean_cycles": float(np.mean(pair_differences_cycles)) if pair_differences_cycles else None,
                    "p95_cycles": (
                        float(np.percentile(pair_differences_cycles, 95)) if pair_differences_cycles else None
                    ),
                }
            )
    differences_cycles = [] if period_s is None else [difference_s / period_s for difference_s in differences_s]
    eligible_pair_cycles = complete_cycles * len(pair_records)
    matched_pair_cycles = len(differences_s)
    return {
        "pairs": len(differences_s),
        "configured_pairs": len(pair_records),
        "eligible_pair_cycles": eligible_pair_cycles,
        "matched_pair_cycles": matched_pair_cycles,
        "unmatched_pair_cycles": eligible_pair_cycles - matched_pair_cycles,
        "coverage_fraction": _safe_divide(matched_pair_cycles, eligible_pair_cycles),
        "pair_records": pair_records,
        "mean_s": float(np.mean(differences_s)) if differences_s else None,
        "p95_s": float(np.percentile(differences_s, 95)) if differences_s else None,
        "mean_cycles": float(np.mean(differences_cycles)) if differences_cycles else None,
        "p95_cycles": float(np.percentile(differences_cycles, 95)) if differences_cycles else None,
    }


def _error_summary(errors: Sequence[float], period_s: float | None) -> dict[str, float | int | None]:
    absolute = np.abs(np.asarray(errors, dtype=float))
    result: dict[str, float | int | None] = {"matched": int(absolute.size)}
    for name, percentile in (("mean", None), ("median", 50.0), ("p95", 95.0)):
        if not absolute.size:
            value = None
        elif percentile is None:
            value = float(np.mean(absolute))
        else:
            value = float(np.percentile(absolute, percentile))
        result[f"abs_error_{name}_cycles"] = value
        result[f"abs_error_{name}_s"] = None if value is None or period_s is None else value * period_s
    return result


def _mode(values: Sequence[str]) -> str | None:
    if not values:
        return None
    counts = Counter(values)
    maximum = max(counts.values())
    return sorted(value for value, count in counts.items() if count == maximum)[0]


def _cyclic_orders_by_cycle(
    touchdown_indices: list[np.ndarray], common_phase: np.ndarray, simultaneous_tolerance: float
) -> dict[int, str]:
    """Return complete touchdown orders keyed by their common-cycle index."""
    start_cycle, stop_cycle = _complete_cycle_range(common_phase)
    orders: dict[int, str] = {}
    for cycle in range(start_cycle, stop_cycle):
        events: list[tuple[float, str]] = []
        for leg, indices in enumerate(touchdown_indices):
            in_cycle = indices[(common_phase[indices] >= cycle) & (common_phase[indices] < cycle + 1)]
            if in_cycle.size:
                events.append((float(common_phase[int(in_cycle[0])] - cycle), LEG_NAMES[leg]))
        if len(events) != 4:
            continue
        events.sort()
        classes: list[tuple[float, list[str]]] = []
        for phase, name in events:
            if not classes or phase - classes[-1][0] > simultaneous_tolerance:
                classes.append((phase, [name]))
            else:
                classes[-1][1].append(name)
        if len(classes) > 1 and classes[0][0] + 1.0 - classes[-1][0] <= simultaneous_tolerance:
            wrapped_group = [*classes[-1][1], *classes[0][1]]
            classes = [(classes[-1][0], wrapped_group), *classes[1:-1]]
        labels = tuple("+".join(sorted(group)) for _, group in classes)
        rotations = [labels[index:] + labels[:index] for index in range(len(labels))]
        orders[cycle] = ">".join(min(rotations))
    return orders


def _cyclic_orders(
    touchdown_indices: list[np.ndarray], common_phase: np.ndarray, simultaneous_tolerance: float
) -> list[str]:
    """Return complete touchdown orders in common-cycle order."""
    return list(_cyclic_orders_by_cycle(touchdown_indices, common_phase, simultaneous_tolerance).values())


def classify_gait_row(
    touchdown_indices: list[np.ndarray],
    common_phase: np.ndarray,
    gait_library: Sequence[Mapping[str, Any]],
    max_error_cycles: float,
    minimum_margin_cycles: float,
) -> dict[str, Any]:
    """Classify mean touchdown phases using all six pairwise circular differences."""
    observed: list[float] = []
    for indices in touchdown_indices:
        if not len(indices):
            return {"row": "unclassified", "family": "unclassified", "score_cycles": None, "margin_cycles": None}
        angles = 2.0 * np.pi * np.remainder(common_phase[indices], 1.0)
        observed.append(
            float(np.remainder(np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles))) / (2.0 * np.pi), 1.0))
        )
    observed_array = np.asarray(observed)
    pairs = tuple((first, second) for first in range(4) for second in range(first + 1, 4))
    observed_pairwise = np.asarray(
        [float(circular_difference(observed_array[second], observed_array[first])) for first, second in pairs]
    )
    scores: list[tuple[float, int, Mapping[str, Any]]] = []
    for stable_index, gait in enumerate(gait_library):
        offsets = np.asarray(gait["phases"], dtype=float)
        expected_touchdown = np.remainder(-offsets, 1.0)
        expected_pairwise = np.asarray(
            [
                float(circular_difference(expected_touchdown[second], expected_touchdown[first]))
                for first, second in pairs
            ]
        )
        score = float(np.mean(np.abs(circular_difference(observed_pairwise, expected_pairwise))))
        scores.append((score, stable_index, gait))
    scores.sort(key=lambda item: (item[0], item[1]))
    best_score, _, best = scores[0]
    margin = float(scores[1][0] - best_score) if len(scores) > 1 else math.inf
    if best_score > max_error_cycles or margin < minimum_margin_cycles:
        return {
            "row": "unclassified",
            "family": "unclassified",
            "score_cycles": best_score,
            "margin_cycles": margin,
            "mean_touchdown_phases": observed,
            "pairwise_touchdown_phase_differences": observed_pairwise.tolist(),
        }
    return {
        "row": str(best["name"]),
        "row_index": int(best.get("index", -1)),
        "family": str(best["family"]),
        "score_cycles": best_score,
        "margin_cycles": margin,
        "mean_touchdown_phases": observed,
        "pairwise_touchdown_phase_differences": observed_pairwise.tolist(),
    }


def gait_metrics(
    vertical_force_n: np.ndarray,
    common_phase_cycles: np.ndarray,
    foot_phase_offsets: Sequence[float],
    duty_factor: float | np.ndarray,
    step_dt: float,
    config: Mapping[str, Any],
    *,
    robot_mass_kg: float | None,
    gait_library: Sequence[Mapping[str, Any]] = (),
) -> tuple[dict[str, Any], np.ndarray]:
    """Compute contact, event, synchronization, and gait-classification metrics."""
    cfg = evaluation_config(config)
    common = np.asarray(common_phase_cycles, dtype=float).reshape(-1)
    forces = np.asarray(vertical_force_n, dtype=float)
    if forces.shape != (common.size, 4):
        raise ValueError("Force and common-phase traces must have matching (T, 4)/(T,) shapes.")
    if common.size < 2 or np.any(np.diff(common) < 0.0):
        raise ValueError("common_phase_cycles must be nondecreasing and contain at least two samples.")
    on_n, off_n = resolve_contact_thresholds(cfg, robot_mass_kg)
    dwell = max(1, int(math.ceil(float(cfg["contact"]["minimum_dwell_s"]) / step_dt - 1.0e-12)))
    contacts = hysteresis_contacts(np.maximum(forces, 0.0), on_n, off_n, dwell)
    desired, boundary_mask, foot_phases = desired_stance_states(
        common,
        foot_phase_offsets,
        duty_factor,
        float(cfg["gait"]["boundary_exclusion_cycles"]),
    )
    duty_trace = np.asarray(duty_factor, dtype=float)
    if duty_trace.ndim == 0:
        duty_trace = np.full(common.shape, float(duty_trace))
    all_mask = np.ones_like(contacts, dtype=bool)
    all_metrics = _classification_metrics(contacts, desired, all_mask)
    excluded_metrics = _classification_metrics(contacts, desired, boundary_mask)
    result: dict[str, Any] = {
        "contact_threshold_on_n": on_n,
        "contact_threshold_off_n": off_n,
        "contact_minimum_dwell_samples": dwell,
        "contact_minimum_dwell_s_effective": dwell * step_dt,
        "agreement": all_metrics["agreement"],
        "agreement_boundary_excluded": excluded_metrics["agreement"],
        "false_swing_contact_fraction": excluded_metrics["false_swing_contact"],
        "missed_stance_contact_fraction": excluded_metrics["missed_stance_contact"],
        "contact_precision": excluded_metrics["precision"],
        "contact_recall": excluded_metrics["recall"],
        "contact_f1": excluded_metrics["f1"],
        "boundary_excluded_fraction": 1.0 - float(np.mean(boundary_mask)),
        "measured_duty_factor": float(np.mean(contacts)),
        "commanded_duty_factor": float(np.mean(duty_trace)),
        "sampled_desired_stance_fraction": float(np.mean(desired)),
        # Backward-compatible alias for the sampled desired binary trace.
        "desired_duty_factor": float(np.mean(desired)),
    }
    per_foot_metrics: list[dict[str, Any]] = []
    measured_duty_per_foot = np.mean(contacts, axis=0)
    sampled_desired_per_foot = np.mean(desired, axis=0)
    commanded_duty_per_foot = np.full(4, float(np.mean(duty_trace)))
    duty_error_per_foot = measured_duty_per_foot - commanded_duty_per_foot
    result["measured_duty_factor_per_foot"] = measured_duty_per_foot.tolist()
    result["commanded_duty_factor_per_foot"] = commanded_duty_per_foot.tolist()
    result["sampled_desired_stance_fraction_per_foot"] = sampled_desired_per_foot.tolist()
    # Backward-compatible alias for the sampled desired binary trace.
    result["desired_duty_factor_per_foot"] = sampled_desired_per_foot.tolist()
    result["duty_factor_error_per_foot"] = duty_error_per_foot.tolist()
    result["duty_factor_abs_error_per_foot"] = np.abs(duty_error_per_foot).tolist()
    result["duty_factor_error"] = float(np.mean(duty_error_per_foot))
    result["duty_factor_abs_error_mean"] = float(np.mean(np.abs(duty_error_per_foot)))
    result["duty_factor_abs_error_max"] = float(np.max(np.abs(duty_error_per_foot)))
    for leg, name in enumerate(LEG_NAMES):
        foot_all = _classification_metrics(contacts[:, leg], desired[:, leg], all_mask[:, leg])
        foot_excluded = _classification_metrics(contacts[:, leg], desired[:, leg], boundary_mask[:, leg])
        per_foot_metrics.append(
            {
                "foot": name,
                "agreement": foot_all["agreement"],
                "agreement_boundary_excluded": foot_excluded["agreement"],
                "false_swing_contact_fraction": foot_excluded["false_swing_contact"],
                "missed_stance_contact_fraction": foot_excluded["missed_stance_contact"],
                "contact_precision": foot_excluded["precision"],
                "contact_recall": foot_excluded["recall"],
                "contact_f1": foot_excluded["f1"],
                "measured_duty_factor": float(measured_duty_per_foot[leg]),
                "commanded_duty_factor": float(commanded_duty_per_foot[leg]),
                "sampled_desired_stance_fraction": float(sampled_desired_per_foot[leg]),
                "desired_duty_factor": float(sampled_desired_per_foot[leg]),
                "duty_factor_error": float(duty_error_per_foot[leg]),
                "duty_factor_abs_error": float(abs(duty_error_per_foot[leg])),
            }
        )
    touchdown = _event_indices(contacts, rising=True)
    liftoff = _event_indices(contacts, rising=False)
    expected_touchdown = _event_indices(desired, rising=True)
    expected_liftoff = _event_indices(desired, rising=False)
    period_s = float(np.median(step_dt / np.diff(common))) if np.all(np.diff(common) > 0.0) else None
    margin = float(cfg["gait"]["event_match_margin_cycles"])
    for label, actual_events, expected_events, target in (
        ("touchdown", touchdown, expected_touchdown, 1.0 - duty_trace),
        ("liftoff", liftoff, expected_liftoff, np.zeros_like(duty_trace)),
    ):
        all_errors: list[float] = []
        unmatched_expected = 0
        unmatched_actual = 0
        counts: list[int] = []
        for leg in range(4):
            errors, missing, extra = _match_events(
                actual_events[leg],
                expected_events[leg],
                common,
                foot_phases[:, leg],
                target,
                margin,
            )
            all_errors.extend(errors)
            unmatched_expected += missing
            unmatched_actual += extra
            counts.append(len(actual_events[leg]))
            foot_summary = _error_summary(errors, period_s)
            per_foot_metrics[leg].update({f"{label}_{key}": value for key, value in foot_summary.items()})
            per_foot_metrics[leg][f"{label}_unmatched_expected"] = missing
            per_foot_metrics[leg][f"{label}_unmatched_actual"] = extra
            foot_matched = int(foot_summary["matched"])
            per_foot_metrics[leg][f"{label}_unmatched_fraction"] = _safe_divide(
                missing + extra, foot_matched + missing + extra
            )
            per_foot_metrics[leg][f"{label}_expected_unmatched_fraction"] = _safe_divide(
                missing, foot_matched + missing
            )
            per_foot_metrics[leg][f"{label}_actual_unmatched_fraction"] = _safe_divide(extra, foot_matched + extra)
            per_foot_metrics[leg][f"{label}_event_count"] = len(actual_events[leg])
        for key, value in _error_summary(all_errors, period_s).items():
            result[f"{label}_{key}"] = value
        result[f"{label}_unmatched_expected"] = unmatched_expected
        result[f"{label}_unmatched_actual"] = unmatched_actual
        matched = int(result[f"{label}_matched"])
        result[f"{label}_unmatched_fraction"] = _safe_divide(
            unmatched_expected + unmatched_actual,
            matched + unmatched_expected + unmatched_actual,
        )
        result[f"{label}_expected_unmatched_fraction"] = _safe_divide(unmatched_expected, matched + unmatched_expected)
        result[f"{label}_actual_unmatched_fraction"] = _safe_divide(unmatched_actual, matched + unmatched_actual)
        result[f"{label}_events_per_foot_min"] = min(counts)
        result[f"{label}_events_per_foot"] = counts
    simultaneous_tolerance = float(cfg["gait"]["simultaneous_tolerance_cycles"])
    orders_by_cycle = _cyclic_orders_by_cycle(touchdown, common, simultaneous_tolerance)
    expected_orders_by_cycle = _cyclic_orders_by_cycle(expected_touchdown, common, simultaneous_tolerance)
    orders = list(orders_by_cycle.values())
    expected_orders = list(expected_orders_by_cycle.values())
    comparable_cycles = sorted(orders_by_cycle.keys() & expected_orders_by_cycle.keys())
    result["measured_cyclic_order"] = _mode(orders)
    result["desired_cyclic_order"] = _mode(expected_orders)
    result["cyclic_order_complete_cycles"] = len(orders)
    result["cyclic_order_expected_complete_cycles"] = len(expected_orders)
    result["cyclic_order_comparable_cycles"] = len(comparable_cycles)
    result["cyclic_order_agreement"] = (
        None
        if not comparable_cycles
        else float(np.mean([orders_by_cycle[cycle] == expected_orders_by_cycle[cycle] for cycle in comparable_cycles]))
    )
    offsets = np.asarray(foot_phase_offsets, dtype=float)
    tolerance = float(cfg["gait"]["phase_sync_tolerance_cycles"])
    for label, event_lists in (("touchdown", touchdown), ("liftoff", liftoff)):
        synchronization = _same_phase_synchronization(
            event_lists,
            common,
            offsets,
            tolerance,
            step_dt,
            period_s,
        )
        for key, value in synchronization.items():
            result[f"{label}_same_phase_sync_{key}"] = value
    # Preserve the original touchdown-only names for existing report consumers.
    for suffix in ("pairs", "mean_s", "p95_s", "mean_cycles", "p95_cycles"):
        result[f"same_phase_sync_{suffix}"] = result[f"touchdown_same_phase_sync_{suffix}"]
    first_complete_cycle, last_complete_cycle = _complete_cycle_range(common)
    complete_cycles = last_complete_cycle - first_complete_cycle
    result["complete_cycles"] = complete_cycles
    minimum_cycles = int(cfg["gait"]["minimum_complete_cycles"])
    minimum_events = int(cfg["gait"]["minimum_events_per_foot"])
    reasons: list[str] = []
    if complete_cycles < minimum_cycles:
        reasons.append(f"complete_cycles<{minimum_cycles}")
    if min(result["touchdown_events_per_foot"]) < minimum_events:
        reasons.append(f"touchdown_events_per_foot<{minimum_events}")
    if min(result["liftoff_events_per_foot"]) < minimum_events:
        reasons.append(f"liftoff_events_per_foot<{minimum_events}")
    result["coverage_valid"] = not reasons
    result["coverage_reason"] = "; ".join(reasons)
    classifier = (
        classify_gait_row(
            touchdown,
            common,
            gait_library,
            float(cfg["gait"]["classifier_max_error_cycles"]),
            float(cfg["gait"]["classifier_min_margin_cycles"]),
        )
        if gait_library and not reasons
        else {
            "row": "unclassified",
            "family": "unclassified",
            "score_cycles": None,
            "margin_cycles": None,
            "reason": "; ".join(reasons) if reasons else "gait_library_unavailable",
        }
    )
    result.update({f"classified_{key}": value for key, value in classifier.items()})
    result["per_foot_metrics"] = per_foot_metrics
    return result, contacts


def velocity_metrics(
    commands: np.ndarray,
    body_velocities: np.ndarray,
    base_positions_xy: np.ndarray,
    step_dt: float,
    *,
    base_headings_rad: np.ndarray | None = None,
    desired_headings_rad: np.ndarray | None = None,
    terminated: bool = False,
    full_commands: np.ndarray | None = None,
    full_body_velocities: np.ndarray | None = None,
    measurement_start_index: int | None = None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute signed velocity, heading, path, and optional transient metrics."""
    cfg = evaluation_config(config)
    command = np.asarray(commands, dtype=float)
    velocity = np.asarray(body_velocities, dtype=float)
    position = np.asarray(base_positions_xy, dtype=float)
    if command.shape != velocity.shape or command.ndim != 2 or command.shape[1] != 3:
        raise ValueError("commands and body_velocities must have matching shape (T, 3).")
    if position.shape[0] != command.shape[0] or position.ndim != 2 or position.shape[1] < 2:
        raise ValueError("base_positions_xy must have shape (T, >=2).")
    vx_error = velocity[:, 0] - command[:, 0]
    vy_error = velocity[:, 1] - command[:, 1]
    yaw_error = velocity[:, 2] - command[:, 2]
    command_x = float(np.mean(command[:, 0]))
    relative_epsilon = float(cfg["tracking"]["relative_error_epsilon_mps"])
    denominator = abs(command_x) + relative_epsilon
    command_sign = 1.0 if command_x >= 0.0 else -1.0
    progress = command_sign * float(position[-1, 0] - position[0, 0])
    requested_distance = float(np.sum(np.abs(command[:, 0])) * step_dt)
    vx_rmse = float(np.sqrt(np.mean(np.square(vx_error))))
    result: dict[str, Any] = {
        "vx_rmse_mps": vx_rmse,
        "vx_mae_mps": float(np.mean(np.abs(vx_error))),
        "vx_bias_mps": float(np.mean(vx_error)),
        "vx_command_mean_mps": command_x,
        "vx_measured_mean_mps": float(np.mean(velocity[:, 0])),
        "vx_gain": None if abs(command_x) <= relative_epsilon else float(np.mean(velocity[:, 0]) / command_x),
        "vx_relative_error_denominator_mps": denominator,
        "vx_relative_error_epsilon_mps": relative_epsilon,
        "vx_relative_rmse": vx_rmse / denominator,
        "vx_sign_error_fraction": float(np.mean(command[:, 0] * velocity[:, 0] < 0.0)),
        "vy_rmse_mps": float(np.sqrt(np.mean(np.square(vy_error)))),
        "vy_mae_mps": float(np.mean(np.abs(vy_error))),
        "yaw_rmse_radps": float(np.sqrt(np.mean(np.square(yaw_error)))),
        "yaw_mae_radps": float(np.mean(np.abs(yaw_error))),
        "lateral_position_rmse_m": float(np.sqrt(np.mean(np.square(position[:, 1] - position[0, 1])))),
        "lateral_position_p95_m": float(np.percentile(np.abs(position[:, 1] - position[0, 1]), 95)),
        "signed_directed_progress_m": progress,
        "commanded_distance_m": requested_distance,
        "progress_per_commanded_distance": _safe_divide(progress, requested_distance),
        "terminated": bool(terminated),
        "lost_progress_m": max(0.0, requested_distance - progress),
    }
    for percentage in (5, 10, 20):
        result[f"vx_within_{percentage}pct_fraction"] = float(
            np.mean(np.abs(vx_error) <= percentage / 100.0 * denominator)
        )
    if base_headings_rad is not None and desired_headings_rad is not None:
        heading_error = circular_difference(
            np.asarray(base_headings_rad, dtype=float) / (2.0 * np.pi),
            np.asarray(desired_headings_rad, dtype=float) / (2.0 * np.pi),
        ) * (2.0 * np.pi)
        result["heading_rmse_rad"] = float(np.sqrt(np.mean(np.square(heading_error))))
        result["heading_p95_rad"] = float(np.percentile(np.abs(heading_error), 95))
    else:
        result["heading_rmse_rad"] = None
        result["heading_p95_rad"] = None
    full_command = command if full_commands is None else np.asarray(full_commands, dtype=float)
    full_velocity = velocity if full_body_velocities is None else np.asarray(full_body_velocities, dtype=float)
    target = float(np.mean(full_command[:, 0]))
    directed_velocity = np.sign(target) * full_velocity[:, 0]
    magnitude = abs(target)
    rise_level = float(cfg["tracking"]["rise_fraction"]) * magnitude
    rise_indices = np.flatnonzero(directed_velocity >= rise_level)
    result["rise_time_90_s"] = float(rise_indices[0] * step_dt) if rise_indices.size else None
    band = float(cfg["tracking"]["settling_relative_band"]) * magnitude
    inside = np.abs(full_velocity[:, 0] - target) <= band
    settling_index = None
    for index in range(len(inside)):
        if bool(np.all(inside[index:])):
            settling_index = index
            break
    result["settling_time_s"] = None if settling_index is None else settling_index * step_dt
    measurement_start = 0 if measurement_start_index is None else int(measurement_start_index)
    if measurement_start < 0 or measurement_start >= len(inside):
        raise ValueError("measurement_start_index must select a sample in the full transient trace.")
    measurement_count = len(inside) - measurement_start
    post_settle_count = 0 if settling_index is None else len(inside) - max(settling_index, measurement_start)
    result["measurement_window_start_s"] = measurement_start * step_dt
    result["post_settle_in_band_fraction"] = (
        None if settling_index is None else float(post_settle_count / measurement_count)
    )
    return result


def validate_effort_limits(effort_limits_nm: np.ndarray) -> np.ndarray:
    """Validate physical configured effort limits [N m], rejecting solver sentinels."""
    limits = np.asarray(effort_limits_nm, dtype=float).reshape(-1)
    if limits.shape != (12,) or not np.all(np.isfinite(limits)) or np.any(limits <= 0.0):
        raise ValueError("Expected 12 finite positive configured joint effort limits.")
    if np.any(limits >= EFFORT_LIMIT_SENTINEL_NM):
        raise ValueError("Effort limits contain a solver sentinel and cannot normalize torque.")
    return limits


def _concentration(values: np.ndarray, prefix: str, epsilon: float = 1.0e-12) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("Concentration epsilon must be finite and positive.")
    total = float(np.sum(values))
    mean = float(np.mean(values))
    std = float(np.std(values))
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    zero_leg_count = int(np.count_nonzero(values <= 0.0))
    front = float(np.sum(values[:2]))
    hind = float(np.sum(values[2:]))
    left = float(values[0] + values[2])
    right = float(values[1] + values[3])
    return {
        f"{prefix}_per_leg": values.tolist(),
        f"{prefix}_mean": mean,
        f"{prefix}_std": std,
        f"{prefix}_epsilon": epsilon,
        f"{prefix}_cv": std / (mean + epsilon),
        f"{prefix}_max_share": maximum / (total + epsilon),
        f"{prefix}_max_to_min": maximum / (minimum + epsilon),
        f"{prefix}_max_to_min_reason": "epsilon_regularized_zero_leg_value" if minimum <= 0.0 else "",
        f"{prefix}_zero_leg_count": zero_leg_count,
        f"{prefix}_front_hind_signed": (front - hind) / (total + epsilon),
        f"{prefix}_front_hind_abs": abs(front - hind) / (total + epsilon),
        f"{prefix}_left_right_signed": (left - right) / (total + epsilon),
        f"{prefix}_left_right_abs": abs(left - right) / (total + epsilon),
        f"{prefix}_worst_leg": LEG_NAMES[int(np.argmax(values))],
        f"{prefix}_worst_leg_value": maximum,
    }


def load_metrics(
    joint_torques_nm: np.ndarray,
    joint_powers_w: np.ndarray,
    configured_effort_limits_nm: np.ndarray | None,
    vertical_ground_force_n: np.ndarray | None,
    contacts: np.ndarray | None,
    step_dt: float,
    *,
    complete_cycles: int | None,
    directed_progress_m: float,
    soft_limit_utilization: np.ndarray | None = None,
    action_clamped: np.ndarray | None = None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute independent raw-actuator, normalized-actuator, and foot-load metrics."""
    cfg = evaluation_config(config)
    torque = np.asarray(joint_torques_nm, dtype=float)
    power = np.asarray(joint_powers_w, dtype=float)
    if torque.ndim != 2 or torque.shape[1] != 12 or power.shape != torque.shape:
        raise ValueError("joint torque and power traces must have shape (T, 12).")
    if not np.all(np.isfinite(torque)) or not np.all(np.isfinite(power)):
        raise ValueError("joint torque and power traces must contain finite samples.")
    grf_available = vertical_ground_force_n is not None and contacts is not None
    force: np.ndarray | None = None
    contact: np.ndarray | None = None
    if grf_available:
        force = np.maximum(np.asarray(vertical_ground_force_n, dtype=float), 0.0)
        contact = np.asarray(contacts, dtype=bool)
        if force.shape != (torque.shape[0], 4) or contact.shape != force.shape:
            raise ValueError("force/contact traces must have shape (T, 4) and match joint traces.")
        if not np.all(np.isfinite(force)):
            raise ValueError("ground-force traces must contain finite samples.")
    duration_s = torque.shape[0] * step_dt
    torque_sq_leg = np.sum(np.square(torque.reshape(-1, 4, 3)), axis=(0, 2)) * step_dt
    work_leg = np.sum(np.abs(power.reshape(-1, 4, 3)), axis=(0, 2)) * step_dt
    concentration_epsilon = float(cfg["load"]["concentration_epsilon"])
    result: dict[str, Any] = {
        "duration_s": duration_s,
        "raw_actuator_load_available": True,
        "grf_load_available": grf_available,
        "raw_torque_squared_total": float(np.sum(torque_sq_leg)),
        "absolute_work_total_j": float(np.sum(work_leg)),
    }
    result.update(_concentration(torque_sq_leg, "torque_squared", concentration_epsilon))
    result.update(_concentration(work_leg, "absolute_work", concentration_epsilon))
    for name, total in (
        ("raw_torque_squared", float(np.sum(torque_sq_leg))),
        ("absolute_work", float(np.sum(work_leg))),
    ):
        result[f"{name}_per_s"] = total / duration_s
        result[f"{name}_per_cycle"] = None if not complete_cycles else total / complete_cycles
        result[f"{name}_per_directed_m"] = None if directed_progress_m <= 0.0 else total / directed_progress_m
    if force is not None and contact is not None:
        impulse_leg = np.sum(force, axis=0) * step_dt
        total_impulse = float(np.sum(impulse_leg))
        result["vertical_grf_impulse_total_ns"] = total_impulse
        result.update(_concentration(impulse_leg, "vertical_grf_impulse", concentration_epsilon))
        result["vertical_grf_impulse_per_s"] = total_impulse / duration_s
        result["vertical_grf_impulse_per_cycle"] = None if not complete_cycles else total_impulse / complete_cycles
        result["vertical_grf_impulse_per_directed_m"] = (
            None if directed_progress_m <= 0.0 else total_impulse / directed_progress_m
        )
    normalized_available = True
    try:
        limits = (
            validate_effort_limits(configured_effort_limits_nm) if configured_effort_limits_nm is not None else None
        )
        if limits is None:
            raise ValueError("Configured effort limits were not archived.")
    except (TypeError, ValueError) as exc:
        normalized_available = False
        limits = None
        result["normalized_effort_unavailable_reason"] = str(exc)
    result["normalized_effort_available"] = normalized_available
    joint_work = np.sum(np.abs(power), axis=0) * step_dt
    worst_work = int(np.argmax(joint_work))
    result["joint_absolute_work_j"] = joint_work.tolist()
    result["worst_absolute_work_joint_index"] = worst_work
    result["worst_absolute_work_j"] = float(joint_work[worst_work])
    if limits is not None:
        utilization = np.abs(torque) / limits[None, :]
        joint_exposure = np.sum(np.square(torque / limits[None, :]), axis=0) * step_dt
        normalized_leg = np.sum(joint_exposure.reshape(4, 3), axis=1)
        result.update(_concentration(normalized_leg, "normalized_torque_squared", concentration_epsilon))
        result["normalized_torque_squared_total"] = float(np.sum(joint_exposure))
        result["normalized_torque_squared_per_s"] = float(np.sum(joint_exposure)) / duration_s
        result["normalized_torque_squared_per_cycle"] = (
            None if not complete_cycles else float(np.sum(joint_exposure)) / complete_cycles
        )
        result["normalized_torque_squared_per_directed_m"] = (
            None if directed_progress_m <= 0.0 else float(np.sum(joint_exposure)) / directed_progress_m
        )
        worst_exposure = int(np.argmax(joint_exposure))
        result["joint_normalized_torque_squared"] = joint_exposure.tolist()
        result["worst_normalized_torque_joint_index"] = worst_exposure
        result["worst_normalized_torque_exposure"] = float(joint_exposure[worst_exposure])
        result["joint_utilization_p95"] = np.percentile(utilization, 95, axis=0).tolist()
        result["joint_utilization_p99"] = np.percentile(utilization, 99, axis=0).tolist()
        saturation = float(cfg["load"]["joint_saturation_fraction"])
        result["joint_saturation_fraction"] = np.mean(utilization >= saturation, axis=0).tolist()
    if soft_limit_utilization is not None:
        soft = np.asarray(soft_limit_utilization, dtype=float)
        if soft.shape == torque.shape:
            result["joint_soft_limit_utilization_p95"] = np.percentile(soft, 95, axis=0).tolist()
            result["joint_soft_limit_utilization_p99"] = np.percentile(soft, 99, axis=0).tolist()
    if action_clamped is not None:
        clamped = np.asarray(action_clamped, dtype=bool)
        if clamped.shape == torque.shape:
            result["joint_action_clamp_fraction"] = np.mean(clamped, axis=0).tolist()
        elif clamped.ndim == 1 and clamped.shape[0] == torque.shape[0]:
            result["action_clamp_fraction"] = float(np.mean(clamped))
    if force is not None and contact is not None:
        for leg, name in enumerate(LEG_NAMES):
            contact_force = force[contact[:, leg], leg]
            result[f"{name}_contact_mean_force_n"] = float(np.mean(contact_force)) if contact_force.size else None
            result[f"{name}_force_p95_n"] = float(np.percentile(contact_force, 95)) if contact_force.size else None
            result[f"{name}_force_p99_n"] = float(np.percentile(contact_force, 99)) if contact_force.size else None
            result[f"{name}_force_max_n"] = float(np.max(contact_force)) if contact_force.size else None
            touchdown = np.flatnonzero(np.diff(contact[:, leg].astype(np.int8)) == 1) + 1
            window = max(1, int(math.ceil(float(cfg["load"]["impact_window_s"]) / step_dt)))
            peaks = [float(np.max(force[index : min(index + window, len(force)), leg])) for index in touchdown]
            impulses = [
                float(np.sum(force[index : min(index + window, len(force)), leg]) * step_dt) for index in touchdown
            ]
            result[f"{name}_impact_event_count"] = len(peaks)
            result[f"{name}_impact_peak_mean_n"] = float(np.mean(peaks)) if peaks else None
            result[f"{name}_impact_peak_p95_n"] = float(np.percentile(peaks, 95)) if peaks else None
            result[f"{name}_impact_peak_max_n"] = float(np.max(peaks)) if peaks else None
            result[f"{name}_impact_impulse_mean_ns"] = float(np.mean(impulses)) if impulses else None
            result[f"{name}_impact_impulse_p95_ns"] = float(np.percentile(impulses, 95)) if impulses else None
            result[f"{name}_impact_impulse_max_ns"] = float(np.max(impulses)) if impulses else None
    return result


def success_metrics(
    velocity: Mapping[str, Any],
    gait: Mapping[str, Any],
    *,
    planned_family: str,
    terminated: bool,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply separate velocity, gait, and joint publication-success rules."""
    cfg = evaluation_config(config)["success"]
    checks = {
        "no_termination": (not terminated) or not bool(cfg["require_no_termination"]),
        "positive_progress": (float(velocity.get("signed_directed_progress_m", 0.0)) > 0.0)
        or not bool(cfg["require_positive_progress"]),
        "vx_relative_rmse": velocity.get("vx_relative_rmse") is not None
        and float(velocity["vx_relative_rmse"]) <= float(cfg["maximum_vx_relative_rmse"]),
        "yaw_rmse": float(velocity.get("yaw_rmse_radps", math.inf)) <= float(cfg["maximum_yaw_rmse_radps"]),
        "gait_agreement": gait.get("agreement_boundary_excluded") is not None
        and float(gait["agreement_boundary_excluded"]) >= float(cfg["minimum_gait_agreement"]),
        "correct_family": str(gait.get("classified_family")) == planned_family
        or not bool(cfg["require_correct_family"]),
        "complete_cycles": int(gait.get("complete_cycles", 0)) >= int(cfg["minimum_complete_cycles"]),
        "sufficient_events": min(gait.get("touchdown_events_per_foot", [0])) >= int(cfg["minimum_events_per_foot"])
        and min(gait.get("liftoff_events_per_foot", [0])) >= int(cfg["minimum_events_per_foot"]),
    }
    velocity_check_names = ("no_termination", "positive_progress", "vx_relative_rmse", "yaw_rmse")
    gait_check_names = (
        "no_termination",
        "gait_agreement",
        "correct_family",
        "complete_cycles",
        "sufficient_events",
    )
    velocity_failed = [name for name in velocity_check_names if not checks[name]]
    gait_failed = [name for name in gait_check_names if not checks[name]]
    failed = [name for name, passed in checks.items() if not passed]
    velocity_success = not velocity_failed
    gait_success = not gait_failed
    joint_success = not failed
    return {
        "velocity_only_success": velocity_success,
        "gait_only_success": gait_success,
        "joint_velocity_and_gait_success": joint_success,
        "joint_success": joint_success,
        "success": joint_success,
        "velocity_only_success_failed_checks": velocity_failed,
        "gait_only_success_failed_checks": gait_failed,
        "success_failed_checks": failed,
        **{f"success_{key}": value for key, value in checks.items()},
    }
