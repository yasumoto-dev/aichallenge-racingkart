"""Frenet Optimal Trajectory candidate generation (Werling et al., 2010).

Pure Python/numpy, no rclpy or ROS message dependency. Obstacles are passed
in as pre-converted (t, s, d) samples so this module stays testable without
a V2X tracker or reference path in the loop.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Candidate:
    d_coeffs: np.ndarray  # quintic, low-to-high degree, length 6
    s_coeffs: np.ndarray  # quartic, low-to-high degree, length 5
    duration: float
    terminal_d: float

    def d(self, t: float) -> float:
        return float(np.polyval(self.d_coeffs[::-1], t))

    def d_dot(self, t: float) -> float:
        return float(np.polyval(np.polyder(self.d_coeffs[::-1]), t))

    def d_ddot(self, t: float) -> float:
        return float(np.polyval(np.polyder(self.d_coeffs[::-1], 2), t))

    def d_dddot(self, t: float) -> float:
        return float(np.polyval(np.polyder(self.d_coeffs[::-1], 3), t))

    def s(self, t: float) -> float:
        return float(np.polyval(self.s_coeffs[::-1], t))

    def s_dot(self, t: float) -> float:
        return float(np.polyval(np.polyder(self.s_coeffs[::-1]), t))


@dataclass
class Sample:
    t: float
    s: float
    d: float
    s_dot: float
    d_dot: float
    d_ddot: float
    d_dddot: float


def _quintic_lateral_coeffs(
    d0: float, d0_dot: float, d0_ddot: float,
    dT: float, dT_dot: float, dT_ddot: float,
    T: float,
) -> np.ndarray:
    a0, a1, a2 = d0, d0_dot, d0_ddot / 2.0
    T2, T3, T4, T5 = T ** 2, T ** 3, T ** 4, T ** 5
    A = np.array([
        [T3, T4, T5],
        [3 * T2, 4 * T3, 5 * T4],
        [6 * T, 12 * T2, 20 * T3],
    ])
    b = np.array([
        dT - (a0 + a1 * T + a2 * T2),
        dT_dot - (a1 + 2 * a2 * T),
        dT_ddot - 2 * a2,
    ])
    a3, a4, a5 = np.linalg.solve(A, b)
    return np.array([a0, a1, a2, a3, a4, a5])


def _quartic_velocity_keeping_coeffs(
    s0: float, s0_dot: float, s0_ddot: float,
    sT_dot: float, sT_ddot: float,
    T: float,
) -> np.ndarray:
    a0, a1, a2 = s0, s0_dot, s0_ddot / 2.0
    T2 = T ** 2
    A = np.array([
        [3 * T2, 4 * T ** 3],
        [6 * T, 12 * T2],
    ])
    b = np.array([
        sT_dot - (a1 + 2 * a2 * T),
        sT_ddot - 2 * a2,
    ])
    a3, a4 = np.linalg.solve(A, b)
    return np.array([a0, a1, a2, a3, a4])


def generate_candidates(
    d0: float, d0_dot: float, d0_ddot: float,
    s0: float, s0_dot: float, s0_ddot: float,
    target_speed: float,
    duration: float,
    lateral_offsets: Sequence[float],
) -> List[Candidate]:
    """One candidate per requested terminal lateral offset.

    All candidates share the same longitudinal (velocity-keeping) profile;
    only the lateral terminal state differs, matching the simplified
    single-time-horizon scope described in the design (see plan doc).
    """
    s_coeffs = _quartic_velocity_keeping_coeffs(
        s0, s0_dot, s0_ddot, target_speed, 0.0, duration)

    candidates = []
    for dT in lateral_offsets:
        d_coeffs = _quintic_lateral_coeffs(
            d0, d0_dot, d0_ddot, dT, 0.0, 0.0, duration)
        candidates.append(Candidate(d_coeffs, s_coeffs, duration, dT))
    return candidates


def sample_candidate(candidate: Candidate, dt: float) -> List[Sample]:
    samples = []
    t = 0.0
    while t <= candidate.duration + 1e-9:
        samples.append(Sample(
            t=t,
            s=candidate.s(t),
            d=candidate.d(t),
            s_dot=candidate.s_dot(t),
            d_dot=candidate.d_dot(t),
            d_ddot=candidate.d_ddot(t),
            d_dddot=candidate.d_dddot(t),
        ))
        t += dt
    return samples


def is_feasible(
    samples: Sequence[Sample],
    *,
    max_lateral_accel: float,
    max_lateral_jerk: float,
    avoid_offset_max: float,
    obstacle_predictions: Dict[str, List[Tuple[float, float]]],
    safety_radius: float,
) -> bool:
    """``obstacle_predictions`` maps vehicle_id to a list of (s, d), one
    entry per ``samples[i]`` — i.e. sampled on the same time grid as
    ``samples`` (index ``i`` corresponds to ``samples[i].t``), not matched
    by timestamp. The caller is responsible for that alignment."""
    for sample in samples:
        if abs(sample.d) > avoid_offset_max:
            return False
        if abs(sample.d_ddot) > max_lateral_accel:
            return False
        if abs(sample.d_dddot) > max_lateral_jerk:
            return False

    for predictions in obstacle_predictions.values():
        for i, (s_obs, d_obs) in enumerate(predictions):
            if i >= len(samples):
                break
            sample = samples[i]
            distance = math.hypot(sample.s - s_obs, sample.d - d_obs)
            if distance < safety_radius:
                return False
    return True


def compute_cost(
    samples: Sequence[Sample],
    candidate: Candidate,
    target_speed: float,
    *,
    w_lateral_jerk: float,
    w_lateral_offset: float,
    w_speed_deviation: float,
) -> float:
    if len(samples) < 2:
        return math.inf
    dt = samples[1].t - samples[0].t
    jerk_cost = sum(s.d_dddot ** 2 for s in samples) * dt
    offset_cost = candidate.terminal_d ** 2
    speed_cost = (target_speed - samples[0].s_dot) ** 2
    return (
        w_lateral_jerk * jerk_cost
        + w_lateral_offset * offset_cost
        + w_speed_deviation * speed_cost
    )


def select_best_candidate(
    candidates: Sequence[Candidate],
    dt: float,
    target_speed: float,
    *,
    max_lateral_accel: float,
    max_lateral_jerk: float,
    avoid_offset_max: float,
    obstacle_predictions: Dict[str, List[Tuple[float, float]]],
    safety_radius: float,
    w_lateral_jerk: float,
    w_lateral_offset: float,
    w_speed_deviation: float,
) -> Tuple[Optional[Candidate], List[Sample]]:
    """Returns the lowest-cost feasible ``(candidate, samples)`` pair, or
    ``(None, [])`` if every candidate is infeasible."""
    best_candidate = None
    best_samples: List[Sample] = []
    best_cost = math.inf

    for candidate in candidates:
        samples = sample_candidate(candidate, dt)
        if not is_feasible(
            samples,
            max_lateral_accel=max_lateral_accel,
            max_lateral_jerk=max_lateral_jerk,
            avoid_offset_max=avoid_offset_max,
            obstacle_predictions=obstacle_predictions,
            safety_radius=safety_radius,
        ):
            continue

        cost = compute_cost(
            samples, candidate, target_speed,
            w_lateral_jerk=w_lateral_jerk,
            w_lateral_offset=w_lateral_offset,
            w_speed_deviation=w_speed_deviation,
        )
        if cost < best_cost:
            best_cost = cost
            best_candidate = candidate
            best_samples = samples

    return best_candidate, best_samples
