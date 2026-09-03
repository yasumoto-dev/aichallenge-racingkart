import pytest

from frenet_optimal_trajectory_planner.trajectory_candidates import (
    generate_candidates,
    is_feasible,
    sample_candidate,
    select_best_candidate,
)


def make_candidates(lateral_offsets=(-2.0, 0.0, 2.0), target_speed=10.0, duration=3.0):
    return generate_candidates(
        d0=0.0, d0_dot=0.0, d0_ddot=0.0,
        s0=0.0, s0_dot=10.0, s0_ddot=0.0,
        target_speed=target_speed,
        duration=duration,
        lateral_offsets=lateral_offsets,
    )


def test_candidate_meets_boundary_conditions():
    [candidate] = make_candidates(lateral_offsets=(3.0,), duration=4.0)
    assert candidate.d(0.0) == pytest.approx(0.0)
    assert candidate.d_dot(0.0) == pytest.approx(0.0)
    assert candidate.d(4.0) == pytest.approx(3.0)
    assert candidate.d_dot(4.0) == pytest.approx(0.0)
    assert candidate.d_ddot(4.0) == pytest.approx(0.0)
    assert candidate.s(0.0) == pytest.approx(0.0)
    assert candidate.s_dot(0.0) == pytest.approx(10.0)
    assert candidate.s_dot(4.0) == pytest.approx(10.0)


def test_sample_candidate_covers_full_duration():
    [candidate] = make_candidates(lateral_offsets=(0.0,), duration=3.0)
    samples = sample_candidate(candidate, dt=0.5)
    assert samples[0].t == pytest.approx(0.0)
    assert samples[-1].t == pytest.approx(3.0)


def test_is_feasible_rejects_offset_beyond_track_limit():
    [candidate] = make_candidates(lateral_offsets=(5.0,), duration=3.0)
    samples = sample_candidate(candidate, dt=0.5)
    assert not is_feasible(
        samples,
        max_lateral_accel=10.0, max_lateral_jerk=100.0,
        avoid_offset_max=3.0,
        obstacle_predictions={},
        safety_radius=1.0,
    )


def test_is_feasible_rejects_collision_with_obstacle():
    [candidate] = make_candidates(lateral_offsets=(0.0,), duration=3.0)
    samples = sample_candidate(candidate, dt=0.5)
    # Obstacle sits exactly on the ego's (s, d) path at every sample.
    predictions = {"d2": [(s.s, s.d) for s in samples]}
    assert not is_feasible(
        samples,
        max_lateral_accel=10.0, max_lateral_jerk=100.0,
        avoid_offset_max=3.0,
        obstacle_predictions=predictions,
        safety_radius=1.0,
    )


def test_is_feasible_accepts_clear_path():
    [candidate] = make_candidates(lateral_offsets=(2.0,), duration=3.0)
    samples = sample_candidate(candidate, dt=0.5)
    # Obstacle stays far away in d at every sample.
    predictions = {"d2": [(s.s, -5.0) for s in samples]}
    assert is_feasible(
        samples,
        max_lateral_accel=10.0, max_lateral_jerk=100.0,
        avoid_offset_max=3.0,
        obstacle_predictions=predictions,
        safety_radius=1.0,
    )


def test_select_best_candidate_prefers_staying_on_centerline_when_clear():
    candidates = make_candidates(lateral_offsets=(-2.0, 0.0, 2.0), duration=3.0)
    best, samples = select_best_candidate(
        candidates, dt=0.5, target_speed=10.0,
        max_lateral_accel=10.0, max_lateral_jerk=100.0,
        avoid_offset_max=3.0,
        obstacle_predictions={},
        safety_radius=1.0,
        w_lateral_jerk=1.0, w_lateral_offset=1.0, w_speed_deviation=1.0,
    )
    assert best is not None
    assert best.terminal_d == pytest.approx(0.0)
    assert len(samples) > 0


def test_select_best_candidate_avoids_obstacle_on_centerline():
    # All three candidates share the same longitudinal profile (s(t) = 10*t,
    # since s0_dot already equals target_speed), so a stationary obstacle
    # directly ahead on the centerline is only ever close in s at t=1.5s
    # (s=15). At that instant the d=0 candidate sits exactly on top of it
    # (distance 0) while the +/-2 candidates have swung out to |d|=1.0
    # (the quintic S-curve is exactly half-elapsed at the midpoint T/2).
    # safety_radius=0.5 cleanly separates "collides" from "clears it".
    candidates = make_candidates(lateral_offsets=(-2.0, 0.0, 2.0), duration=3.0)
    num_samples = len(sample_candidate(candidates[0], dt=0.5))
    obstacle_on_centerline = {"d2": [(15.0, 0.0)] * num_samples}

    best, _ = select_best_candidate(
        candidates, dt=0.5, target_speed=10.0,
        max_lateral_accel=10.0, max_lateral_jerk=100.0,
        avoid_offset_max=3.0,
        obstacle_predictions=obstacle_on_centerline,
        safety_radius=0.5,
        w_lateral_jerk=1.0, w_lateral_offset=1.0, w_speed_deviation=1.0,
    )
    assert best is not None
    assert best.terminal_d != pytest.approx(0.0)


def test_select_best_candidate_returns_none_when_all_infeasible():
    candidates = make_candidates(lateral_offsets=(-2.0, 0.0, 2.0), duration=3.0)
    samples_by_offset = {c.terminal_d: sample_candidate(c, dt=0.5) for c in candidates}
    blocking_predictions = {}
    for i, (offset, samples) in enumerate(samples_by_offset.items()):
        blocking_predictions[f"blocker_{i}"] = [(s.s, s.d) for s in samples]

    best, samples = select_best_candidate(
        candidates, dt=0.5, target_speed=10.0,
        max_lateral_accel=10.0, max_lateral_jerk=100.0,
        avoid_offset_max=3.0,
        obstacle_predictions=blocking_predictions,
        safety_radius=1.0,
        w_lateral_jerk=1.0, w_lateral_offset=1.0, w_speed_deviation=1.0,
    )
    assert best is None
    assert samples == []
