"""Sanity check that the vendored FrenetOptimalTrajectory .so loads and
finds a laterally-offset avoidance path around a single obstacle placed
directly on a straight reference line.

Note on timing: the underlying quintic lateral profile ramps from d=0 to
the sampled target d smoothly over the candidate duration Ti, so a
candidate only clears an obstacle if enough of the maneuver has completed
*by the time the ego reaches the obstacle's arc length*. The obstacle here
is placed close to the end of the chosen (fixed) duration so the ramp is
nearly complete when the path passes it -- an obstacle placed too early
relative to Ti is a legitimately infeasible scenario (every sampled d is
still mid-transition and collides), not a wrapper bug.
"""
import numpy as np

from frenet_optimal_trajectory_planner import fot_wrapper

HYPERPARAMETERS = {
    "max_speed": 15.0,
    "max_accel": 5.0,
    "max_curvature": 1.0,
    "max_road_width_l": 2.0,
    "max_road_width_r": 2.0,
    "d_road_w": 0.5,
    "dt": 0.1,
    "maxt": 2.5,
    "mint": 2.5,
    "d_t_s": 1.0,
    "n_s_sample": 1.0,
    "obstacle_clearance": 0.5,
    "kd": 1.0,
    "kv": 1.0,
    "ka": 0.1,
    "kj": 0.1,
    "kt": 0.1,
    "ko": 1.0,
    "klat": 1.0,
    "klon": 1.0,
    "num_threads": 0,
}


def test_run_fot_avoids_obstacle_on_straight_line():
    wp = np.array([[float(x), 0.0] for x in range(0, 51)])
    # Reached at ~2.2-2.6s at target_speed=5.0 m/s, close to the 2.5s
    # candidate duration so the lateral maneuver has (almost) completed.
    obs = np.array([[11.0, -0.5, 13.0, 0.5]])

    initial_conditions = {
        "ps": 0.0,
        "target_speed": 5.0,
        "pos": np.array([0.0, 0.0]),
        "vel": np.array([5.0, 0.0]),
        "wp": wp,
        "obs": obs,
    }

    x_path, y_path, speeds, _ix, _iy, _iyaw, d, s, _sx, _sy, params, _costs, \
        success = fot_wrapper.run_fot(initial_conditions, HYPERPARAMETERS)

    assert success
    assert len(x_path) > 1
    # The path must swerve laterally (nonzero d) to clear the obstacle.
    assert np.max(np.abs(d)) > 0.3
    assert "s" in params


def test_run_fot_straight_line_with_no_obstacle():
    wp = np.array([[float(x), 0.0] for x in range(0, 51)])
    obs = np.empty((0, 4))

    initial_conditions = {
        "ps": 0.0,
        "target_speed": 5.0,
        "pos": np.array([0.0, 0.0]),
        "vel": np.array([5.0, 0.0]),
        "wp": wp,
        "obs": obs,
    }

    x_path, y_path, speeds, _ix, _iy, _iyaw, d, _s, _sx, _sy, _params, \
        _costs, success = fot_wrapper.run_fot(initial_conditions, HYPERPARAMETERS)

    assert success
    assert len(x_path) > 1
    assert np.max(np.abs(d)) < 1e-6
