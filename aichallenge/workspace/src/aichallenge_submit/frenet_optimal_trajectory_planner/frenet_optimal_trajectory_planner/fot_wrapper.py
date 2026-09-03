"""ctypes bridge to the vendored erdos-project FrenetOptimalTrajectory C++ core.

Adapted from
https://github.com/erdos-project/frenet_optimal_trajectory_planner
(commit ba0cb5662a0e2ea668b1c2b2951c0f6b84f44f5b, Apache-2.0). Only the shared
library discovery is changed (resolved via ament_index instead of a relative
build/ path or $PYLOT_HOME) and the sibling-module import path is updated;
the ctypes struct marshalling is unchanged.
"""
import os
from ctypes import CDLL, POINTER, byref, c_double, c_int

import numpy as np
from ament_index_python.packages import get_package_prefix

from frenet_optimal_trajectory_planner.py_cpp_struct import (
    FrenetHyperparameters,
    FrenetInitialConditions,
    FrenetReturnValues,
)

_LIB_PATH = os.path.join(
    get_package_prefix("frenet_optimal_trajectory_planner"),
    "lib",
    "frenet_optimal_trajectory_planner",
    "libFrenetOptimalTrajectory.so",
)
cdll = CDLL(_LIB_PATH)

_c_double_p = POINTER(c_double)

# func / return type declarations for C++ run_fot
_run_fot = cdll.run_fot
_run_fot.argtypes = (
    POINTER(FrenetInitialConditions),
    POINTER(FrenetHyperparameters),
    POINTER(FrenetReturnValues),
)
_run_fot.restype = None

# func / return type declarations for C++ to_frenet_initial_conditions
_to_frenet_initial_conditions = cdll.to_frenet_initial_conditions
_to_frenet_initial_conditions.restype = None
_to_frenet_initial_conditions.argtypes = (c_double, c_double, c_double,
                                          c_double, c_double, c_double,
                                          _c_double_p, _c_double_p, c_int,
                                          _c_double_p)


def _parse_hyperparameters(hp):
    return FrenetHyperparameters(
        hp["max_speed"], hp["max_accel"], hp["max_curvature"],
        hp["max_road_width_l"], hp["max_road_width_r"], hp["d_road_w"],
        hp["dt"], hp["maxt"], hp["mint"], hp["d_t_s"], hp["n_s_sample"],
        hp["obstacle_clearance"], hp["kd"], hp["kv"], hp["ka"], hp["kj"],
        hp["kt"], hp["ko"], hp["klat"], hp["klon"], hp["num_threads"])


def run_fot(initial_conditions, hyperparameters):
    """Return the frenet optimal trajectory given initial conditions in
    cartesian space.

    Args:
        initial_conditions (dict): dict containing the following items
            ps (float): previous longitudinal position
            target_speed (float): target speed [m/s]
            pos (np.ndarray([float, float])): initial position in global coord
            vel (np.ndarray([float, float])): initial velocity [m/s]
            wp (np.ndarray([float, float])): list of global waypoints
            obs (np.ndarray([float, float, float, float])): list of obstacles
                as: [lower left x, lower left y, upper right x, upper right y]

        hyperparameters (dict): a dict of optional hyperparameters, see
            py_cpp_struct.FrenetHyperparameters for the full list of fields.

    Returns:
        result_x, result_y, speeds, ix, iy, iyaw, d, s, speeds_x, speeds_y
        (np.ndarray(float)): fields of the best frenet path, if it exists
        params (dict): next frenet coordinates, if they exist
        costs (dict): costs of best frenet path, if it exists
        success (bool): whether a fot was found or not
    """
    fot_initial_conditions, misc = to_frenet_initial_conditions(
        initial_conditions)
    fot_hp = _parse_hyperparameters(hyperparameters)
    fot_rv = FrenetReturnValues(0)

    _run_fot(fot_initial_conditions, fot_hp, fot_rv)

    x_path = np.array([fot_rv.x_path[i] for i in range(fot_rv.path_length)])
    y_path = np.array([fot_rv.y_path[i] for i in range(fot_rv.path_length)])
    speeds = np.array([fot_rv.speeds[i] for i in range(fot_rv.path_length)])
    ix = np.array([fot_rv.ix[i] for i in range(fot_rv.path_length)])
    iy = np.array([fot_rv.iy[i] for i in range(fot_rv.path_length)])
    iyaw = np.array([fot_rv.iyaw[i] for i in range(fot_rv.path_length)])
    d = np.array([fot_rv.d[i] for i in range(fot_rv.path_length)])
    s = np.array([fot_rv.s[i] for i in range(fot_rv.path_length)])
    speeds_x = np.array([fot_rv.speeds_x[i] for i in range(fot_rv.path_length)])
    speeds_y = np.array([fot_rv.speeds_y[i] for i in range(fot_rv.path_length)])
    params = {
        "s": fot_rv.params[0],
        "s_d": fot_rv.params[1],
        "d": fot_rv.params[2],
        "d_d": fot_rv.params[3],
        "d_dd": fot_rv.params[4],
    }
    costs = {
        "c_lateral_deviation": fot_rv.costs[0],
        "c_lateral_velocity": fot_rv.costs[1],
        "c_lateral_acceleration": fot_rv.costs[2],
        "c_lateral_jerk": fot_rv.costs[3],
        "c_lateral": fot_rv.costs[4],
        "c_longitudinal_acceleration": fot_rv.costs[5],
        "c_longitudinal_jerk": fot_rv.costs[6],
        "c_time_taken": fot_rv.costs[7],
        "c_end_speed_deviation": fot_rv.costs[8],
        "c_longitudinal": fot_rv.costs[9],
        "c_inv_dist_to_obstacles": fot_rv.costs[10],
        "cf": fot_rv.costs[11],
    }

    success = bool(fot_rv.success)

    return (x_path, y_path, speeds, ix, iy, iyaw, d, s, speeds_x, speeds_y,
            params, costs, success)


def to_frenet_initial_conditions(initial_conditions):
    """Convert the cartesian initial conditions into frenet initial conditions.

    Args:
        initial_conditions (dict): see run_fot()

    Returns:
        FrenetInitialConditions, dictionary for debugging
    """
    ps = initial_conditions['ps']
    pos = initial_conditions['pos']
    vel = initial_conditions['vel']
    wp = initial_conditions['wp']
    obs = initial_conditions['obs']
    target_speed = initial_conditions['target_speed']
    if obs.shape[0] == 0:
        obs = np.empty((0, 4))
    x = pos[0].item()
    y = pos[1].item()
    vx = vel[0].item()
    vy = vel[1].item()
    wx = wp[:, 0].astype(np.float64)
    wy = wp[:, 1].astype(np.float64)
    o_llx = np.copy(obs[:, 0]).astype(np.float64)
    o_lly = np.copy(obs[:, 1]).astype(np.float64)
    o_urx = np.copy(obs[:, 2]).astype(np.float64)
    o_ury = np.copy(obs[:, 3]).astype(np.float64)
    forward_speed = np.hypot(vx, vy).item()

    misc = np.zeros(5)
    _to_frenet_initial_conditions(c_double(ps), c_double(x), c_double(y),
                                  c_double(vx), c_double(vy),
                                  c_double(forward_speed),
                                  wx.ctypes.data_as(_c_double_p),
                                  wy.ctypes.data_as(_c_double_p),
                                  c_int(len(wx)),
                                  misc.ctypes.data_as(_c_double_p))

    return FrenetInitialConditions(
        misc[0],  # c_s
        misc[1],  # c_speed
        misc[2],  # c_d
        misc[3],  # c_d_d
        misc[4],  # c_d_dd
        target_speed,
        wx.ctypes.data_as(_c_double_p),
        wy.ctypes.data_as(_c_double_p),
        len(wx),
        o_llx.ctypes.data_as(_c_double_p),
        o_lly.ctypes.data_as(_c_double_p),
        o_urx.ctypes.data_as(_c_double_p),
        o_ury.ctypes.data_as(_c_double_p),
        len(o_llx),
    ), misc
