import math

import pytest

from frenet_optimal_trajectory_planner.frenet_frame import ReferencePath


def straight_line_path():
    xs = [0.0, 10.0, 20.0, 30.0]
    ys = [0.0, 0.0, 0.0, 0.0]
    return ReferencePath(xs, ys)


def test_length_is_total_arc_length():
    path = straight_line_path()
    assert path.length == pytest.approx(30.0)


def test_to_frenet_on_path_has_zero_lateral_offset():
    path = straight_line_path()
    s, d = path.to_frenet(15.0, 0.0)
    assert s == pytest.approx(15.0)
    assert d == pytest.approx(0.0)


def test_to_frenet_left_of_path_is_positive_d():
    path = straight_line_path()
    _, d = path.to_frenet(15.0, 2.0)
    assert d == pytest.approx(2.0)


def test_to_frenet_right_of_path_is_negative_d():
    path = straight_line_path()
    _, d = path.to_frenet(15.0, -2.0)
    assert d == pytest.approx(-2.0)


def test_to_cartesian_round_trips_with_to_frenet():
    path = straight_line_path()
    x, y, heading = path.to_cartesian(12.5, 1.5)
    s, d = path.to_frenet(x, y)
    assert s == pytest.approx(12.5)
    assert d == pytest.approx(1.5)
    assert heading == pytest.approx(0.0)


def test_to_cartesian_offsets_perpendicular_to_heading():
    xs = [0.0, 0.0, 0.0]
    ys = [0.0, 10.0, 20.0]
    path = ReferencePath(xs, ys)
    x, y, heading = path.to_cartesian(5.0, 2.0)
    assert heading == pytest.approx(math.pi / 2)
    assert x == pytest.approx(-2.0)
    assert y == pytest.approx(5.0)


def test_velocity_at_interpolates_linearly():
    xs = [0.0, 10.0, 20.0]
    ys = [0.0, 0.0, 0.0]
    velocities = [5.0, 10.0, 10.0]
    path = ReferencePath(xs, ys, velocities=velocities)
    assert path.velocity_at(0.0) == pytest.approx(5.0)
    assert path.velocity_at(5.0) == pytest.approx(7.5)
    assert path.velocity_at(10.0) == pytest.approx(10.0)


def test_velocity_at_without_velocities_raises():
    path = straight_line_path()
    with pytest.raises(ValueError):
        path.velocity_at(0.0)


def test_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        ReferencePath([0.0, 1.0], [0.0])
