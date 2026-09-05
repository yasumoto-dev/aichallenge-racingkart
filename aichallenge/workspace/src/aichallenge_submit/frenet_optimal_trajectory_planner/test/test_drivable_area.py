"""Tests for the drivable-area gate and the width table generator.

The gate is the only thing standing between a FOT candidate and the wall:
the C++ core filters on speed, acceleration, curvature and obstacle
collision only, and its lateral sampling range is a single constant. These
tests pin down the two properties that matter -- that an off-track point is
rejected, and that "left" means the same thing in the generator and in the
runtime check (a sign flip between them would silently allow exactly the
excursions the gate exists to stop).
"""
import csv
import importlib.util
import sys
from pathlib import Path

import pytest

from frenet_optimal_trajectory_planner.drivable_area import TrackWidthTable

HALF_WIDTH = 0.65
MARGIN = 0.2
# w=3.0 minus half width and margin.
ALLOWED = 3.0 - HALF_WIDTH - MARGIN


def straight_table(width=3.0, count=5, spacing=10.0):
    """Straight raceline along +x with a constant corridor half-width."""
    xs = [i * spacing for i in range(count)]
    ys = [0.0] * count
    return TrackWidthTable(xs, ys, [width] * count, [width] * count)


class _Point:
    """Minimal stand-in for a TrajectoryPoint."""

    def __init__(self, x, y):
        self.pose = type("Pose", (), {})()
        self.pose.position = type("Position", (), {})()
        self.pose.position.x = x
        self.pose.position.y = y


def violation_of(table, xs, ys, lo=0, hi=4):
    return table.first_violation(xs, ys, lo, hi, HALF_WIDTH, MARGIN)


def test_path_inside_corridor_is_accepted():
    table = straight_table()
    assert violation_of(table, [5.0, 15.0], [1.0, -1.0]) is None


def test_path_past_left_boundary_is_rejected():
    table = straight_table()
    result = violation_of(table, [5.0], [ALLOWED + 0.5])
    assert result is not None
    assert result["side"] == "left"
    assert result["offset"] == pytest.approx(ALLOWED + 0.5)
    assert result["allowed"] == pytest.approx(ALLOWED)


def test_path_past_right_boundary_is_rejected():
    table = straight_table()
    result = violation_of(table, [5.0], [-(ALLOWED + 0.5)])
    assert result is not None
    assert result["side"] == "right"
    assert result["offset"] == pytest.approx(ALLOWED + 0.5)


def test_offending_point_is_reported():
    table = straight_table()
    result = violation_of(table, [5.0, 15.0, 25.0], [0.0, 0.0, ALLOWED + 1.0])
    assert result is not None
    assert result["path_index"] == 2


def test_narrow_section_leaves_no_room():
    """Where the corridor is barely wider than the kart, nothing is allowed."""
    table = TrackWidthTable([0.0, 10.0, 20.0], [0.0] * 3,
                            [0.8] * 3, [0.8] * 3)
    assert table.first_violation([5.0], [0.5], 0, 2, HALF_WIDTH, MARGIN) \
        is not None
    assert table.first_violation([5.0], [0.0], 0, 2, HALF_WIDTH, MARGIN) \
        is None


def test_margin_is_interpolated_between_waypoints():
    """A corridor that narrows between two waypoints narrows continuously."""
    table = TrackWidthTable([0.0, 10.0], [0.0, 0.0], [4.0, 2.0], [4.0, 2.0])
    # Halfway along, w_left interpolates to 3.0 -> 2.15m usable.
    assert table.first_violation([5.0], [2.0], 0, 1, HALF_WIDTH, MARGIN) \
        is None
    assert table.first_violation([5.0], [2.3], 0, 1, HALF_WIDTH, MARGIN) \
        is not None


def test_empty_window_is_treated_as_a_violation():
    """No reference to check against must fail safe, not pass silently."""
    table = straight_table()
    assert violation_of(table, [5.0], [0.0], lo=2, hi=2) is not None


def test_point_beyond_the_window_is_rejected():
    """A point outside the window cannot pass, whatever its position."""
    table = straight_table()
    assert violation_of(table, [45.0], [3.0], lo=0, hi=2) is not None


def test_mismatch_detects_different_waypoint_count():
    table = straight_table(count=5)
    points = [_Point(i * 10.0, 0.0) for i in range(4)]
    assert "count differs" in table.mismatch_against(points, 0.5)


def test_mismatch_detects_shifted_raceline():
    table = straight_table(count=5)
    points = [_Point(i * 10.0, 0.0) for i in range(5)]
    assert table.mismatch_against(points, 0.5) is None
    points[3].pose.position.y = 2.0
    assert "waypoint 3" in table.mismatch_against(points, 0.5)


def test_from_csv_round_trip(tmp_path):
    path = tmp_path / "track_width.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "x", "y", "w_left", "w_right"])
        writer.writerow([0, 0.0, 0.0, 3.0, 3.0])
        writer.writerow([1, 10.0, 0.0, 3.0, 3.0])
    table = TrackWidthTable.from_csv(str(path))
    assert len(table) == 2
    assert table.first_violation([5.0], [0.0], 0, 1, HALF_WIDTH, MARGIN) \
        is None


def test_from_csv_rejects_missing_columns(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("index,x,y\n0,0.0,0.0\n1,10.0,0.0\n")
    with pytest.raises(ValueError):
        TrackWidthTable.from_csv(str(path))


def _load_generator():
    script = Path(__file__).resolve().parents[1] / "scripts" / \
        "generate_track_width.py"
    spec = importlib.util.spec_from_file_location("generate_track_width",
                                                  script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_lanelet2_map(path, left_y=3.0, right_y=-3.0, xs=(0, 10, 20, 30)):
    def nodes(start_id, y):
        return "".join(
            f'<node id="{start_id + i}" lat="0" lon="0">'
            f'<tag k="local_x" v="{x}"/><tag k="local_y" v="{y}"/>'
            f'<tag k="ele" v="0"/></node>'
            for i, x in enumerate(xs))

    def way(way_id, start_id):
        refs = "".join(f'<nd ref="{start_id + i}"/>' for i in range(len(xs)))
        return (f'<way id="{way_id}">{refs}'
                f'<tag k="type" v="line_thin"/>'
                f'<tag k="subtype" v="solid"/></way>')

    path.write_text(
        '<?xml version="1.0"?><osm version="0.6">'
        + nodes(100, left_y) + nodes(200, right_y)
        + way(1, 100) + way(2, 200)
        + '<relation id="10">'
          '<member type="way" role="left" ref="1"/>'
          '<member type="way" role="right" ref="2"/>'
          '<tag k="type" v="lanelet"/><tag k="subtype" v="road"/>'
          '</relation></osm>')


def _write_raceline(path, xs=(0, 10, 20, 30)):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "z", "x_quat", "y_quat", "z_quat",
                         "w_quat", "speed"])
        for x in xs:
            writer.writerow([float(x), 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 5.0])


def test_generator_measures_a_symmetric_corridor(tmp_path):
    generator = _load_generator()
    map_path = tmp_path / "lanelet2_map.osm"
    raceline_path = tmp_path / "raceline.csv"
    _write_lanelet2_map(map_path)
    _write_raceline(raceline_path)

    segments = generator.parse_lanelet_boundaries(str(map_path))
    points = generator.read_raceline(str(raceline_path))
    tangents = generator.heading_directions(points)
    w_left, w_right, missing = generator.measure_margins(points, tangents,
                                                          segments)

    assert not missing
    assert w_left == pytest.approx([3.0] * 4)
    assert w_right == pytest.approx([3.0] * 4)


def test_generator_left_matches_runtime_left(tmp_path):
    """An asymmetric corridor must reach the gate with its sides intact."""
    generator = _load_generator()
    map_path = tmp_path / "lanelet2_map.osm"
    raceline_path = tmp_path / "raceline.csv"
    _write_lanelet2_map(map_path, left_y=4.0, right_y=-1.0)
    _write_raceline(raceline_path)
    output = tmp_path / "track_width.csv"

    generator.main(["--map", str(map_path), "--raceline", str(raceline_path),
                    "--output", str(output)])
    table = TrackWidthTable.from_csv(str(output))

    assert table.w_left == pytest.approx([4.0] * 4)
    assert table.w_right == pytest.approx([1.0] * 4)
    # 4.0 - 0.85 = 3.15m of room to the left, 1.0 - 0.85 = 0.15m to the right.
    assert table.first_violation([15.0], [3.0], 0, 3, HALF_WIDTH, MARGIN) \
        is None
    assert table.first_violation([15.0], [-0.5], 0, 3, HALF_WIDTH, MARGIN) \
        is not None
