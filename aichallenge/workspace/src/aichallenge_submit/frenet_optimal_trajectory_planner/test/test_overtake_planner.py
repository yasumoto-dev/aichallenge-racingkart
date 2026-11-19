"""Tests for the lap geometry and the overtaking behaviour layer.

They use the real citycircuit raceline / width table / zones shipped with
the package, because the bugs they guard against are properties of this
track's shape (parallel sections a few metres apart, a lap seam, narrow
stretches).
"""
import csv
import math
from pathlib import Path

import numpy as np
import pytest

from frenet_optimal_trajectory_planner import fot_wrapper
from frenet_optimal_trajectory_planner.drivable_area import TrackWidthTable
from frenet_optimal_trajectory_planner.overtake_planner import (
    CRUISE, FOLLOW, OVERTAKE, Opponent, OvertakePlanner, PlannerParams)
from frenet_optimal_trajectory_planner.raceline import (
    Raceline, load_overtake_zones)

PKG = Path(__file__).resolve().parents[1]
RACELINE = (PKG.parent / "simple_trajectory_generator" / "data" /
            "raceline_awsim_30km_from_garage.csv")


@pytest.fixture(scope="module")
def raceline():
    if not RACELINE.exists():
        pytest.skip(f"{RACELINE} not found")
    rows = [list(map(float, r)) for r in
            list(csv.reader(open(RACELINE)))[1:]]
    tw = TrackWidthTable.from_csv(str(PKG / "data" / "track_width.csv"))
    ww = TrackWidthTable.from_csv(
        str(PKG / "data" / "track_width_walls.csv"))
    # same as the node: the narrower of lanelet and wall widths
    rl = Raceline([r[0] for r in rows], [r[1] for r in rows],
                  [r[7] for r in rows], np.minimum(tw.w_left, ww.w_left),
                  np.minimum(tw.w_right, ww.w_right))
    rl.set_zones_from_waypoints(
        load_overtake_zones(str(PKG / "data" / "overtake_zones.csv")))
    return rl


def opponent_at(rl, s, d, v, vid="d2"):
    x, y, _ = rl.to_xy(s, d)
    return Opponent(vid, x, y, rl.wrap_s(s), d, v, True)


def test_loop_and_seam(raceline):
    # waypoint 130 duplicates waypoint 12 on this raceline
    assert (raceline.loop_start, raceline.loop_end) == (12, 130)
    assert raceline.delta_s(2.0, raceline.length - 3.0) == pytest.approx(5.0)
    s, d = raceline.project(*raceline.to_xy(100.0, 1.5)[:2], s_hint=99.0)
    assert s == pytest.approx(100.0, abs=0.1)
    assert d == pytest.approx(1.5, abs=0.05)


def test_kart_on_parallel_section_is_ignored(raceline):
    """Euclidean-near but far along the lap must not cause any action.

    This was the "avoidance path in a strange place" bug.
    """
    planner = OvertakePlanner(raceline, PlannerParams(), run_fot=None)
    ego_s = raceline.waypoint_s(20)
    ex, ey, eyaw = raceline.pose(ego_s)
    # find a raceline point > 30 m away along the lap but < 15 m in space
    best = None
    for k in range(int(raceline.length)):
        s = float(k)
        if abs(raceline.delta_s(s, ego_s)) < 30.0:
            continue
        x, y, _ = raceline.pose(s)
        dist = math.hypot(x - ex, y - ey)
        if best is None or dist < best[0]:
            best = (dist, s)
    assert best[0] < 15.0
    out = planner.step(0.0, ex, ey, eyaw, 8.0,
                       [opponent_at(raceline, best[1], 0.0, 3.0)])
    assert out.state == CRUISE
    assert out.xs is None


def test_kart_behind_is_ignored(raceline):
    planner = OvertakePlanner(raceline, PlannerParams(), run_fot=None)
    ego_s = raceline.waypoint_s(20)
    ex, ey, eyaw = raceline.pose(ego_s)
    out = planner.step(0.0, ex, ey, eyaw, 8.0,
                       [opponent_at(raceline, ego_s - 5.0, 0.0, 9.0)])
    assert out.state == CRUISE


def test_follow_outside_zone_slows_down(raceline):
    planner = OvertakePlanner(raceline, PlannerParams(), run_fot=None)
    ego_s = raceline.waypoint_s(90)          # not in any zone
    assert raceline.zone_remaining(ego_s) is None
    ex, ey, eyaw = raceline.pose(ego_s)
    out = planner.step(0.0, ex, ey, eyaw, 8.3,
                       [opponent_at(raceline, ego_s + 8.0, 0.0, 3.0)])
    assert out.state == FOLLOW
    assert out.speeds is not None
    # far below the 8.3 m/s reference: lead speed + gentle gap closing
    assert max(out.speeds) < 4.0


def test_follow_stops_behind_stopped_kart(raceline):
    planner = OvertakePlanner(raceline, PlannerParams(), run_fot=None)
    ego_s = raceline.waypoint_s(90)
    ex, ey, eyaw = raceline.pose(ego_s)
    out = planner.step(0.0, ex, ey, eyaw, 1.0,
                       [opponent_at(raceline, ego_s + 2.064 + 0.4, 0.0, 0.0)])
    assert out.state == FOLLOW
    assert max(out.speeds) == pytest.approx(0.0)


def test_overtake_inside_zone_goes_to_one_side(raceline):
    planner = OvertakePlanner(raceline, PlannerParams(),
                              run_fot=fot_wrapper.run_fot)
    ego_s = raceline.waypoint_s(15)          # start of the back straight
    ex, ey, eyaw = raceline.pose(ego_s)
    out = planner.step(0.0, ex, ey, eyaw, 8.3,
                       [opponent_at(raceline, ego_s + 9.0, 0.0, 3.0)])
    assert out.state == OVERTAKE
    assert out.source == "fot"
    # the end of the manoeuvre is clearly to one side of the other kart
    s_end, d_end = raceline.project(out.xs[-1], out.ys[-1], ego_s)
    assert abs(d_end) >= PlannerParams().lateral_pass - 0.05
    # and the published speed lets the kart actually accelerate past
    assert min(out.speeds) > 7.0
