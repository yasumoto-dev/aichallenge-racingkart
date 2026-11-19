"""Closed-loop raceline geometry: Frenet projection, widths, zones, paths.

Why this exists
---------------
The node used to reason in *Euclidean* distance: every kart within 15 m of
the ego became an obstacle. On this circuit 109 of 131 raceline waypoints
have a different part of the track (more than 30 m away along the lap)
within 15 m, so karts on a parallel straight -- or simply behind the ego --
triggered avoidance manoeuvres in places where nothing was in the way.

Everything here works in Frenet coordinates of the *lap* instead:

* ``s`` -- arc length along the closed raceline loop, in ``[0, length)``.
* ``d`` -- signed lateral offset, positive to the LEFT of travel (the same
  convention as the FOT core and the track width table).

so "ahead", "behind" and "beside" have their racing meaning, and the lap
wraps around correctly at the start/finish seam.

The published raceline starts with a garage out-lap that joins the loop
(waypoint 130 coincides with waypoint 12 on citycircuit). Only the loop
part is used here; while the ego is still on the out-lap it is far from the
loop and the planner simply stays inactive.
"""
import csv
import math

import numpy as np
from scipy.interpolate import CubicSpline


def _detect_loop(xs, ys, tolerance=0.05):
    """Return (start, end) such that waypoint ``end`` duplicates ``start``.

    The loop is then waypoints ``start .. end-1``, closed back onto
    ``start``. If no duplicate exists the whole list is treated as a loop.
    """
    pts = np.stack([xs, ys], axis=1)
    for j in range(2, len(pts)):
        dist = np.hypot(*(pts[:j - 1] - pts[j]).T)
        i = int(np.argmin(dist))
        if dist[i] < tolerance:
            return i, j
    return 0, len(pts)


class Raceline:
    """Dense, periodic representation of the raceline loop."""

    def __init__(self, xs, ys, speeds, w_left=None, w_right=None,
                 resolution=0.2):
        xs = np.asarray(xs, dtype=np.float64)
        ys = np.asarray(ys, dtype=np.float64)
        speeds = np.asarray(speeds, dtype=np.float64)
        start, end = _detect_loop(xs, ys)
        self.loop_start = start
        self.loop_end = end

        lx = xs[start:end]
        ly = ys[start:end]
        lv = speeds[start:end]
        # Chord-length parameter, including the closing segment back to the
        # first loop point, so the periodic spline closes smoothly.
        seg = np.hypot(np.diff(np.r_[lx, lx[0]]), np.diff(np.r_[ly, ly[0]]))
        t = np.r_[0.0, np.cumsum(seg)]
        spline = CubicSpline(t, np.stack([np.r_[lx, lx[0]],
                                          np.r_[ly, ly[0]]], axis=1),
                             bc_type="periodic")

        # Resample densely and re-measure the true arc length.
        n = max(int(t[-1] / resolution), 16)
        tt = np.linspace(0.0, t[-1], n, endpoint=False)
        xy = spline(tt)
        dxy = spline(tt, 1)
        step = np.hypot(np.diff(np.r_[xy[:, 0], xy[0, 0]]),
                        np.diff(np.r_[xy[:, 1], xy[0, 1]]))
        self.s = np.r_[0.0, np.cumsum(step)[:-1]]
        self.length = float(np.sum(step))
        self.x = xy[:, 0]
        self.y = xy[:, 1]
        self.yaw = np.arctan2(dxy[:, 1], dxy[:, 0])

        # Arc length of every original loop waypoint, by the same mapping.
        scale = self.length / t[-1]
        self._wp_s = t[:-1] * scale
        self._wp_index = np.arange(start, end)

        def periodic_interp(values):
            return np.interp(self.s, np.r_[self._wp_s, self.length],
                             np.r_[values, values[0]])

        self.v_ref = periodic_interp(lv)
        if w_left is not None and w_right is not None:
            self.w_left = periodic_interp(np.asarray(w_left)[start:end])
            self.w_right = periodic_interp(np.asarray(w_right)[start:end])
        else:
            self.w_left = None
            self.w_right = None

        self._zones = []
        self.set_body_half_length(1.2)

    def set_body_half_length(self, half_length):
        step = self.length / len(self.s)
        self._body_cells = max(0, int(math.ceil(half_length / step)))

    # ------------------------------------------------------------------
    # basic helpers
    # ------------------------------------------------------------------
    def wrap_s(self, s):
        return s % self.length

    def delta_s(self, s_to, s_from):
        """Signed along-track distance from ``s_from`` to ``s_to``.

        Result is in ``[-length/2, length/2)``: positive means ``s_to`` is
        ahead.
        """
        half = 0.5 * self.length
        return (s_to - s_from + half) % self.length - half

    def _index(self, s):
        s = self.wrap_s(s)
        i = int(np.searchsorted(self.s, s, side="right") - 1)
        return max(0, min(i, len(self.s) - 1))

    def pose(self, s):
        """(x, y, yaw) on the raceline at arc length ``s`` (interpolated)."""
        s = self.wrap_s(s)
        i = self._index(s)
        j = (i + 1) % len(self.s)
        s_i = self.s[i]
        s_j = self.s[j] if j != 0 else self.length
        r = (s - s_i) / max(s_j - s_i, 1e-9)
        x = self.x[i] + r * (self.x[j] - self.x[i])
        y = self.y[i] + r * (self.y[j] - self.y[i])
        dyaw = math.atan2(math.sin(self.yaw[j] - self.yaw[i]),
                          math.cos(self.yaw[j] - self.yaw[i]))
        return x, y, self.yaw[i] + r * dyaw

    def to_xy(self, s, d):
        x, y, yaw = self.pose(s)
        return x - d * math.sin(yaw), y + d * math.cos(yaw), yaw

    def speed_at(self, s):
        return float(self.v_ref[self._index(s)])

    def project(self, x, y, s_hint=None, window=15.0):
        """Frenet coordinates (s, d) of a point.

        With ``s_hint`` only raceline within ``window`` metres of the hint is
        searched. That matters on this circuit: parallel sections lie a few
        metres apart, and a global nearest-point search can latch onto the
        wrong one when a kart is near the edge of the track.
        """
        if s_hint is None:
            idx = np.arange(len(self.s))
        else:
            ds = self.delta_s(self.s, s_hint)
            idx = np.nonzero(np.abs(ds) <= window)[0]
            if idx.size == 0:
                idx = np.arange(len(self.s))
        dist2 = (self.x[idx] - x) ** 2 + (self.y[idx] - y) ** 2
        i = int(idx[int(np.argmin(dist2))])
        yaw = self.yaw[i]
        c, sn = math.cos(yaw), math.sin(yaw)
        rx, ry = x - self.x[i], y - self.y[i]
        along = rx * c + ry * sn
        d = -rx * sn + ry * c
        return self.wrap_s(self.s[i] + along), d

    # ------------------------------------------------------------------
    # drivable width
    # ------------------------------------------------------------------
    def allowed_offsets(self, s, half_width, margin):
        """(max offset to the left, max offset to the right), both >= 0.

        How far the *centre* of a kart may be from the raceline at ``s``
        without its body crossing the measured track boundary. The width is
        the minimum over the kart's own length (``body_half_length`` either
        side of ``s``): at a corner exit the width can halve within 4 m,
        and checking only at the centre let the front of the kart reach the
        wall (observed in AWSIM after waypoint 48).
        """
        if self.w_left is None:
            return math.inf, math.inf
        i = self._index(s)
        k = self._body_cells
        idx = np.arange(i - k, i + k + 1) % len(self.s)
        return (max(0.0, float(self.w_left[idx].min()) - half_width - margin),
                max(0.0, float(self.w_right[idx].min()) - half_width -
                    margin))

    def min_allowed_offsets(self, s_from, length, half_width, margin,
                            step=1.0):
        """Tightest (left, right) allowance over ``[s_from, s_from+length]``."""
        left, right = math.inf, math.inf
        for k in range(int(max(length, 0.0) / step) + 1):
            al, ar = self.allowed_offsets(s_from + k * step, half_width,
                                          margin)
            left = min(left, al)
            right = min(right, ar)
        return left, right

    def check_path(self, xs, ys, s_hint, half_width, margin):
        """First path point outside the drivable area, or None.

        Points are projected in order, each using the previous one as the
        hint, so the check follows the path along the lap and cannot latch
        onto a neighbouring section of track.
        """
        hint = s_hint
        for k, (px, py) in enumerate(zip(xs, ys)):
            s, d = self.project(float(px), float(py), hint, window=8.0)
            hint = s
            left, right = self.allowed_offsets(s, half_width, margin)
            if d > left:
                return {"path_index": k, "side": "left", "offset": d,
                        "allowed": left, "s": s}
            if -d > right:
                return {"path_index": k, "side": "right", "offset": -d,
                        "allowed": right, "s": s}
        return None

    # ------------------------------------------------------------------
    # overtaking zones
    # ------------------------------------------------------------------
    def waypoint_s(self, index):
        """Loop arc length of an original raceline waypoint index."""
        index = int(index)
        if index >= self.loop_end:
            index = self.loop_start + (index - self.loop_end)
        if index < self.loop_start:
            raise ValueError(
                f"waypoint {index} is on the out-lap (loop starts at "
                f"{self.loop_start})")
        return float(self._wp_s[index - self.loop_start])

    def set_zones_from_waypoints(self, pairs):
        """Overtaking zones given as (start_wp, end_wp) raceline indices."""
        zones = []
        for a, b in pairs:
            s0 = self.waypoint_s(a)
            s1 = self.waypoint_s(b)
            length = (s1 - s0) % self.length
            zones.append((s0, length))
        self._zones = zones

    @property
    def zones(self):
        return list(self._zones)

    def zone_remaining(self, s):
        """Metres left in the overtaking zone containing ``s`` (None: none)."""
        for s0, length in self._zones:
            into = (s - s0) % self.length
            if into <= length:
                return length - into
        return None

    # ------------------------------------------------------------------
    # path generation along the raceline
    # ------------------------------------------------------------------
    def offset_path(self, s_from, length, d_start, d_end, blend,
                    step=0.5, clamp=None, anchor_s=None):
        """Path following the raceline at an offset that blends smoothly.

        The offset goes from ``d_start`` to ``d_end`` over the first
        ``blend`` metres and then stays at ``d_end``. The profile is an
        ease-out (1 - (1 - r)^2): it already moves at the start and arrives
        tangentially, so even when a new plan has to start from the kart's
        current offset, pure pursuit's short lookahead point already moves
        sideways (a smoothstep, flat at the start, did not: observed in
        AWSIM).

        ``anchor_s`` is where the blend started (default ``s_from``): the
        lateral plan is fixed in s and only *evaluated* from the kart's
        current position, so the controller corrects deviations from it
        instead of the plan following the kart's drift.

        ``clamp=(half_width, margin)`` keeps every point inside the
        drivable area (the offset is limited point by point), so these
        paths can never be the thing that puts the kart into a wall.

        Returns arrays (x, y, yaw, s, d).
        """
        n = max(int(length / step), 2)
        ss = s_from + np.arange(n + 1) * step
        start = s_from if anchor_s is None else \
            s_from - self.delta_s(s_from, anchor_s)
        r = np.clip((ss - start) / max(blend, 1e-6), 0.0, 1.0)
        smooth = 1.0 - (1.0 - r) ** 2
        dd = d_start + (d_end - d_start) * smooth
        if clamp is not None:
            for k, s in enumerate(ss):
                left, right = self.allowed_offsets(s, *clamp)
                dd[k] = min(left, max(-right, dd[k]))
        xs = np.empty_like(ss)
        ys = np.empty_like(ss)
        for k, (s, d) in enumerate(zip(ss, dd)):
            xs[k], ys[k], _ = self.to_xy(s, d)
        yaws = np.arctan2(np.gradient(ys), np.gradient(xs))
        return xs, ys, yaws, self.wrap_s(ss), dd

    def sample_points(self, s_from, s_to, spacing):
        """Raceline points every ``spacing`` metres (used as FOT waypoints)."""
        n = max(int((s_to - s_from) / spacing), 1)
        pts = [self.pose(s_from + k * spacing)[:2] for k in range(n + 1)]
        return np.array(pts, dtype=np.float64)


def load_overtake_zones(path):
    """Read ``start_waypoint,end_waypoint`` rows (comments with ``#``)."""
    pairs = []
    with open(path, newline="") as f:
        rows = [line for line in f if line.strip() and
                not line.lstrip().startswith("#")]
    reader = csv.DictReader(rows)
    required = {"start_waypoint", "end_waypoint"}
    if reader.fieldnames is None or not required.issubset(reader.fieldnames):
        raise ValueError(f"{path}: expected columns {sorted(required)}, got "
                         f"{reader.fieldnames}")
    for row in reader:
        pairs.append((int(row["start_waypoint"]), int(row["end_waypoint"])))
    return pairs


__all__ = ["Raceline", "load_overtake_zones"]
