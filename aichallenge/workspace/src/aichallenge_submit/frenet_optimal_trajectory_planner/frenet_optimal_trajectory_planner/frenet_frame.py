"""Arc-length parameterized reference path and Frenet (s, d) conversion.

Pure Python, no rclpy/ROS message dependency, so it can be unit tested
without a running ROS graph.
"""

import math
from typing import List, Optional, Sequence, Tuple


class ReferencePath:
    """A polyline reference path indexed by cumulative arc length ``s``."""

    def __init__(
        self,
        xs: Sequence[float],
        ys: Sequence[float],
        velocities: Optional[Sequence[float]] = None,
    ):
        if len(xs) != len(ys) or len(xs) < 2:
            raise ValueError("ReferencePath requires at least 2 matching (x, y) points")
        if velocities is not None and len(velocities) != len(xs):
            raise ValueError("velocities must have the same length as xs/ys")

        self._xs = list(xs)
        self._ys = list(ys)
        self._velocities = list(velocities) if velocities is not None else None

        self._s: List[float] = [0.0]
        for i in range(1, len(self._xs)):
            seg = math.hypot(self._xs[i] - self._xs[i - 1], self._ys[i] - self._ys[i - 1])
            self._s.append(self._s[-1] + seg)

    @property
    def length(self) -> float:
        return self._s[-1]

    @property
    def arc_lengths(self) -> List[float]:
        return self._s

    def _segment_index_for_s(self, s: float) -> int:
        s = min(max(s, 0.0), self._s[-1])
        for i in range(len(self._s) - 1):
            if self._s[i] <= s <= self._s[i + 1]:
                return i
        return len(self._s) - 2

    def to_cartesian(self, s: float, d: float) -> Tuple[float, float, float]:
        """Returns ``(x, y, heading)`` for the given ``(s, d)``."""
        idx = self._segment_index_for_s(s)
        x0, y0 = self._xs[idx], self._ys[idx]
        x1, y1 = self._xs[idx + 1], self._ys[idx + 1]
        seg_len = self._s[idx + 1] - self._s[idx]
        heading = math.atan2(y1 - y0, x1 - x0)

        ratio = 0.0 if seg_len <= 1e-9 else (min(max(s, self._s[idx]), self._s[idx + 1]) - self._s[idx]) / seg_len
        base_x = x0 + ratio * (x1 - x0)
        base_y = y0 + ratio * (y1 - y0)

        normal_x = -math.sin(heading)
        normal_y = math.cos(heading)
        return base_x + d * normal_x, base_y + d * normal_y, heading

    def to_frenet(self, x: float, y: float) -> Tuple[float, float]:
        """Projects ``(x, y)`` onto the path, returning ``(s, signed_d)``.

        ``d`` is positive to the left of the path's direction of travel.
        Searches every segment (the path here is short, ~100-300 points),
        so this returns the globally nearest projection rather than one
        local to a previous estimate.
        """
        best_dist_sq = math.inf
        best = (0.0, 0.0)
        for i in range(len(self._xs) - 1):
            x0, y0 = self._xs[i], self._ys[i]
            x1, y1 = self._xs[i + 1], self._ys[i + 1]
            dx, dy = x1 - x0, y1 - y0
            seg_len_sq = dx * dx + dy * dy
            if seg_len_sq <= 1e-12:
                continue

            t = ((x - x0) * dx + (y - y0) * dy) / seg_len_sq
            t = min(max(t, 0.0), 1.0)
            proj_x = x0 + t * dx
            proj_y = y0 + t * dy
            dist_sq = (x - proj_x) ** 2 + (y - proj_y) ** 2

            if dist_sq < best_dist_sq:
                seg_len = math.sqrt(seg_len_sq)
                heading = math.atan2(dy, dx)
                normal_x = -math.sin(heading)
                normal_y = math.cos(heading)
                signed_d = (x - proj_x) * normal_x + (y - proj_y) * normal_y
                best_dist_sq = dist_sq
                best = (self._s[i] + t * seg_len, signed_d)
        return best

    def velocity_at(self, s: float) -> float:
        if self._velocities is None:
            raise ValueError("ReferencePath was built without per-point velocities")
        idx = self._segment_index_for_s(s)
        seg_len = self._s[idx + 1] - self._s[idx]
        ratio = 0.0 if seg_len <= 1e-9 else (min(max(s, self._s[idx]), self._s[idx + 1]) - self._s[idx]) / seg_len
        v0, v1 = self._velocities[idx], self._velocities[idx + 1]
        return v0 + ratio * (v1 - v0)
