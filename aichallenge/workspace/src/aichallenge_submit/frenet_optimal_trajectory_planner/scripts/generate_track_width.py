#!/usr/bin/env python3
"""Precompute the drivable lateral margin of a raceline from a lanelet2 map.

The Frenet Optimal Trajectory core has no notion of a drivable area: the
only road information it accepts is the scalar pair max_road_width_l/r,
and is_valid_path() filters candidates on speed / acceleration / curvature
/ obstacle collision only.  A constant lateral bound cannot describe a
circuit whose width varies by a factor of ~3.7, so avoidance manoeuvres
could legally be placed outside the track.

This script measures, for every raceline waypoint, how far the track
boundary actually is on each side.  frenet_optimal_trajectory_node loads
the resulting table and rejects any candidate path that would put the
vehicle body past that boundary (see drivable_area.py).

The boundary is taken from the left/right linestrings of every lanelet in
the lanelet2 map -- the same map the running system loads via
lanelet2_map_loader -- so the table is only valid for the exact
(map, raceline) pair it was generated from.  The node verifies that pairing
at runtime by comparing the table's waypoint positions against the
reference trajectory it actually receives.

Both the map's local_x/local_y and the raceline CSV are already in the same
MGRS-projected frame, so no reprojection is needed here.

Usage:
    generate_track_width.py --map <lanelet2_map.osm> \
                            --raceline <raceline.csv> \
                            --output <track_width.csv>
"""
import argparse
import csv
import sys
import xml.etree.ElementTree as ET

import numpy as np


def parse_lanelet_boundaries(osm_path):
    """Return the boundary segments of every lanelet as an (N, 2, 2) array.

    Only ways referenced by a lanelet relation through role="left" or
    role="right" are collected, which excludes unrelated geometry such as
    the parking_space ways used for the garage.
    """
    root = ET.parse(osm_path).getroot()

    nodes = {}
    for node in root.findall("node"):
        tags = {t.get("k"): t.get("v") for t in node.findall("tag")}
        if "local_x" in tags and "local_y" in tags:
            nodes[node.get("id")] = (float(tags["local_x"]),
                                     float(tags["local_y"]))

    ways = {}
    for way in root.findall("way"):
        pts = [nodes[nd.get("ref")] for nd in way.findall("nd")
               if nd.get("ref") in nodes]
        ways[way.get("id")] = pts

    boundary_way_ids = set()
    for relation in root.findall("relation"):
        tags = {t.get("k"): t.get("v") for t in relation.findall("tag")}
        if tags.get("type") != "lanelet":
            continue
        for member in relation.findall("member"):
            if member.get("type") == "way" and \
                    member.get("role") in ("left", "right"):
                boundary_way_ids.add(member.get("ref"))

    segments = []
    for way_id in boundary_way_ids:
        pts = ways.get(way_id, [])
        segments.extend(zip(pts[:-1], pts[1:]))

    if not segments:
        raise ValueError(
            f"no lanelet boundary segments found in {osm_path}; is this a "
            "lanelet2 map with left/right roles?")
    return np.array(segments, dtype=np.float64)


def read_raceline(csv_path):
    """Return the raceline waypoints as an (N, 2) array."""
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) < 2:
        raise ValueError(f"{csv_path} has fewer than 2 waypoints")
    return np.array([[float(r["x"]), float(r["y"])] for r in rows],
                    dtype=np.float64)


def heading_directions(points):
    """Unit tangent at every waypoint, by central difference.

    The runtime check measures the lateral offset against the raceline
    *segment* it projects onto, so the tangent -- not the CSV quaternion --
    is the consistent definition of "left" between the two.  A central
    difference keeps the two within a fraction of a segment's turn angle.
    """
    d = np.empty_like(points)
    d[1:-1] = points[2:] - points[:-2]
    d[0] = points[1] - points[0]
    d[-1] = points[-1] - points[-2]
    norm = np.hypot(d[:, 0], d[:, 1])
    norm[norm == 0.0] = 1.0
    return d / norm[:, None]


def measure_margins(points, tangents, segments):
    """Distance from each waypoint to the nearest boundary on each side.

    Returns (w_left, w_right); a side with no boundary at all yields 0.0,
    which disables lateral motion to that side rather than allowing an
    unbounded one.
    """
    a = segments[:, 0, :]
    b = segments[:, 1, :]
    ab = b - a
    length_sq = (ab ** 2).sum(axis=1)
    length_sq[length_sq == 0.0] = 1e-12

    w_left = np.zeros(len(points))
    w_right = np.zeros(len(points))
    missing = []

    for i, p in enumerate(points):
        # Closest point on every boundary segment (clamped to the segment).
        t = np.clip(((p - a) * ab).sum(axis=1) / length_sq, 0.0, 1.0)
        closest = a + t[:, None] * ab
        v = closest - p
        dist = np.hypot(v[:, 0], v[:, 1])
        # Positive cross product => the boundary lies left of the heading.
        side = tangents[i, 0] * v[:, 1] - tangents[i, 1] * v[:, 0]

        left = dist[side > 0.0]
        right = dist[side < 0.0]
        w_left[i] = left.min() if left.size else 0.0
        w_right[i] = right.min() if right.size else 0.0
        if not left.size or not right.size:
            missing.append(i)

    return w_left, w_right, missing


def write_table(output_path, points, w_left, w_right):
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "x", "y", "w_left", "w_right"])
        for i, (p, wl, wr) in enumerate(zip(points, w_left, w_right)):
            writer.writerow([i, f"{p[0]:.6f}", f"{p[1]:.6f}",
                             f"{wl:.4f}", f"{wr:.4f}"])


def print_summary(points, w_left, w_right, half_width, margin):
    usable_l = np.maximum(w_left - half_width - margin, 0.0)
    usable_r = np.maximum(w_right - half_width - margin, 0.0)
    tightest = np.argsort(np.minimum(usable_l, usable_r))[:8]

    print(f"  waypoints             : {len(points)}")
    print(f"  margin to boundary  L : min {w_left.min():.2f} "
          f"median {np.median(w_left):.2f} max {w_left.max():.2f} m")
    print(f"  margin to boundary  R : min {w_right.min():.2f} "
          f"median {np.median(w_right):.2f} max {w_right.max():.2f} m")
    print(f"  track width (L+R)     : min {(w_left + w_right).min():.2f} "
          f"median {np.median(w_left + w_right):.2f} "
          f"max {(w_left + w_right).max():.2f} m")
    print(f"  usable offset (half width {half_width} + margin {margin}) :")
    print(f"      L min {usable_l.min():.2f} median {np.median(usable_l):.2f} m")
    print(f"      R min {usable_r.min():.2f} median {np.median(usable_r):.2f} m")
    print(f"  tightest waypoints    : "
          f"{sorted(int(i) for i in tightest)}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Precompute per-waypoint drivable lateral margins for "
                    "frenet_optimal_trajectory_node.")
    parser.add_argument("--map", required=True,
                        help="lanelet2 .osm map (the one the system loads)")
    parser.add_argument("--raceline", required=True,
                        help="raceline CSV used by simple_trajectory_generator")
    parser.add_argument("--output", required=True,
                        help="destination track width CSV")
    parser.add_argument("--half-width", type=float, default=0.65,
                        help="vehicle half width, for the summary only "
                             "(default: 0.65, racing_kart_description)")
    parser.add_argument("--margin", type=float, default=0.2,
                        help="safety margin, for the summary only "
                             "(default: 0.2)")
    args = parser.parse_args(argv)

    segments = parse_lanelet_boundaries(args.map)
    points = read_raceline(args.raceline)
    tangents = heading_directions(points)
    w_left, w_right, missing = measure_margins(points, tangents, segments)

    write_table(args.output, points, w_left, w_right)

    print(f"wrote {args.output}")
    print(f"  map                   : {args.map} "
          f"({len(segments)} boundary segments)")
    print(f"  raceline              : {args.raceline}")
    print_summary(points, w_left, w_right, args.half_width, args.margin)

    if missing:
        print(f"  WARNING: {len(missing)} waypoint(s) have no boundary on at "
              f"least one side and were set to 0.0 (no lateral room): "
              f"{missing[:10]}{'...' if len(missing) > 10 else ''}",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
