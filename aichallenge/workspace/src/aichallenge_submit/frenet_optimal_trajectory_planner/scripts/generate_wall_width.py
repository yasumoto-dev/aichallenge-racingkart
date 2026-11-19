#!/usr/bin/env python3
"""Measure the distance from each raceline waypoint to the physical walls.

track_width.csv is measured from the lanelet2 lane boundary, but on
citycircuit the walls of the occupancy grid map lie up to ~0.8 m inside it
(waypoints 29-31, 38-48, 60-61, 74, ...). In AWSIM the kart hit such a wall
at waypoint 48-49 while the lanelet table still showed 2 m of room. This
script ray-casts along the left/right normal of every raceline waypoint in
an occupancy grid map (ROS map_server format) and writes a table in the same
format as track_width.csv; the node takes the smaller of the two.

Usage:
  generate_wall_width.py --map occupancy_grid_map.yaml \
      --raceline raceline.csv --output track_width_walls.csv
"""
import argparse
import csv
import math
import os

import numpy as np
import yaml
from PIL import Image


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--map", required=True, help="map_server yaml")
    ap.add_argument("--raceline", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-range", type=float, default=8.0)
    ap.add_argument("--step", type=float, default=0.05)
    args = ap.parse_args()

    meta = yaml.safe_load(open(args.map))
    image_path = os.path.join(os.path.dirname(args.map), meta["image"])
    img = np.asarray(Image.open(image_path)).astype(np.float64) / 255.0
    if meta.get("negate", 0):
        img = 1.0 - img
    occupied = (1.0 - img) > float(meta["occupied_thresh"])
    res = float(meta["resolution"])
    ox, oy = float(meta["origin"][0]), float(meta["origin"][1])
    height, width = occupied.shape

    def is_wall(x, y):
        col = int((x - ox) / res)
        row = height - 1 - int((y - oy) / res)
        if not (0 <= row < height and 0 <= col < width):
            return True
        return bool(occupied[row, col])

    rows = list(csv.reader(open(args.raceline)))[1:]
    xs = np.array([float(r[0]) for r in rows])
    ys = np.array([float(r[1]) for r in rows])

    with open(args.output, "w", newline="") as f:
        out = csv.writer(f)
        out.writerow(["index", "x", "y", "w_left", "w_right"])
        for i in range(len(xs)):
            j, k = min(i + 1, len(xs) - 1), max(i - 1, 0)
            yaw = math.atan2(ys[j] - ys[k], xs[j] - xs[k])
            # Positive = left of travel, same as track_width.csv.
            nx, ny = -math.sin(yaw), math.cos(yaw)

            def ray(sign):
                for t in np.arange(0.0, args.max_range, args.step):
                    if is_wall(xs[i] + sign * nx * t, ys[i] + sign * ny * t):
                        return float(t)
                return args.max_range

            out.writerow([i, f"{xs[i]:.6f}", f"{ys[i]:.6f}",
                          f"{ray(+1):.4f}", f"{ray(-1):.4f}"])
    print(f"wrote {args.output} ({len(xs)} waypoints)")


if __name__ == "__main__":
    main()
