#!/usr/bin/env python3
"""ROS2 node: overtaking / following local planner around the FOT core.

Design note: the reference trajectory published by simple_trajectory_generator
is treated as read-only and is never republished in modified form. This node
only publishes an *overlay* trajectory on output/trajectory_avoidance, which
simple_pure_pursuit follows while it is non-empty and fresh, falling back to
the untouched reference otherwise (see its avoidance_timeout parameter).

What is decided where:

* overtake_planner.OvertakePlanner -- the behaviour (CRUISE / FOLLOW /
  OVERTAKE / RETURN), in Frenet coordinates of the lap. ROS-free.
* fot_wrapper / the vendored C++ core -- the lateral overtaking manoeuvre.
* raceline.Raceline -- lap geometry, drivable width, overtaking zones.

This node only converts messages, runs the planner at a fixed rate and
publishes the result plus RViz debug markers.
"""
import dataclasses
import math
import os

import numpy as np
import rclpy
from autoware_auto_planning_msgs.msg import Trajectory, TrajectoryPoint
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String
from v2x_msgs.msg import V2XVehiclePositionArray
from visualization_msgs.msg import Marker, MarkerArray

from frenet_optimal_trajectory_planner import fot_wrapper
from frenet_optimal_trajectory_planner.drivable_area import TrackWidthTable
from frenet_optimal_trajectory_planner.overtake_planner import (
    INACTIVE, OpponentTracker, OvertakePlanner, PlannerParams)
from frenet_optimal_trajectory_planner.raceline import (
    Raceline, load_overtake_zones)

_FOT_HP_NAMES = (
    "max_speed", "max_accel", "max_curvature", "d_road_w", "dt", "maxt",
    "mint", "d_t_s", "n_s_sample", "obstacle_clearance", "kd", "kv", "ka",
    "kj", "kt", "ko", "klat", "klon", "num_threads")

_STATE_COLORS = {
    "CRUISE": (0.6, 0.6, 0.6),
    "FOLLOW": (1.0, 0.6, 0.0),
    "OVERTAKE": (0.0, 0.8, 1.0),
    "RETURN": (0.6, 1.0, 0.3),
    INACTIVE: (0.4, 0.4, 0.4),
}


def _quaternion_to_yaw(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _yaw_to_quaternion(yaw: float):
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


class FrenetOptimalTrajectoryNode(Node):

    def __init__(self):
        super().__init__("frenet_optimal_trajectory_node")

        # ---- data files / safety gate ----
        self.enable_drivable_area_check_ = self.declare_parameter(
            "enable_drivable_area_check", True).value
        self.track_width_csv_ = self.declare_parameter(
            "track_width_csv", "").value
        self.track_width_match_tolerance_ = self.declare_parameter(
            "track_width_match_tolerance", 0.5).value
        # Optional second table measured from the occupancy grid map's
        # walls (scripts/generate_wall_width.py); the smaller width wins.
        self.wall_width_csv_ = self.declare_parameter(
            "wall_width_csv", "").value
        self.overtake_zones_csv_ = self.declare_parameter(
            "overtake_zones_csv", "").value
        self.planning_rate_ = self.declare_parameter(
            "planning_rate", 20.0).value
        # V2X reports every kart, possibly including this one. Anything
        # this close to the ego is treated as the ego itself.
        self.self_exclusion_radius_ = self.declare_parameter(
            "self_exclusion_radius", 1.0).value
        self.self_vehicle_id_ = self.declare_parameter(
            "self_vehicle_id",
            f"d{os.environ.get('ROS_DOMAIN_ID', '')}").value

        # ---- planner parameters: one ROS parameter per scalar field ----
        params = PlannerParams()
        for f in dataclasses.fields(PlannerParams):
            value = getattr(params, f.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            declared = self.declare_parameter(f.name, value).value
            setattr(params, f.name, type(value)(declared))
        hp = dict(params.fot_hyperparameters)
        for name in _FOT_HP_NAMES:
            hp[name] = type(hp[name])(
                self.declare_parameter(name, hp[name]).value)
        params.fot_hyperparameters = hp
        self._params = params

        self._trajectory = None
        self._odometry = None
        self._raceline = None
        self._planner = None
        self._tracker = None
        self._track_width = None
        self._zones_marker = None
        self._disabled_reason = None

        self._wall_width = None
        if self.enable_drivable_area_check_:
            self._track_width = self._load_track_width(self.track_width_csv_)
            if self.wall_width_csv_:
                self._wall_width = self._load_track_width(
                    self.wall_width_csv_)
            if self._track_width is None:
                self._disabled_reason = (
                    "drivable area check is enabled but the track width "
                    "table could not be loaded")
        else:
            self.get_logger().warn(
                "drivable area check is DISABLED; avoidance paths are not "
                "checked against the track boundary and may leave the track.")

        trajectory_qos = rclpy.qos.QoSProfile(
            depth=1,
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
            durability=rclpy.qos.DurabilityPolicy.VOLATILE,
        )
        self._sub_trajectory = self.create_subscription(
            Trajectory, "input/trajectory", self._on_trajectory, trajectory_qos)
        self._sub_v2x = self.create_subscription(
            V2XVehiclePositionArray, "input/v2x_vehicle_positions",
            self._on_v2x, 10)
        self._sub_kinematics = self.create_subscription(
            Odometry, "input/kinematics", self._on_kinematics, 10)
        self._pub_trajectory_avoidance = self.create_publisher(
            Trajectory, "output/trajectory_avoidance", trajectory_qos)
        self._pub_state = self.create_publisher(String, "~/debug/state", 10)
        self._pub_markers = self.create_publisher(
            MarkerArray, "~/debug/markers", 1)
        self._timer = self.create_timer(1.0 / self.planning_rate_,
                                        self._on_timer)

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def _load_track_width(self, path):
        if not path:
            self.get_logger().error("track width csv parameter is empty")
            return None
        try:
            table = TrackWidthTable.from_csv(path)
        except (OSError, ValueError) as exc:
            self.get_logger().error(f"failed to load width table: {exc}")
            return None
        self.get_logger().info(
            f"loaded width table: {path} ({len(table)} waypoints)")
        return table

    def _build_raceline(self, msg: Trajectory):
        points = msg.points
        xs = [p.pose.position.x for p in points]
        ys = [p.pose.position.y for p in points]
        vs = [p.longitudinal_velocity_mps for p in points]
        w_left = w_right = None
        if self._track_width is not None:
            mismatch = self._track_width.mismatch_against(
                points, self.track_width_match_tolerance_)
            if mismatch is not None:
                self._disabled_reason = (
                    f"track width table does not match the reference "
                    f"trajectory ({mismatch}); regenerate it from the "
                    f"raceline this launch actually uses")
                self.get_logger().error(self._disabled_reason)
                return
            w_left = self._track_width.w_left
            w_right = self._track_width.w_right
            if self._wall_width is not None:
                wall_mismatch = self._wall_width.mismatch_against(
                    points, self.track_width_match_tolerance_)
                if wall_mismatch is None:
                    w_left = np.minimum(w_left, self._wall_width.w_left)
                    w_right = np.minimum(w_right, self._wall_width.w_right)
                    self.get_logger().info(
                        "using min(lanelet, wall) track widths")
                else:
                    self.get_logger().warn(
                        f"wall width table ignored: {wall_mismatch}")
        raceline = Raceline(xs, ys, vs, w_left, w_right)

        zones = []
        if self.overtake_zones_csv_:
            try:
                zones = load_overtake_zones(self.overtake_zones_csv_)
                raceline.set_zones_from_waypoints(zones)
            except (OSError, ValueError) as exc:
                self.get_logger().error(
                    f"failed to load overtaking zones: {exc}; overtaking is "
                    f"disabled (the kart will only follow)")
                zones = []
        else:
            self.get_logger().warn(
                "overtake_zones_csv is empty: overtaking is disabled, the "
                "kart will only follow")

        self._raceline = raceline
        self._tracker = OpponentTracker(raceline)
        self._planner = OvertakePlanner(
            raceline, self._params, fot_wrapper.run_fot,
            logger=lambda m: self.get_logger().info(f"[state] {m}"))
        self._zones_marker = self._make_zone_markers(msg.header.frame_id)
        self.get_logger().info(
            f"raceline loop: waypoints {raceline.loop_start}.."
            f"{raceline.loop_end} ({raceline.length:.1f} m); "
            f"overtaking zones (waypoints): {zones}")

    # ------------------------------------------------------------------
    # callbacks
    # ------------------------------------------------------------------
    def _on_trajectory(self, msg: Trajectory):
        # Read-only reference (see module docstring).
        previous = self._trajectory
        self._trajectory = msg
        if not msg.points or self._disabled_reason is not None:
            return
        if self._raceline is None or previous is None or \
                len(previous.points) != len(msg.points):
            self._build_raceline(msg)

    def _on_kinematics(self, msg: Odometry):
        self._odometry = msg

    def _on_v2x(self, msg: V2XVehiclePositionArray):
        if self._tracker is None or self._odometry is None:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        ego = self._odometry.pose.pose.position
        for vehicle in msg.vehicles:
            if vehicle.vehicle_id == self.self_vehicle_id_:
                continue
            pos = vehicle.position
            if math.hypot(pos.x - ego.x, pos.y - ego.y) < \
                    self.self_exclusion_radius_:
                continue
            self._tracker.update(vehicle.vehicle_id, pos.x, pos.y, now)

    def _on_timer(self):
        if self._trajectory is None or not self._trajectory.points or \
                self._odometry is None:
            return
        if self._disabled_reason is not None or self._planner is None:
            self.get_logger().error(
                f"avoidance suppressed: {self._disabled_reason}",
                throttle_duration_sec=5.0)
            self._publish_empty()
            return

        now = self.get_clock().now().nanoseconds * 1e-9
        pose = self._odometry.pose.pose
        yaw = _quaternion_to_yaw(pose.orientation)
        ego_v = self._odometry.twist.twist.linear.x
        opponents = self._tracker.opponents(
            now, self._params.min_track_samples)
        out = self._planner.step(now, pose.position.x, pose.position.y, yaw,
                                 ego_v, opponents)

        if out.xs is None or len(out.xs) < 2:
            self._publish_empty()
        else:
            self._publish_path(out.xs, out.ys, out.yaws, out.speeds)

        state_msg = String()
        state_msg.data = (f"{out.state} [{out.source or '-'}] "
                          f"{out.target or ''} {out.reason}")
        self._pub_state.publish(state_msg)
        if out.source == "fallback":
            self.get_logger().warn(f"overtake: {out.reason}",
                                   throttle_duration_sec=1.0)
        self._publish_markers(out, pose.position)

    # ------------------------------------------------------------------
    # output
    # ------------------------------------------------------------------
    def _publish_empty(self):
        # An empty trajectory means "follow the reference" to
        # simple_pure_pursuit, exactly like a stale one; publishing it
        # explicitly keeps RViz honest.
        out = Trajectory()
        out.header = self._trajectory.header
        out.header.stamp = self.get_clock().now().to_msg()
        self._pub_trajectory_avoidance.publish(out)

    def _publish_path(self, xs, ys, yaws, speeds):
        z = self._trajectory.points[0].pose.position.z
        out = Trajectory()
        out.header = self._trajectory.header
        out.header.stamp = self.get_clock().now().to_msg()
        for x, y, yaw, v in zip(xs, ys, yaws, speeds):
            point = TrajectoryPoint()
            point.pose.position.x = float(x)
            point.pose.position.y = float(y)
            point.pose.position.z = z
            qx, qy, qz, qw = _yaw_to_quaternion(float(yaw))
            point.pose.orientation.x = qx
            point.pose.orientation.y = qy
            point.pose.orientation.z = qz
            point.pose.orientation.w = qw
            point.longitudinal_velocity_mps = max(float(v), 0.0)
            out.points.append(point)
        self._pub_trajectory_avoidance.publish(out)

    def _make_zone_markers(self, frame_id):
        markers = []
        z = self._trajectory.points[0].pose.position.z + 0.1
        for k, (s0, length) in enumerate(self._raceline.zones):
            m = Marker()
            m.header.frame_id = frame_id or "map"
            m.ns = "overtake_zones"
            m.id = k
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = 0.4
            m.color.r, m.color.g, m.color.b, m.color.a = 0.1, 0.9, 0.2, 0.5
            m.pose.orientation.w = 1.0
            for s in np.arange(0.0, length + 0.5, 1.0):
                x, y, _ = self._raceline.pose(s0 + s)
                m.points.append(Point(x=float(x), y=float(y), z=z))
            markers.append(m)
        return markers

    def _publish_markers(self, out, ego_pos):
        arr = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        for m in self._zones_marker or []:
            m.header.stamp = stamp
            arr.markers.append(m)
        text = Marker()
        text.header.frame_id = self._trajectory.header.frame_id or "map"
        text.header.stamp = stamp
        text.ns = "planner_state"
        text.id = 0
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position.x = ego_pos.x
        text.pose.position.y = ego_pos.y
        text.pose.position.z = ego_pos.z + 2.5
        text.pose.orientation.w = 1.0
        text.scale.z = 1.2
        r, g, b = _STATE_COLORS.get(out.state, (1.0, 1.0, 1.0))
        text.color.r, text.color.g, text.color.b, text.color.a = r, g, b, 1.0
        text.text = out.state + (f" ({out.target})" if out.target else "")
        arr.markers.append(text)
        self._pub_markers.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = FrenetOptimalTrajectoryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
