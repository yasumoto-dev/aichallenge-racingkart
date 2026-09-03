#!/usr/bin/env python3
"""ROS2 node wrapping the erdos-project FrenetOptimalTrajectory planner.

Design note: the reference trajectory published by simple_trajectory_generator
is treated as read-only and is never republished in modified form. This node
only publishes an *avoidance overlay* trajectory, and only while it actually
has a valid avoidance solution; simple_pure_pursuit falls back to the
untouched reference trajectory whenever this topic goes quiet (see
simple_pure_pursuit's avoidance_timeout parameter).
"""
import math

import numpy as np
import rclpy
from autoware_auto_planning_msgs.msg import Trajectory, TrajectoryPoint
from nav_msgs.msg import Odometry
from rclpy.node import Node
from v2x_msgs.msg import V2XVehiclePositionArray

from frenet_optimal_trajectory_planner import fot_wrapper


def _quaternion_to_yaw(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                       1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _yaw_to_quaternion(yaw: float):
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


class FrenetOptimalTrajectoryNode(Node):

    def __init__(self):
        super().__init__("frenet_optimal_trajectory_node")

        self.avoid_distance_threshold_ = self.declare_parameter(
            "avoid_distance_threshold", 15.0).value
        self.obstacle_clearance_ = self.declare_parameter(
            "obstacle_clearance", 0.6).value
        self.waypoint_lookahead_distance_ = self.declare_parameter(
            "waypoint_lookahead_distance", 90.0).value
        self.waypoint_lookbehind_points_ = self.declare_parameter(
            "waypoint_lookbehind_points", 5).value
        # The published path's first point must land within this distance of
        # the ego's actual position; otherwise a candidate is discarded (see
        # _plan_avoidance). Guards against a bad local spline fit sending the
        # car toward the wrong place.
        self.max_start_position_error_ = self.declare_parameter(
            "max_start_position_error", 2.0).value

        self.max_speed_ = self.declare_parameter("max_speed", 15.0).value
        self.max_accel_ = self.declare_parameter("max_accel", 5.0).value
        self.max_curvature_ = self.declare_parameter("max_curvature", 0.7).value
        self.max_road_width_l_ = self.declare_parameter(
            "max_road_width_l", 1.5).value
        self.max_road_width_r_ = self.declare_parameter(
            "max_road_width_r", 1.5).value
        self.d_road_w_ = self.declare_parameter("d_road_w", 0.5).value
        self.dt_ = self.declare_parameter("dt", 0.1).value
        self.maxt_ = self.declare_parameter("maxt", 2.0).value
        self.mint_ = self.declare_parameter("mint", 1.0).value
        self.d_t_s_ = self.declare_parameter("d_t_s", 1.0).value
        self.n_s_sample_ = self.declare_parameter("n_s_sample", 1.0).value
        self.kd_ = self.declare_parameter("kd", 1.0).value
        self.kv_ = self.declare_parameter("kv", 0.1).value
        self.ka_ = self.declare_parameter("ka", 0.1).value
        self.kj_ = self.declare_parameter("kj", 0.1).value
        self.kt_ = self.declare_parameter("kt", 0.1).value
        self.ko_ = self.declare_parameter("ko", 0.5).value
        self.klat_ = self.declare_parameter("klat", 1.0).value
        self.klon_ = self.declare_parameter("klon", 1.0).value
        self.num_threads_ = self.declare_parameter("num_threads", 0).value

        # waypoint_lookahead_distance must cover the farthest any sampled
        # candidate can travel (maxt * fastest sampled speed), or the local
        # spline gets asked to extrapolate past its supplied waypoints and
        # candidates start failing outright (this is what silently breaks
        # "no avoidance path is generated" if lookahead is set too small
        # without also shortening maxt/max_speed). Enforce it here instead
        # of relying on the launch file value being kept in sync by hand.
        min_required_lookahead = self.maxt_ * (
            self.max_speed_ + self.d_t_s_ * self.n_s_sample_)
        if self.waypoint_lookahead_distance_ < min_required_lookahead:
            self.get_logger().warn(
                f"waypoint_lookahead_distance "
                f"({self.waypoint_lookahead_distance_}m) is less than "
                f"maxt*max_speed ({min_required_lookahead:.1f}m); raising it "
                f"to {min_required_lookahead:.1f}m to avoid spline "
                "extrapolation. To use a shorter lookahead, reduce maxt "
                "and/or max_speed instead.")
            self.waypoint_lookahead_distance_ = min_required_lookahead

        self._trajectory = None
        self._odometry = None
        self._prev_s = None

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

    def _on_trajectory(self, msg: Trajectory):
        # Read-only reference: stored for use as FOT waypoints, never
        # republished (see module docstring).
        self._trajectory = msg

    def _on_kinematics(self, msg: Odometry):
        self._odometry = msg

    def _on_v2x(self, msg: V2XVehiclePositionArray):
        if self._trajectory is None or not self._trajectory.points or \
                self._odometry is None:
            return

        ego_x = self._odometry.pose.pose.position.x
        ego_y = self._odometry.pose.pose.position.y

        obstacles = self._build_obstacles(msg, ego_x, ego_y)
        if not obstacles:
            self._publish_empty()
            return

        self._plan_avoidance(ego_x, ego_y, obstacles)

    def _publish_empty(self):
        # Explicitly publish a pointless (0-point) Trajectory whenever we
        # decide not to avoid, rather than just not publishing. simple_
        # pure_pursuit already treats an empty points list as "no avoidance
        # active" (same as a stale/absent message), so this doesn't change
        # control behavior -- but it keeps the RViz display (and anyone
        # else watching this topic) honest in real time instead of showing
        # a stale path from the last time avoidance actually succeeded.
        out = Trajectory()
        out.header = self._trajectory.header
        out.header.stamp = self.get_clock().now().to_msg()
        self._pub_trajectory_avoidance.publish(out)

    def _build_obstacles(self, v2x_msg: V2XVehiclePositionArray,
                          ego_x: float, ego_y: float):
        obstacles = []
        r = self.obstacle_clearance_
        for vehicle in v2x_msg.vehicles:
            vx = vehicle.position.x
            vy = vehicle.position.y
            if math.hypot(vx - ego_x, vy - ego_y) < self.avoid_distance_threshold_:
                obstacles.append([vx - r, vy - r, vx + r, vy + r])
        return obstacles

    def _nearest_index(self, ego_x: float, ego_y: float) -> int:
        points = self._trajectory.points
        best_idx = 0
        best_dist = math.inf
        for i, p in enumerate(points):
            d = math.hypot(p.pose.position.x - ego_x, p.pose.position.y - ego_y)
            if d < best_dist:
                best_dist = d
                best_idx = i
        return best_idx

    def _local_waypoints(self, nearest_idx: int) -> np.ndarray:
        points = self._trajectory.points
        start = max(0, nearest_idx - self.waypoint_lookbehind_points_)

        end = nearest_idx
        acc_dist = 0.0
        while end + 1 < len(points) and acc_dist < self.waypoint_lookahead_distance_:
            p0 = points[end].pose.position
            p1 = points[end + 1].pose.position
            acc_dist += math.hypot(p1.x - p0.x, p1.y - p0.y)
            end += 1

        wp = np.array([[p.pose.position.x, p.pose.position.y]
                        for p in points[start:end + 1]])
        return wp

    def _plan_avoidance(self, ego_x: float, ego_y: float, obstacles: list):
        nearest_idx = self._nearest_index(ego_x, ego_y)
        wp = self._local_waypoints(nearest_idx)
        if wp.shape[0] < 4:
            self.get_logger().warn(
                "not enough local waypoints to run FOT",
                throttle_duration_sec=1.0)
            self._publish_empty()
            return

        target_speed = self._trajectory.points[nearest_idx].longitudinal_velocity_mps
        if target_speed <= 0.0:
            target_speed = self._odometry.twist.twist.linear.x

        yaw = _quaternion_to_yaw(self._odometry.pose.pose.orientation)
        v_body_x = self._odometry.twist.twist.linear.x
        v_body_y = self._odometry.twist.twist.linear.y
        vx = v_body_x * math.cos(yaw) - v_body_y * math.sin(yaw)
        vy = v_body_x * math.sin(yaw) + v_body_y * math.cos(yaw)

        initial_conditions = {
            "ps": self._prev_s if self._prev_s is not None else 0.0,
            "target_speed": target_speed,
            "pos": np.array([ego_x, ego_y]),
            "vel": np.array([vx, vy]),
            "wp": wp,
            "obs": np.array(obstacles, dtype=np.float64),
        }
        hyperparameters = {
            "max_speed": self.max_speed_,
            "max_accel": self.max_accel_,
            "max_curvature": self.max_curvature_,
            "max_road_width_l": self.max_road_width_l_,
            "max_road_width_r": self.max_road_width_r_,
            "d_road_w": self.d_road_w_,
            "dt": self.dt_,
            "maxt": self.maxt_,
            "mint": self.mint_,
            "d_t_s": self.d_t_s_,
            "n_s_sample": self.n_s_sample_,
            "obstacle_clearance": self.obstacle_clearance_,
            "kd": self.kd_,
            "kv": self.kv_,
            "ka": self.ka_,
            "kj": self.kj_,
            "kt": self.kt_,
            "ko": self.ko_,
            "klat": self.klat_,
            "klon": self.klon_,
            "num_threads": self.num_threads_,
        }

        (x_path, y_path, speeds, _ix, _iy, _iyaw, _d, _s, speeds_x, speeds_y,
         params, _costs, success) = fot_wrapper.run_fot(
            initial_conditions, hyperparameters)

        if not success or len(x_path) < 2:
            self.get_logger().warn(
                "FOT: no feasible avoidance path found",
                throttle_duration_sec=1.0)
            self._publish_empty()
            return

        # Sanity check: the first path point represents "now" and must be
        # close to the ego's actual position. A sparse/tightly-curving local
        # waypoint window can throw off the internal spline fit (and thus
        # the Frenet-frame projection of the ego position) enough to produce
        # a path that starts somewhere else entirely -- publishing that
        # would send the car off toward a wrong (and possibly off-track)
        # target. Discard and fall back to the reference trajectory instead.
        start_error = math.hypot(x_path[0] - ego_x, y_path[0] - ego_y)
        if start_error > self.max_start_position_error_:
            self.get_logger().warn(
                f"FOT: discarding path, start point is {start_error:.2f}m "
                "from ego position (bad local spline fit?)",
                throttle_duration_sec=1.0)
            self._publish_empty()
            return

        self._prev_s = params["s"]
        self._publish_avoidance(x_path, y_path, speeds, speeds_x, speeds_y)

    def _publish_avoidance(self, x_path, y_path, speeds, speeds_x, speeds_y):
        z = self._trajectory.points[0].pose.position.z
        prev_yaw = _quaternion_to_yaw(self._odometry.pose.pose.orientation)

        out = Trajectory()
        out.header = self._trajectory.header
        out.header.stamp = self.get_clock().now().to_msg()

        for i in range(len(x_path)):
            speed = float(speeds[i])
            if speed > 1e-2:
                yaw = math.atan2(speeds_y[i], speeds_x[i])
            elif i + 1 < len(x_path):
                yaw = math.atan2(y_path[i + 1] - y_path[i],
                                  x_path[i + 1] - x_path[i])
            else:
                yaw = prev_yaw
            prev_yaw = yaw

            point = TrajectoryPoint()
            point.pose.position.x = float(x_path[i])
            point.pose.position.y = float(y_path[i])
            point.pose.position.z = z
            qx, qy, qz, qw = _yaw_to_quaternion(yaw)
            point.pose.orientation.x = qx
            point.pose.orientation.y = qy
            point.pose.orientation.z = qz
            point.pose.orientation.w = qw
            point.longitudinal_velocity_mps = max(speed, 0.0)
            out.points.append(point)

        self._pub_trajectory_avoidance.publish(out)


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
