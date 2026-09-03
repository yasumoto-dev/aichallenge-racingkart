#!/usr/bin/env python3
import math

import numpy as np
import rclpy
from autoware_auto_planning_msgs.msg import Trajectory, TrajectoryPoint
from multi_purpose_mpc_ros.v2x_vehicle_tracker import V2XVehicleTracker
from nav_msgs.msg import Odometry
from rclpy.node import Node
from v2x_msgs.msg import V2XVehiclePositionArray

from frenet_optimal_trajectory_planner.frenet_frame import ReferencePath
from frenet_optimal_trajectory_planner.trajectory_candidates import (
    generate_candidates,
    select_best_candidate,
)


def _yaw_to_quaternion(yaw: float):
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


class FrenetOptimalTrajectoryNode(Node):

    def __init__(self):
        super().__init__("frenet_optimal_trajectory_node")

        self.avoid_distance_threshold_ = self.declare_parameter(
            "avoid_distance_threshold", 5.0).value
        self.avoid_offset_max_ = self.declare_parameter(
            "avoid_offset_max", 3.0).value
        self.planning_horizon_ = self.declare_parameter(
            "planning_horizon", 3.0).value
        self.num_lateral_samples_ = self.declare_parameter(
            "num_lateral_samples", 7).value
        self.dt_sample_ = self.declare_parameter("dt_sample", 0.1).value
        self.safety_radius_ = self.declare_parameter("safety_radius", 1.2).value
        self.max_lateral_accel_ = self.declare_parameter(
            "max_lateral_accel", 4.0).value
        self.max_lateral_jerk_ = self.declare_parameter(
            "max_lateral_jerk", 20.0).value
        self.w_lateral_jerk_ = self.declare_parameter("w_lateral_jerk", 0.1).value
        self.w_lateral_offset_ = self.declare_parameter("w_lateral_offset", 1.0).value
        self.w_speed_deviation_ = self.declare_parameter(
            "w_speed_deviation", 1.0).value
        v_max_safety = self.declare_parameter("v_max_safety", 25.0).value
        position_jump_threshold = self.declare_parameter(
            "position_jump_threshold", 10.0).value

        self._tracker = V2XVehicleTracker(
            v_max_safety=v_max_safety,
            position_jump_threshold=position_jump_threshold,
            warn_callback=lambda msg: self.get_logger().warn(msg))

        self._trajectory = None
        self._reference_path = None
        self._odometry = None

        trajectory_qos = rclpy.qos.QoSProfile(
            depth=1,
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
            durability=rclpy.qos.DurabilityPolicy.VOLATILE,
        )

        self._sub_trajectory = self.create_subscription(
            Trajectory, "input/trajectory", self._on_trajectory, trajectory_qos)
        self._sub_v2x = self.create_subscription(
            V2XVehiclePositionArray, "input/v2x_vehicle_positions", self._on_v2x, 10)
        self._sub_kinematics = self.create_subscription(
            Odometry, "input/kinematics", self._on_kinematics, 10)
        self._pub_trajectory = self.create_publisher(
            Trajectory, "output/trajectory", trajectory_qos)

    def _on_trajectory(self, msg: Trajectory):
        self._trajectory = msg
        self._reference_path = self._build_reference_path(msg)
        self._pub_trajectory.publish(msg)

    def _on_kinematics(self, msg: Odometry):
        self._odometry = msg

    def _on_v2x(self, msg: V2XVehiclePositionArray):
        self._tracker.update(msg)

        if self._trajectory is None or self._reference_path is None or self._odometry is None:
            return

        avoided = self._plan_avoidance(msg)
        self._pub_trajectory.publish(avoided if avoided is not None else self._trajectory)

    @staticmethod
    def _build_reference_path(trajectory: Trajectory) -> ReferencePath:
        xs = [p.pose.position.x for p in trajectory.points]
        ys = [p.pose.position.y for p in trajectory.points]
        velocities = [p.longitudinal_velocity_mps for p in trajectory.points]
        return ReferencePath(xs, ys, velocities=velocities)

    def _plan_avoidance(self, v2x_msg: V2XVehiclePositionArray):
        ego_x = self._odometry.pose.pose.position.x
        ego_y = self._odometry.pose.pose.position.y
        ego_speed = self._odometry.twist.twist.linear.x

        s0, d0 = self._reference_path.to_frenet(ego_x, ego_y)

        relevant_vehicle_ids = [
            v.vehicle_id for v in v2x_msg.vehicles
            if math.hypot(v.position.x - ego_x, v.position.y - ego_y)
            < self.avoid_distance_threshold_
        ]
        if not relevant_vehicle_ids:
            return None

        num_samples = int(round(self.planning_horizon_ / self.dt_sample_)) + 1
        t_samples = [i * self.dt_sample_ for i in range(num_samples)]

        obstacle_predictions = {}
        for vehicle_id in relevant_vehicle_ids:
            world_positions = self._tracker.predict_positions(vehicle_id, t_samples)
            if not world_positions:
                continue
            obstacle_predictions[vehicle_id] = [
                self._reference_path.to_frenet(x, y) for (x, y) in world_positions
            ]

        target_speed = self._reference_path.velocity_at(s0)
        if target_speed <= 0.0:
            target_speed = ego_speed

        lateral_offsets = np.linspace(
            -self.avoid_offset_max_, self.avoid_offset_max_, self.num_lateral_samples_)
        candidates = generate_candidates(
            d0=d0, d0_dot=0.0, d0_ddot=0.0,
            s0=s0, s0_dot=ego_speed, s0_ddot=0.0,
            target_speed=target_speed,
            duration=self.planning_horizon_,
            lateral_offsets=lateral_offsets,
        )

        best_candidate, samples = select_best_candidate(
            candidates, self.dt_sample_, target_speed,
            max_lateral_accel=self.max_lateral_accel_,
            max_lateral_jerk=self.max_lateral_jerk_,
            avoid_offset_max=self.avoid_offset_max_,
            obstacle_predictions=obstacle_predictions,
            safety_radius=self.safety_radius_,
            w_lateral_jerk=self.w_lateral_jerk_,
            w_lateral_offset=self.w_lateral_offset_,
            w_speed_deviation=self.w_speed_deviation_,
        )
        if best_candidate is None:
            return None

        return self._splice_trajectory(s0, samples)

    def _splice_trajectory(self, s0: float, samples) -> Trajectory:
        candidate_points = []
        for sample in samples:
            x, y, heading = self._reference_path.to_cartesian(sample.s, sample.d)
            point = TrajectoryPoint()
            point.pose.position.x = x
            point.pose.position.y = y
            point.pose.position.z = self._trajectory.points[0].pose.position.z
            qx, qy, qz, qw = _yaw_to_quaternion(heading)
            point.pose.orientation.x = qx
            point.pose.orientation.y = qy
            point.pose.orientation.z = qz
            point.pose.orientation.w = qw
            point.longitudinal_velocity_mps = max(sample.s_dot, 0.0)
            candidate_points.append(point)

        s_final = samples[-1].s
        arc_lengths = self._reference_path.arc_lengths
        original_points = self._trajectory.points

        start_idx = len(arc_lengths)
        end_idx = len(arc_lengths)
        for i, s_i in enumerate(arc_lengths):
            if s_i >= s0 and start_idx == len(arc_lengths):
                start_idx = i
            if s_i > s_final:
                end_idx = i
                break

        output = Trajectory()
        output.header = self._trajectory.header
        output.points = (
            list(original_points[:start_idx])
            + candidate_points
            + list(original_points[end_idx:])
        )
        return output


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
