#!/usr/bin/env python3

"""!
@file csv_creator_node.py
@package visual_odom.csv_creator_node
@brief ROS 2 node that logs odometry and IMU measurements to a unified CSV file.
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
import csv
from math import pi, atan2
import os
from visual_odom.constants import VISUAL_ODOM_MSG_TOPIC

class OdomImuCsvWriter(Node):
    """!
    @brief ROS 2 Node responsible for subscribing to localization data topics and 
           logging state measurements into a unified CSV file for analysis.
    """

    def __init__(self):
        """!
        @brief Initializes the data logging node, configures ROS parameters, files, and subscriptions.
        """
        super().__init__('odom_imu_csv_writer')

        # Parameters
        self.declare_parameter(
            'csv_path',
            os.path.expanduser('~/Projekt/SLAM/odom_imu_data.csv')
        )
        self.csv_path = self.get_parameter('csv_path').value  ##< Absolute file path where the generated CSV data log will be stored.

        # Current state values
        self.wheel_x = 0.0  ##< Latest recorded global X position from wheel odometry in meters.
        self.wheel_y = 0.0  ##< Latest recorded global Y position from wheel odometry in meters.
        self.imu_theta = 0.0  ##< Latest recorded planar yaw heading orientation from the IMU sensor in radians.
        self.odom_visual_x = 0.0  ##< Latest recorded global X position calculated by visual odometry in meters.
        self.odom_visual_y = 0.0  ##< Latest recorded global Y position calculated by visual odometry in meters.
        self.odom_visual_theta = 0.0  ##< Latest recorded planar yaw heading calculated by visual odometry in radians.

        # CSV File Configuration
        self.csv_file = open(self.csv_path, 'w', newline='')  ##< Internal file descriptor handling active stream writes to disk.
        self.writer = csv.writer(self.csv_file)  ##< CSV writer interface mapping programmatic arrays to formatted textual table records.
        self.writer.writerow(['timestamp', 'wheel_x', 'wheel_y', 'imu_theta', 'odom_visual_x', 'odom_visual_y', 'odom_visual_theta'])  ##< File structure line defining data labels inside column headers.

        # Subscriptions
        self.create_subscription(
            Odometry,
            '/serf01/odometry/wheel',
            self.odom_callback,
            10
        )

        self.create_subscription(
            Imu,
            '/serf01/odometry/imu',
            self.imu_callback,
            10
        )

        self.create_subscription(
            Odometry,
            VISUAL_ODOM_MSG_TOPIC,
            self.visual_odom_callback,
            10
        )

        self.get_logger().info(f"Writing CSV data to target path: {self.csv_path}")

    def odom_callback(self, msg: Odometry):
        """!
        @brief Processes standard wheel odometry data updates, calculates float timestamps, 
               and flushes synchronized state snapshots down into the CSV table log.

        @param msg The incoming wheel odometry message package (nav_msgs.msg.Odometry).
        """
        self.wheel_x = msg.pose.pose.position.x
        self.wheel_y = msg.pose.pose.position.y

        timestamp = (
            msg.header.stamp.sec +
            msg.header.stamp.nanosec * 1e-9
        )

        self.writer.writerow([timestamp, self.wheel_x, self.wheel_y, self.imu_theta, self.odom_visual_x, self.odom_visual_y, self.odom_visual_theta])
        self.csv_file.flush()

    def imu_callback(self, msg: Imu):
        """!
        @brief Receives structural IMU orientation frames, extracts yaw orientation data, 
               and applies a rigid frame offset transformation to align it with reference norms.

        @param msg Incoming raw IMU reading container (sensor_msgs.msg.Imu).
        """
        q = msg.orientation
        self.imu_theta = self.normalize_angle(self.quaternion_to_yaw(q) - (3/2) * pi)

    def visual_odom_callback(self, msg: Odometry):
        """!
        @brief Processes updates published by the visual odometry algorithm pipeline, 
               unpacking 2D translation states and rotation yaw angles.

        @param msg The incoming camera odometry message packet (nav_msgs.msg.Odometry).
        """
        self.odom_visual_x = msg.pose.pose.position.x
        self.odom_visual_y = msg.pose.pose.position.y
        self.odom_visual_theta = self.quaternion_to_yaw(msg.pose.pose.orientation)

    def quaternion_to_yaw(self, q) -> float:
        """!
        @brief Calculates planar yaw angle heading out of 3D spatial rotation quaternion parameters.

        @param q Input object storing spatial orientation data (geometry_msgs.msg.Quaternion).
        @return Planar heading representation mapped in radians (float).
        """
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return atan2(siny_cosp, cosy_cosp)

    def destroy_node(self):
        """!
        @brief Safely terminates subscription ties and guarantees that data buffers flush completely before closing the file streams.
        """
        self.csv_file.close()
        super().destroy_node()

    def normalize_angle(self, angle: float) -> float:
        """!
        @brief Wraps an input angle parameter into standard bounded circle space [-pi, pi].

        @param angle Arbitrary orientation scale input in radians.
        @return Wrapped equivalent within circular system boundaries (float).
        """
        if angle > pi:
            angle -= 2 * pi
        elif angle < -pi:
            angle += 2 * pi

        normalized_angle = angle
        return normalized_angle    


def main():
    """!
    @brief Starts the infrastructure runtime framework layer, instantiates OdomImuCsvWriter execution loops, and ensures safe process cleanups.
    """
    rclpy.init()
    node = OdomImuCsvWriter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()