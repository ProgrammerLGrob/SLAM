#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
import csv
from math import *
import os
from visual_odom.constants import *

class OdomImuCsvWriter(Node):

    def __init__(self):
        super().__init__('odom_imu_csv_writer')

        # Parameter
        self.declare_parameter(
            'csv_path',
            os.path.expanduser('~/Projekt/SLAM/odom_imu_data.csv')
        )
        self.csv_path = self.get_parameter('csv_path').value

        # Aktuelle Werte
        self.wheel_x = 0.0
        self.wheel_y = 0.0
        self.imu_theta = 0.0
        self.odom_visual_x = 0.0
        self.odom_visual_y = 0.0
        self.odom_visual_theta = 0.0

        # CSV-Datei
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow(['timestamp', 'wheel_x', 'wheel_y', 'imu_theta', 'odom_visual_x', 'odom_visual_y', 'odom_visual_theta'])  # Header

        # Subscriber
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

        self.get_logger().info(f"Schreibe CSV nach: {self.csv_path}")

    def odom_callback(self, msg):
        self.wheel_x = msg.pose.pose.position.x
        self.wheel_y = msg.pose.pose.position.y

        timestamp = (
            msg.header.stamp.sec +
            msg.header.stamp.nanosec * 1e-9
        )

        self.writer.writerow([timestamp, self.wheel_x, self.wheel_y, self.imu_theta, self.odom_visual_x, self.odom_visual_y, self.odom_visual_theta])
        self.csv_file.flush()

    def imu_callback(self, msg):
        q = msg.orientation
        self.imu_theta = self.normalize_angle(self.quaternion_to_yaw(q)-(3/2)*pi)

    def visual_odom_callback(self, msg):
        self.odom_visual_x = msg.pose.pose.position.x
        self.odom_visual_y = msg.pose.pose.position.y
        self.odom_visual_theta = self.quaternion_to_yaw(msg.pose.pose.orientation)

    def quaternion_to_yaw(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return atan2(siny_cosp, cosy_cosp)

    def destroy_node(self):
        self.csv_file.close()
        super().destroy_node()

    def normalize_angle(self,angle: float) -> float:
        if angle > pi:
            angle -= 2*pi
        elif angle < -pi:
            angle += 2*pi

        return angle    


def main():
    rclpy.init()
    node = OdomImuCsvWriter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()