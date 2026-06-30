#!/usr/bin/env python3

KINECT_FRAME_ID = "kinect_depth"

MIN_DEPTH = 400
MAX_DEPTH = 5000

F = 526.61
CU = 318.525
CV = 241.181


from math import atan2, cos

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2

from cv_bridge import CvBridge
import cv2
import numpy as np
from geometry_msgs.msg import Vector3

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.bridge = CvBridge()
        # Initiate ORB detector
        self.orb = cv2.ORB_create()
        self.subscription_rgb = self.create_subscription(Image,'/serf01/nav_rgbd_1/rgb/image_raw', self.listener_rgb_callback, 10)
        self.subscription_depth = self.create_subscription(Image,'/serf01/nav_rgbd_1/depth/image_raw', self.listener_depth_callback, 10)

        self.publisher_keypoints_3d = self.create_publisher(PointCloud2, 'keypoint_3d', 10)
        self.publisher_3d = self.create_publisher(PointCloud2, 'points_3d', 10)

    def listener_rgb_callback(self,msg):
        self.frame_rgb=self.bridge.imgmsg_to_cv2(msg,'bgr8')
        #cv2.imshow("RGB Image",frame)

        # find the keypoints with ORB and compute the descriptors with ORB
        self.kp, des = self.orb.detectAndCompute(self.frame_rgb, None)
        

    def listener_depth_callback(self,msg):
        if not hasattr(self, 'frame_rgb'):
            return
        
        self.frame_depth=self.bridge.imgmsg_to_cv2(msg,'passthrough')

        # draw only keypoints location,not size and orientation
        valid_kp = []
        calculated_point_coordinates = []
        calculated_keypoint_coordinates = []

        img = self.frame_rgb.copy()

        for pix_u in range(self.frame_depth.shape[1]):
            for pix_v in range(self.frame_depth.shape[0]):
                depth_value = self.frame_depth[pix_v, pix_u]
                if depth_value > MIN_DEPTH and depth_value < MAX_DEPTH:
                    calculated_point_coordinates.append(self.calculate_coordinate(pix_u, pix_v, depth_value))

        
        for p in self.kp:
            u, v = p.pt
            u = round(u)
            v = round(v)
            depth_value = self.frame_depth[v, u]
                        
            if depth_value > MIN_DEPTH and depth_value < MAX_DEPTH:
                img = cv2.putText(img, f"{depth_value/1000:.2f}m", (u+5, v+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 1, cv2.LINE_AA)
                valid_kp.append(p)

                calculated_keypoint_coordinates.append(self.calculate_coordinate(u, v, depth_value))
              
        
        self.kp = valid_kp
        
        self.publish_pointcloud(calculated_point_coordinates, publisher=self.publisher_3d)
        self.publish_pointcloud(calculated_keypoint_coordinates, publisher=self.publisher_keypoints_3d)
        
        img = cv2.drawKeypoints(img, self.kp, None, color=(0,255,0), flags=0)
        cv2.imshow("Image with text", img)
        cv2.waitKey(1)



    def calculate_coordinate(self, u, v, depth_value):
        z = depth_value
        y = depth_value * (v-CV)/F
        x = depth_value * (u-CU)/F

        p3d_vector = np.array([x, y, z])/1000.0
        return p3d_vector



    def publish_pointcloud(self, calc_points, publisher):
        msg = point_cloud2.create_cloud_xyz32(
            header=self.get_header(),
            points=calc_points
        )

        publisher.publish(msg)
        self.get_logger().info("PointCloud gesendet!")


    def get_header(self):
        from std_msgs.msg import Header
        h = Header()
        h.stamp = self.get_clock().now().to_msg()
        h.frame_id = KINECT_FRAME_ID
        return h

    


def main():
    rclpy.init()
    node=ImageSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()