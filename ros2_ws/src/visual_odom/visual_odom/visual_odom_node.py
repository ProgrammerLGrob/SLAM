#!/usr/bin/env python3

KINECT_FRAME_ID = "kinect_depth"

MIN_DEPTH = 400
MAX_DEPTH = 5000

DESCRIPTOR_TOLERANCE = 50

F = 526.61
CU = 318.525
CV = 241.181


from math import atan2, cos, pi

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2

from cv_bridge import CvBridge
import cv2
import numpy as np
from visual_odom.landmark import kabash

class VisualOdom(Node):
    def __init__(self):
        super().__init__('visual_odom')
        self.bridge = CvBridge()
        # Initiate ORB detector
        self.orb = cv2.ORB_create()

        self.subscription_rgb = self.create_subscription(Image,'/serf01/nav_rgbd_1/rgb/image_raw', self.listener_rgb_callback, 10)
        self.subscription_depth = self.create_subscription(Image,'/serf01/nav_rgbd_1/depth/image_raw', self.listener_depth_callback, 10)

        self.publisher_keypoints_3d = self.create_publisher(PointCloud2, 'keypoint_3d', 10)
        self.publisher_3d = self.create_publisher(PointCloud2, 'points_3d', 10)

        self.landmarks = []
        self.frame_rgb_first = None
        self.frame_rgb_end = None
        self.frame_depth_first = None
        self.frame_depth_end = None
        self.frame_rgb = None
        self.counter = 0

    def listener_rgb_callback(self,msg):
        self.frame_rgb=self.bridge.imgmsg_to_cv2(msg,'bgr8')

        if self.frame_rgb_first is None:
            self.frame_rgb_first = self.frame_rgb
            self.kp_first, self.des_first = self.orb.detectAndCompute(self.frame_rgb_first, None)

        self.counter += 1

        if self.frame_rgb_end is None and self.counter > 205:
            self.frame_rgb_end = self.frame_rgb
            self.kp_end, self.des_end = self.orb.detectAndCompute(self.frame_rgb_end, None)
            
        
            

        

    def listener_depth_callback(self,msg):

        if self.frame_depth_first is None and self.frame_rgb_first is not None:
            self.frame_depth_first=self.bridge.imgmsg_to_cv2(msg,'passthrough')
            return
        
        if  self.frame_depth_first is not None and self.frame_depth_end is None and self.frame_rgb_end is not None:
            self.frame_depth_end=self.bridge.imgmsg_to_cv2(msg,'passthrough')
            
            #create BFMatcher object
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            # Match descriptors.
            matches = bf.match(self.des_first,self.des_end)
            # Sort them in the order of their distance.
            matches = sorted(matches, key = lambda x:x.distance)

            P_i = []
            Q_i = []

            valid_count = 0
            for m in range(len(matches)):
                if valid_count >= 2:  
                    break
                    
                #Matrix mit Koordinaten im kinect frame
                keypoint_first = self.kp_first[matches[m].queryIdx].pt
                keypoint_end = self.kp_end[matches[m].trainIdx].pt
                
                u_first, v_first = keypoint_first
                u_first = round(u_first)
                v_first = round(v_first)
                z_first = self.frame_depth_first[v_first, u_first]

                u_end, v_end = keypoint_end
                u_end = round(u_end)
                v_end = round(v_end)
                z_end = self.frame_depth_end[v_end, u_end]


                if matches[m].distance < DESCRIPTOR_TOLERANCE and z_first > MIN_DEPTH and z_first < MAX_DEPTH and z_end > MIN_DEPTH and z_end < MAX_DEPTH:
                    P_i.append(self.calculate_coordinate(u_first, v_first, z_first))
                    Q_i.append(self.calculate_coordinate(u_end, v_end, z_end))
                    valid_count += 1

            if len(P_i) >= 2:
                kabash_result = kabash(np.array(P_i), np.array(Q_i))
                self.get_logger().info(f"Rotation: {kabash_result[0]}, Translation: {kabash_result[1]}, Theta: {kabash_result[2]*180/pi} degrees")
            else:
                self.get_logger().warn(f"Nicht genug gültige Punkte: {len(P_i)} (mindestens 2 erforderlich)")
            
            cv2.imshow("End RGB Image", self.frame_rgb_end)
            cv2.waitKey(0)



    def calculate_coordinate(self, u, v, z):
        z = z
        y = z * (v-CV)/F
        x = z * (u-CU)/F

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
    node=VisualOdom()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()