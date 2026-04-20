#!/usr/bin/env python3

KINECT_FRAME_ID = "kinect_depth"

MIN_DEPTH = 400
MAX_DEPTH = 5000

DESCRIPTOR_TOLERANCE = 50
RANSAC_EVALUATION_TOLERANCE = 30 #in mm
RANSAC_ITERATION = 2000
RANSAC_SAMPLE_SIZE = 3

F = 526.61
CU = 318.525
CV = 241.181


from math import atan2, cos, pi
import random

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2

from cv_bridge import CvBridge
import cv2
import numpy as np
from visual_odom.landmark import landmark, kabash

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
        self.frame_depth = None
        self.frame_rgb = None
        self.counter = 0

    def listener_rgb_callback(self,msg):
        self.frame_rgb=self.bridge.imgmsg_to_cv2(msg,'bgr8')
        self.kp = self.orb.detect(self.frame_rgb, None)

        if self.frame_depth is None:
            return


        for pix_u in range(self.frame_depth.shape[1]):
            for pix_v in range(self.frame_depth.shape[0]):
                depth_value = self.frame_depth[pix_v, pix_u]    


        valid_kp = []
        for p in self.kp:
            u, v = p.pt
            u = int(round(u))
            v = int(round(v))

            depth_value = self.frame_depth[v, u]

            if depth_value > MIN_DEPTH and depth_value < MAX_DEPTH:
                valid_kp.append(p) 

        if self.frame_rgb_first is None:
            self.frame_rgb_first = self.frame_rgb
            self.frame_depth_first = self.frame_depth
            self.kp_first,self.des_first = self.orb.compute(self.frame_rgb, valid_kp)

        self.counter += 1

        if self.frame_rgb_end is None and self.counter > 100:
            self.frame_rgb_end = self.frame_rgb
            self.frame_depth_end = self.frame_depth
            self.kp_end,self.des_end = self.orb.compute(self.frame_rgb, valid_kp)
        else:
            return

        #create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # Match descriptors.
        matches = bf.match(self.des_first,self.des_end)

        ransac_result = ransac(RANSAC_EVALUATION_TOLERANCE, RANSAC_ITERATION, RANSAC_SAMPLE_SIZE, matches, self.kp_first, self.kp_end, self.frame_depth_first, self.frame_depth_end)

        self.get_logger().info(f"RANSAC Result: Rotationsmatrix {ransac_result[0]}, Translationsvektor {ransac_result[1]}, Theta in ° {ransac_result[2]*180/pi}")
        
        cv2.imshow("End RGB Image", self.frame_rgb_end)
        cv2.waitKey(0)
            

    def listener_depth_callback(self,msg):
        self.frame_depth=self.bridge.imgmsg_to_cv2(msg,'passthrough')


def ransac(tolerance, iteration, n_samples, matches, kp_first, kp_end, frame_depth_first, frame_depth_end):
    sorted_matches = sorted(matches, key = lambda x:x.distance)
    samples = sorted_matches[:n_samples] #Take the best n matches as start samples
    new_matches = []

    P_i = []
    Q_i = []

    R = np.eye(2)
    t = np.zeros((2,1))
    theta = 0


    for iter in range(iteration):
        for m in range(n_samples):            
            #Matrix mit Koordinaten im kinect frame
            keypoint_first = kp_first[samples[m].queryIdx].pt
            keypoint_end = kp_end[samples[m].trainIdx].pt
            
            u_first, v_first = keypoint_first
            u_first = round(u_first)
            v_first = round(v_first)
            z_first = frame_depth_first[v_first, u_first]

            u_end, v_end = keypoint_end
            u_end = round(u_end)
            v_end = round(v_end)
            z_end = frame_depth_end[v_end, u_end]

            coor = calculate_coordinate(u_first, v_first, z_first)
            P_i.append([coor[0], coor[2]])
            coor = calculate_coordinate(u_end, v_end, z_end)
            Q_i.append([coor[0], coor[2]])

        R, t, theta  = kabash(np.array(P_i), np.array(Q_i))


        for m in range(len(matches)):            
            #Matrix mit Koordinaten im kinect frame
            keypoint_first = kp_first[matches[m].queryIdx].pt
            keypoint_end = kp_end[matches[m].trainIdx].pt
            
            u_first, v_first = keypoint_first
            u_first = round(u_first)
            v_first = round(v_first)
            z_first = frame_depth_first[v_first, u_first]

            u_end, v_end = keypoint_end
            u_end = round(u_end)
            v_end = round(v_end)
            z_end = frame_depth_end[v_end, u_end]

            coor_first = calculate_coordinate(u_first, v_first, z_first)
            corr_first = np.array([coor_first[0], coor_first[2]])
            coor_end = calculate_coordinate(u_end, v_end, z_end)
            coor_end = np.array([coor_end[0], coor_end[2]])


            e = np.linalg.norm(corr_first - (R*coor_end + t))
            
            if e < tolerance:
                new_matches.append(matches[m])

        if(len(new_matches) <= len(matches)):
            return R, t, theta
        
        matches = new_matches
        samples = random.sample(matches, n_samples)

    return R, t, theta
    
    
    

def calculate_coordinate(u, v, z):
        y = z * (v-CV)/F
        x = z * (u-CU)/F

        p3d_vector = np.array([x, y, z])
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