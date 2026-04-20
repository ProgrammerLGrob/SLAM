#!/usr/bin/env python3

KINECT_FRAME_ID = "kinect_depth"

MIN_DEPTH = 400
MAX_DEPTH = 5000

DESCRIPTOR_TOLERANCE = 50
RANSAC_EVALUATION_TOLERANCE = 50 #in mm
RANSAC_ITERATION = 1000
RANSAC_SAMPLE_SIZE = 3
RGB_DEPTH_SYNC_TOLERANCE_SEC = 0.05

F = 526.61
CU = 318.525
CV = 241.181


from math import pi
import random

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
        self.frame_depth = None
        self.frame_depth_stamp = None
        self.frame_rgb = None
        self.counter = 0

    def listener_rgb_callback(self,msg):
        self.frame_rgb=self.bridge.imgmsg_to_cv2(msg,'bgr8')
        self.kp = self.orb.detect(self.frame_rgb, None)

        if self.frame_depth is None or self.frame_depth_stamp is None:
            return  

        rgb_stamp = msg.header.stamp
        if abs(self._stamp_to_sec(rgb_stamp) - self._stamp_to_sec(self.frame_depth_stamp)) > RGB_DEPTH_SYNC_TOLERANCE_SEC:
            self.get_logger().debug("RGB/Depth nicht ausreichend synchron - Frame wird uebersprungen.")
            return


        valid_kp = []
        h, w = self.frame_depth.shape[:2]
        for p in self.kp:
            u, v = p.pt
            u = int(round(u))
            v = int(round(v))

            if not (0 <= u < w and 0 <= v < h):
                continue

            depth_value = self.frame_depth[v, u]

            if depth_value > MIN_DEPTH and depth_value < MAX_DEPTH:
                valid_kp.append(p) 

        if len(valid_kp) < RANSAC_SAMPLE_SIZE:
            self.get_logger().warn("Zu wenige valide Keypoints fuer robuste Schaetzung.")
            return

        if self.frame_rgb_first is None:
            self.frame_rgb_first = self.frame_rgb
            self.frame_depth_first = self.frame_depth
            self.kp_first,self.des_first = self.orb.compute(self.frame_rgb, valid_kp)
            return

        self.counter += 1

        if self.frame_rgb_end is None and self.counter > 230:
            self.frame_rgb_end = self.frame_rgb
            self.frame_depth_end = self.frame_depth
            self.kp_end,self.des_end = self.orb.compute(self.frame_rgb, valid_kp)
        else:
            return

        #create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # Match descriptors.
        matches = bf.match(self.des_first,self.des_end)

        if len(matches) < RANSAC_SAMPLE_SIZE:
            self.get_logger().warn(f"Zu wenige Matches fuer RANSAC: {len(matches)}")
            return

        ransac_result = self.ransac(RANSAC_EVALUATION_TOLERANCE, RANSAC_ITERATION, RANSAC_SAMPLE_SIZE, matches, self.kp_first, self.kp_end, self.frame_depth_first, self.frame_depth_end)

        self.get_logger().info(f"RANSAC Result: Rotationsmatrix {ransac_result[0]}, Translationsvektor {ransac_result[1]}, Theta in ° {ransac_result[2]*180/pi}")
        
        cv2.imshow("End RGB Image", self.frame_rgb_end)
        cv2.waitKey(0)
            

    def listener_depth_callback(self,msg):
        self.frame_depth=self.bridge.imgmsg_to_cv2(msg,'passthrough')
        self.frame_depth_stamp = msg.header.stamp


    def _stamp_to_sec(self, stamp):
        return stamp.sec + stamp.nanosec * 1e-9



    def ransac(self,tolerance, iteration, n_samples, matches, kp_first, kp_end, frame_depth_first, frame_depth_end):
        P = []
        Q = []

        best_inlier_count = 0
        inlier_count = 0
        best_R = np.eye(2)
        best_t = np.zeros((2,1))
        best_theta = 0

        h_first, w_first = frame_depth_first.shape[:2]
        h_end, w_end = frame_depth_end.shape[:2]

        for m in range(len(matches)):            
            #Matrix mit Koordinaten im kinect frame
            keypoint_first = kp_first[matches[m].queryIdx].pt
            keypoint_end = kp_end[matches[m].trainIdx].pt
            
            u_first, v_first = keypoint_first
            u_first = round(u_first)
            v_first = round(v_first)

            if not (0 <= u_first < w_first and 0 <= v_first < h_first):
                continue

            z_first = frame_depth_first[v_first, u_first]

            u_end, v_end = keypoint_end
            u_end = round(u_end)
            v_end = round(v_end)

            if not (0 <= u_end < w_end and 0 <= v_end < h_end):
                continue

            z_end = frame_depth_end[v_end, u_end]

            if z_first <= MIN_DEPTH or z_first >= MAX_DEPTH or z_end <= MIN_DEPTH or z_end >= MAX_DEPTH:
                continue

            coor = calculate_coordinate(u_first, v_first, z_first)
            P.append([coor[0], coor[2]])
            coor = calculate_coordinate(u_end, v_end, z_end)
            Q.append([coor[0], coor[2]])


        if len(P) < n_samples:
            self.get_logger().warn(f"Nicht genug Punkte für RANSAC: {len(P)} < {n_samples}")
            return best_R, best_t, best_theta

        P_array = np.array(P)
        Q_array = np.array(Q)

        for iter in range(iteration):
            P_samples = []
            Q_samples = []

            inlier_count = 0 
            samples = random.sample(range(len(P)), n_samples) #Take new random samples for next iteration
            P_samples = [P[i] for i in samples]
            Q_samples = [Q[i] for i in samples]


            R, t, theta  = kabash(np.array(P_samples), np.array(Q_samples))
   
            e = np.linalg.norm(P_array - ((R @ Q_array.T).T + t.T), axis=1) #calculate error for all points
            inlier_count = np.sum(e < tolerance) #count inliers for this iteration

            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_R = R
                best_t = t
                best_theta = theta
                
        self.get_logger().info(f"RANSAC abgeschlossen. Beste Lösung hatte {best_inlier_count} Inlier von {len(matches)} Punkten.")
        return best_R, best_t, best_theta
    
    
    
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
        
        
        

def calculate_coordinate(u, v, z):
        y = z * (v-CV)/F
        x = z * (u-CU)/F

        p3d_vector = np.array([x, y, z])
        return p3d_vector

    
def main():
    rclpy.init()
    node=VisualOdom()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()