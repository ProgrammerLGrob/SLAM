#!/usr/bin/env python3
from tf2_ros import Buffer, TransformBroadcaster, TransformListener
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation
from numpy.typing import NDArray
from rclpy.time import Time
from scipy.spatial import KDTree

from math import pi
import random
from typing import List, Tuple

import copy

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from sensor_msgs_py import point_cloud2

import cv2
from cv_bridge import CvBridge

import numpy as np

from visual_odom.landmark import *
from visual_odom.visual_odom_map import *
from visual_odom.tf_methods import *
from visual_odom.constants import *
from visual_odom.ekf_robot import *
from visual_odom.robot import *


from nav_msgs.msg import Odometry

class VisualOdom(Node):
    def __init__(self):
        super().__init__('visual_odom')

        self.parameters = self.parameter_initialization()
        
        self.bridge = CvBridge()
        self.orb = cv2.ORB_create(
            nfeatures=2000,        # maximale Anzahl an zu detektierenden Keypoints
            scaleFactor=1.1,       # Skalierungsfaktor zwischen den Pyramidenlevels (feine Größenabstufung)
            nlevels=10,            # Anzahl der Bildpyramiden-Level (Skalenbereich)
            edgeThreshold=40,      # Mindestabstand eines Keypoints vom Bildrand
            patchSize=40,          # Größe des Bereichs zur Descriptor-Berechnung
            fastThreshold=5,      # Schwellwert für FAST-Feature-Erkennung (Empfindlichkeit)
            scoreType=cv2.ORB_HARRIS_SCORE  # Methode zur Bewertung der Keypoint-Qualität
        )
        #create BFMatcher object
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self.subscription_rgb = self.create_subscription(Image,'/serf01/nav_rgbd_1/rgb/image_raw', self.listener_rgb_callback, 10)
        self.subscription_depth = self.create_subscription(Image,'/serf01/nav_rgbd_1/depth/image_raw', self.listener_depth_callback, 10)

        self.publisher_keypoints_3d = self.create_publisher(PointCloud2, KEYPOINT_POINTCLOUD_FRAME_ID, 10)
        self.publisher_3d = self.create_publisher(PointCloud2, POINTCLOUD_FRAME_ID, 10)
        self.publisher_visual_odometry_msg = self.create_publisher(Odometry, self.parameters.topic_visual_odometry_msg, 10)


        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.frame_depth = None
        self.frame_depth_stamp = None
        self.frame_rgb = None
        #self.theta = -98.0 * pi / 180.0 
        self.theta = 0.0 
        self.pos_baselink = Coordinate(0.0, 0.0, 0.0) #Position in odom frame

        self.first_iteration = True

		# self.P = self.Q:
        sigma_x0   = 0.1   # 10cm Anfangsunsicherheit in x
        sigma_y0   = 0.1   # 10cm in y
        sigma_th0  = 0.05  # ~3° in theta

        self.covariance_P =  np.diag([sigma_x0**2, sigma_y0**2, sigma_th0**2])

        self.robots: List[VisualRobotSample] = []
        self.best_robot_idx = 0
        self.best_weight = 0.0

        self.get_logger().info("Visual Odometry Node gestartet und bereit für die Verarbeitung von RGB-D Daten.")

        

    def listener_rgb_callback(self,msg):
        valid_kp, valid_des, valid_kp_depth, rgb_stamp = self.img_to_kp_des_filtered(msg)

        if valid_kp is None or valid_des is None or valid_kp_depth is None:
            return
      
        if self.first_iteration:
            odom_to_base_footprint = calculate_tf(self.pos_baselink, self.theta, rgb_stamp,  VISUAL_ODOM_FRAME_ID, BASE_LINK_FRAME_ID)
            self.tf_broadcaster.sendTransform(odom_to_base_footprint)

            self.first_iteration = False
            for _ in range(self.parameters.n_robot_samples):
                
                self.robots.append(VisualRobotSample(self.pos_baselink, self.theta, self.covariance_P, self.bf, self.publisher_visual_odometry_msg, self.publisher_keypoints_3d,valid_kp, valid_des, valid_kp_depth, self.frame_rgb, self.parameters))
            return
        else:  
            self.robot_iteration(valid_kp, valid_des, valid_kp_depth, self.frame_rgb, self.frame_depth, rgb_stamp)  
          
            self.pos_baselink = self.best_robot.get_position()
            self.theta = self.best_robot.get_theta()
            self.covariance_P = self.best_robot.get_covariance_P()
          
            odom_to_base_footprint = calculate_tf(self.pos_baselink, self.theta, rgb_stamp, VISUAL_ODOM_FRAME_ID, BASE_LINK_FRAME_ID)
            self.tf_broadcaster.sendTransform(odom_to_base_footprint)


            self.best_robot.publish_yourself(rgb_stamp)           

            frame_rgb_drawn = cv2.drawKeypoints(self.frame_rgb, self.best_robot.get_drawn_keypoints(), None, color=(0,255,0), flags=0)
            cv2.imshow("second RGB Image", frame_rgb_drawn)
            cv2.waitKey(1)
                

    def listener_depth_callback(self,msg):
        self.frame_depth=self.bridge.imgmsg_to_cv2(msg,'passthrough')
        self.frame_depth_stamp = msg.header.stamp

    def _stamp_to_sec(self, stamp) -> float:
        return stamp.sec + stamp.nanosec * 1e-9

    def publish_pixels(self, frame_depth: NDArray, frame_rgb: NDArray, publisher, time: Time, frame_id: str):
        calculated_point_coordinates = []

        divisor = 15
        for pix_u in range(frame_depth.shape[1]//divisor):
            for pix_v in range(frame_depth.shape[0]//divisor):
                depth_value = frame_depth[pix_v*divisor, pix_u*divisor]
                if depth_value > self.parameters.min_depth and depth_value < self.parameters.max_depth:
                    pos = pixel_to_kinect(PixelCoordinate(pix_u*divisor, pix_v*divisor, depth_value))
                    b, g, r = frame_rgb[pix_v*divisor, pix_u*divisor]
                    rgb = (int(r) << 16) | (int(g) << 8) | int(b)

                    calculated_point_coordinates.append((pos.x, pos.y, pos.z, rgb))

        from std_msgs.msg import Header
        h = Header()
        h.stamp = time
        h.frame_id = frame_id

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
        ]

        msg = point_cloud2.create_cloud(
            header=h,
            fields=fields,
            points=calculated_point_coordinates
        )

        publisher.publish(msg)
        self.get_logger().info(f"PointCloud mit {len(calculated_point_coordinates)} Punkten gesendet!")

    def img_to_kp_des_filtered(self, msg) -> Tuple[List[cv2.KeyPoint], np.ndarray, np.ndarray, Time]:
        valid_kp = []
        valid_des = []
        valid_kp_depth = []

        self.frame_rgb=self.bridge.imgmsg_to_cv2(msg,'bgr8')
        kp = self.orb.detect(self.frame_rgb, None)


        rgb_stamp = msg.header.stamp
        if self.frame_depth is None or self.frame_depth_stamp is None:
            return None, None, None, None  

        
        if abs(self._stamp_to_sec(rgb_stamp) - self._stamp_to_sec(self.frame_depth_stamp)) > self.parameters.rgb_depth_sync_tolerance_sec:
            self.get_logger().debug("RGB/Depth nicht ausreichend synchron - Frame wird uebersprungen.")
            return None, None, None, None

        # Evaluation of the detected keypoints: Only keypoints with valid depth values are kept for further processing
        h, w = self.frame_depth.shape[:2]

        occupied_pixels = {} 
        all_kp = []
        valid_kp = []
        valid_kp_depth = []

        for p in kp:
            u, v = p.pt
            u = int(round(u))
            v = int(round(v))

            # Bildgrenzen prüfen
            if not (0 <= u < w and 0 <= v < h):
                continue

            depth_value = self.frame_depth[v, u]

            # Tiefenwert prüfen
            if self.parameters.min_depth < depth_value < self.parameters.max_depth:
                all_kp.append(p)
                
                is_too_close = False
                
                # Super-schneller Lookup im Dictionary
                for neighbor_u in range(u - self.parameters.pixel_tolerance, u + self.parameters.pixel_tolerance + 1):
                    for neighbor_v in range(v - self.parameters.pixel_tolerance, v + self.parameters.pixel_tolerance + 1):
                        if (neighbor_u, neighbor_v) in occupied_pixels:
                            is_too_close = True
                            break
                    if is_too_close:
                        break
                
                # Wenn kein anderer Punkt im Umkreis ist -> Hinzufügen
                if not is_too_close:
                    valid_kp.append(p)
                    valid_kp_depth.append(depth_value)
                    
                    # Position im Dictionary als "belegt" markieren
                    occupied_pixels[(u, v)] = True

        # ORB Deskriptoren nur für die validen Keypoints berechnen
        valid_kp, valid_des = self.orb.compute(self.frame_rgb, valid_kp)

        return valid_kp, valid_des, valid_kp_depth, rgb_stamp
 
    def robot_iteration(self, valid_kp: List[cv2.KeyPoint], valid_des: np.ndarray, valid_kp_depth: np.ndarray, frame_rgb: NDArray, frame_depth: NDArray, rgb_stamp: Time):
        weights = []
        for robot in self.robots:
            robot.robot_iteration(valid_kp, valid_des, valid_kp_depth, frame_rgb, frame_depth, rgb_stamp)
            weights.append(robot.get_weight())

        sum = np.sum(weights)
        
        normalized_weights = weights/sum
    
        self.best_robot_idx = np.argmax(normalized_weights)
        self.best_weight = normalized_weights[self.best_robot_idx]
        self.best_robot = self.robots[self.best_robot_idx]


    def parameter_initialization(self) -> Parameters:
        # Load parameters from config file with explicit type casting
        parameters = Parameters(
            min_depth = int(self.declare_parameter('min_depth', MIN_DEPTH).value),
            max_depth = int(self.declare_parameter('max_depth', MAX_DEPTH).value),
            ransac_evaluation_tolerance = float(self.declare_parameter('ransac.evaluation_tolerance', RANSAC_EVALUATION_TOLERANCE).value),
            ransac_iteration = int(self.declare_parameter('ransac.iterations', RANSAC_ITERATION).value),
            ransac_sample_size = int(self.declare_parameter('ransac.sample_size', RANSAC_SAMPLE_SIZE).value),
            rgb_depth_sync_tolerance_sec = float(self.declare_parameter('rgb_depth_sync_tolerance_sec', RGB_DEPTH_SYNC_TOLERANCE_SEC).value),
            pixel_tolerance = int(self.declare_parameter('pixel_tolerance', PIXEL_TOLERANCE).value),
            matches_for_new_landmarks = int(self.declare_parameter('matches_for_new_landmarks', MATCHES_FOR_NEW_LANDMARKS).value),
            min_matches_for_ransac = int(self.declare_parameter('ransac.min_matches_for_ransac', MIN_MATCHES_FOR_RANSAC).value),
            max_rotation_angle_deg = float(self.declare_parameter('ransac.max_rotation_angle_deg', MAX_ROTATION_ANGLE_DEG).value),
            max_landmark_age = float(self.declare_parameter('max_landmark_age', MAX_LANDMARK_AGE).value),
            ransac_min_inlier_ratio = float(self.declare_parameter('ransac.min_inlier_ratio', RANSAC_MIN_INLIER_RATIO).value),
            n_robot_samples = int(self.declare_parameter('n_robot_samples', N_ROBOT_SAMPLES).value),
            topic_visual_odometry_msg = self.declare_parameter('topics.visual_odometry_msg', VISUAL_ODOM_MSG_TOPIC).value
        )
        return parameters

def calculate_tf(Position: Coordinate, theta: float, timestamp,  parent_frame_id: str, child_frame_id: str) -> Tuple[TransformStamped, NDArray]:

    t = TransformStamped()
    t.header.stamp = timestamp
    t.header.frame_id = parent_frame_id
    t.child_frame_id = child_frame_id

    t.transform.translation.x = Position.x
    t.transform.translation.y = Position.y
    t.transform.translation.z = Position.z

    euler = Rotation.from_euler('z', float(theta))
    quat = euler.as_quat(canonical=True)
    # Cast quaternion components to Python floats
    t.transform.rotation.x = quat[0]
    t.transform.rotation.y = quat[1]
    t.transform.rotation.z = quat[2]
    t.transform.rotation.w = quat[3]

    return t  


def main():
    rclpy.init()
    node=VisualOdom()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
