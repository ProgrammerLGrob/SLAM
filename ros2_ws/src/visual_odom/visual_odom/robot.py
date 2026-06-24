#!/usr/bin/env python3
import math

from tf2_ros import Buffer, TransformBroadcaster, TransformListener
from geometry_msgs.msg import Pose2D, TransformStamped, Point
from scipy.spatial.transform import Rotation
from visualization_msgs.msg import Marker

from numpy.typing import NDArray
from rclpy.time import Time
from scipy.spatial import KDTree

from math import pi, sin, cos, tan
import random
from typing import List, Tuple

import time

import rclpy
from rclpy.logging import get_logger
from rclpy.node import Node, Publisher
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
import visual_odom.constants as constants


from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker

class VisualRobotSample():
    def __init__(self, pos: Coordinate, theta: float, covariance_P: NDArray, matcher: cv2.BFMatcher, odom_publisher:Publisher, map_publisher:Publisher, cone_publisher:Publisher, valid_kp: List[cv2.KeyPoint], valid_des: np.ndarray, valid_kp_depth: np.ndarray, frame_rgb: NDArray, index:int, map = None):        
        self.theta = theta 
        self.pos_baselink = pos
        self.index = index
        if map is None:
            map = VisualOdomMap()
        self.visual_odom_map = map

        #create BFMatcher object
        self.bf = matcher

        self.first_iteration = True
        self.ransac_delta_p = None
        self.ransac_inlier_ratio = 0.0
        self.visual_odom_map.add_landmarks_from_kps(covariance_P, valid_kp, valid_des, frame_rgb, valid_kp_depth, self.theta, self.pos_baselink)
        self.keyframe_map = copy.deepcopy(self.visual_odom_map)

        self.odometry_msg_publisher = odom_publisher
        self.map_publisher = map_publisher
        self.cone_publisher = cone_publisher
        self.covariance_P = covariance_P
        self.noise_Q = np.eye(3) * 1e-6
        self.ekf = ExtendedKalmanFilterRobot(State(self.pos_baselink.x, self.pos_baselink.y, self.theta), self.covariance_P, self.noise_Q)
        self.log_weight = 0.0
        self.ransac_draw_keypoints = []
        self.last_wheel_odom = None
        self.delta_wheel_odom_ransac = State(0.0, 0.0, 0.0)
        self.acc_ransac_noise_delta = State(0.0, 0.0, 0.0)
        self.pos_visual_odom = State(self.pos_baselink.x, self.pos_baselink.y, self.theta)

        self.path = []  
        
        marker = Marker()
        # Zufällige Farbe, aber nicht rot
        marker.color.r = random.uniform(0.0, 0.8)
        marker.color.g = random.uniform(0.0, 1.0)
        marker.color.b = random.uniform(0.0, 1.0)

        # Falls die Farbe zu rot-lastig wird, neu würfeln
        while (
            marker.color.r > 0.8 and
            marker.color.g < 0.3 and
            marker.color.b < 0.3
        ):
            marker.color.r = random.uniform(0.0, 0.8)
            marker.color.g = random.uniform(0.0, 1.0)
            marker.color.b = random.uniform(0.0, 1.0)

        marker.color.a = 1.0
        self.path_color = marker.color


    def publish_yourself(self, rgb_stamp):
        self.publish_odometry_msg(self.odometry_msg_publisher, self.pos_baselink, self.theta, self.covariance_P, rgb_stamp, VISUAL_ODOM_FRAME_ID, BASE_LINK_FRAME_ID)
        self.publish_pointcloud_map(self.map_publisher, rgb_stamp)
        self.publish_vision_cone(self.cone_publisher, rgb_stamp)


    def predict_with_wheel_odom(self, state_wheel_odom: State):
        if self.last_wheel_odom is None:
            self.last_wheel_odom = state_wheel_odom  # ersten Wert nur speichern, kein Delta
            return
        delta_wheel_odom = state_wheel_odom - self.last_wheel_odom
        self.ekf.prediction(delta_wheel_odom)
        self.last_wheel_odom = state_wheel_odom

        self.delta_wheel_odom_ransac += delta_wheel_odom



    def robot_iteration(self, valid_kp: List[cv2.KeyPoint], valid_des: np.ndarray, valid_kp_depth: np.ndarray, frame_rgb, depth_frame, rgb_stamp, ransac_result):
        self.frame_rgb = frame_rgb
        self.frame_depth = depth_frame

        if self.first_iteration and len(self.visual_odom_map) < INITIAL_LANDMARK_COUNT:
            self.visual_odom_map.add_landmarks_from_kps(self.covariance_P, valid_kp, valid_des, self.frame_rgb, valid_kp_depth, self.theta, self.pos_baselink)
            return
        else:       
            #if math.sqrt(self.delta_wheel_odom_ransac.x**2 + self.delta_wheel_odom_ransac.y**2) < 0.05 and abs(self.delta_wheel_odom_ransac.theta) < 0.06:
                #rclpy.logging.get_logger("VisualRobotSample").info(f"Skipping visual odometry update due to low wheel odometry movement: {self.delta_wheel_odom_ransac}")
               # return  
            
            self.first_iteration = False   
            pos_camera = kinect_depth_to_odom(Coordinate(0.0, 0.0, 0.0), self.theta,self.pos_baselink)

            self.visible_landmarks = self.visual_odom_map.get_visible_landmarks(pos_camera, self.theta)
            #self.visible_landmarks = self.visual_odom_map

            # Match descriptors.
            matches = self.bf.match(self.visible_landmarks.get_descriptors(), valid_des)

            if len(matches) < constants.parameters.min_matches_for_ransac:
                self.visual_odom_map.add_landmarks_from_kps(self.covariance_P, valid_kp, valid_des, self.frame_rgb, valid_kp_depth, self.theta, self.pos_baselink)
                return
            landmark_indices = []
            matched_kp_index = []

            for m in range(len(matches)): 
                landmark_indices.append(matches[m].queryIdx)
                matched_kp_index.append(matches[m].trainIdx)

            valid_set = set(matched_kp_index)
            not_matched_kp_indices = [i for i in range(len(valid_kp)) if i not in valid_set]

            
            
            self.ransac_delta_p = ransac_result[0]

            self.ransac_delta_theta = ransac_result[1]
            ransac_landmark_indices = landmark_indices
            self.ransac_draw_keypoints = ransac_result[2]
            ransac_inlier_count = ransac_result[3]
            self.ransac_inlier_ratio = float(ransac_inlier_count)/len(matches) if len(matches) > 0 else 0.0

            if self.ransac_inlier_ratio < constants.parameters.ransac_min_inlier_ratio:  # Weniger als x% Inlier nach RANSAC
                self.visual_odom_map.add_landmarks_from_kps(self.covariance_P, valid_kp, valid_des, self.frame_rgb, valid_kp_depth, self.theta, self.pos_baselink)
                #rclpy.logging.get_logger("RANSAC").info(f"RANSAC result rejected due to low inlier ratio: {float(ransac_inlier_count)/len(matches):.2f} with {ransac_inlier_count} inliers out of {len(matches)} matches.")
                self.ransac_failed()
                return
            
            c = cos(self.theta + self.ransac_delta_theta)
            s = sin(self.theta + self.ransac_delta_theta)

        R = np.array([[c, -s],
                        [s,  c]])
        delta = R @ np.array([self.ransac_delta_p.x, self.ransac_delta_p.y])
        self.ransac_delta_p = Coordinate(delta[0], delta[1], 0.0)

            self.delta_ransac_state = State(self.ransac_delta_p.x, self.ransac_delta_p.y, self.ransac_delta_theta)

            self.acc_ransac_noise_delta =  self.acc_ransac_noise_delta + self.delta_ransac_state  
            
            if math.sqrt(self.acc_ransac_noise_delta.x**2 + self.acc_ransac_noise_delta.y**2) > 0.10:

                noise_x = np.random.normal(0.0, GAUSS_NOISE_X_SIGMA)
                noise_y = np.random.normal(0.0, GAUSS_NOISE_Y_SIGMA)
                self.delta_ransac_state.x += noise_x
                self.delta_ransac_state.y += noise_y

                self.acc_ransac_noise_delta.x = 0.0
                self.acc_ransac_noise_delta.y = 0.0

            if abs(self.acc_ransac_noise_delta.theta) > 0.06:
                noise_theta = np.random.normal(0.0, GAUSS_NOISE_THETA_SIGMA)
                self.delta_ransac_state.theta += noise_theta
                self.acc_ransac_noise_delta.theta = 0.0

            new_state = self.pos_visual_odom + self.delta_ransac_state
            new_state.theta = normalize_angle(new_state.theta)
            self.pos_visual_odom = new_state

            self.ekf.update(self.pos_visual_odom, self.ransac_inlier_ratio)
            self.delta_wheel_odom_ransac = State(0.0, 0.0, 0.0)

            state = self.ekf.get_state()
            self.pos_baselink = Coordinate(state.x, state.y, 0.0)
            self.theta = state.theta
            self.covariance_P = self.ekf.get_covariance_p()

            #state = self.ekf.get_state()
            #self.pos_baselink = Coordinate(self.pos_visual_odom.x, self.pos_visual_odom.y, 0.0)
            #self.theta = self.pos_visual_odom.theta
            #self.covariance_P = self.ekf.get_covariance_p()



            kp_pos = []
            for i in matched_kp_index:
                u, v = valid_kp[i].pt
                u = int(round(u))
                v = int(round(v))

            depth_value = self.frame_depth[v, u]

            kp_pos.append(PixelCoordinate(u, v, depth_value))
        
        self.log_weight = self.visible_landmarks.calculate_log_weight(self.pos_baselink, self.theta, ransac_landmark_indices, kp_pos)
        self.visible_landmarks.landmark_kalman_iteration(self.pos_baselink, self.theta, ransac_landmark_indices, kp_pos)

        self.visual_odom_map.cleanup_old_landmarks(self.visible_landmarks,ransac_landmark_indices)

            if len(matches) < constants.parameters.matches_for_new_landmarks:
                not_matched_kp = []
                not_matched_des = []
                not_matched_kp_depth = []
                for idx in not_matched_kp_indices:
                    not_matched_kp.append(valid_kp[idx])
                    not_matched_des.append(valid_des[idx])
                    not_matched_kp_depth.append(valid_kp_depth[idx])

            
            #self.visual_odom_map.add_landmarks_from_kps(self.covariance_P, valid_kp, valid_des, self.frame_rgb, valid_kp_depth, self.theta, self.pos_baselink)
            self.visual_odom_map.add_landmarks_from_kps(self.covariance_P, not_matched_kp, not_matched_des, self.frame_rgb, not_matched_kp_depth, self.theta, self.pos_baselink)

            self.path.append(Pose2D(x=self.pos_baselink.x, y=self.pos_baselink.y))





    def ransac_failed(self):
        new_state = self.pos_visual_odom + self.delta_wheel_odom_ransac
        new_state.theta = normalize_angle(new_state.theta)
        self.pos_visual_odom = new_state

        self.ekf.add_noise_to_R()
        self.ekf.update(self.pos_visual_odom, self.ransac_inlier_ratio)
        state = self.ekf.get_state()
        self.covariance_P = self.ekf.get_covariance_p()
        self.pos_baselink = Coordinate(state.x, state.y, 0.0)
        self.theta = state.theta
        
        self.delta_wheel_odom_ransac = State(0.0, 0.0, 0.0)

        #state = self.ekf.get_state()
        #self.pos_baselink = Coordinate(self.pos_visual_odom.x, self.pos_visual_odom.y, 0.0)
        #self.theta = self.pos_visual_odom.theta
        #self.delta_wheel_odom_ransac = State(0.0, 0.0, 0.0)


   

    def publish_odometry_msg(self, publisher, pos: Coordinate, theta: float, covariance: NDArray, timestamp, parent_frame_id: str, child_frame_id: str):
        msg = Odometry()

        # Header
        msg.header.stamp = timestamp
        msg.header.frame_id = parent_frame_id
        msg.child_frame_id = child_frame_id

        # Pose
        msg.pose.pose.position.x = pos.x
        msg.pose.pose.position.y = pos.y
        msg.pose.pose.position.z = pos.z

        euler = Rotation.from_euler('z', float(theta))
        quat = euler.as_quat(canonical=True)
        msg.pose.pose.orientation.x = quat[0]
        msg.pose.pose.orientation.y = quat[1]
        msg.pose.pose.orientation.z = quat[2]
        msg.pose.pose.orientation.w = quat[3]

        # Covariance
        msg.pose.covariance = [
           covariance[0, 0], covariance[0, 1],          0.0,    0.0,    0.0,   covariance[0, 2],
            covariance[1, 0], covariance[1, 1],         0.0,    0.0,    0.0,    covariance[1, 2],
            0.0,                0.0,                    0.0,    0.0,    0.0,    0.0,
            0.0,                0.0,                    0.0,    0.0,    0.0,    0.0,
            0.0,                0.0,                    0.0,    0.0,    0.0,    0.0,
            covariance[2, 0],    covariance[2, 1],      0.0,    0.0,    0.0,    covariance[2, 2]
        ]
        #msg.pose.covariance *= 10.0  # Skalierung der Kovarianz für bessere Visualisierung in RViz
        publisher.publish(msg)

    def publish_vision_cone(self, cone_publisher:Publisher, rgb_stamp:Time):
        cone_msg = self.init_camera_cone(rgb_stamp)
        cone_publisher.publish(cone_msg)

    def init_camera_cone(self, rgb_stamp):
        cone_msg = Marker()
        cone_msg.header.frame_id = "kinect_depth"
        cone_msg.header.stamp = rgb_stamp
        cone_msg.ns = "vision_cone"
        cone_msg.id = 0

        cone_msg.type = Marker.TRIANGLE_LIST    #cone-mode
        cone_msg.action = Marker.ADD
        cone_msg.scale.x = 1.0
        cone_msg.scale.y = 1.0
        cone_msg.scale.z = 1.0

        cone_msg.color.r = 0.0
        cone_msg.color.g = 1.0
        cone_msg.color.b = 0.0
        cone_msg.color.a = 0.3      #transparency
              
        h = tan(CAMERA_ANGLE_VER_RAD / 2)               #half for angle calcualtion
        w = tan(CAMERA_ANGLE_HOR_RAD / 2)               #half for angle calculation
        d = constants.parameters.max_depth/1000          #depth in meters for ros
        
        #center point of the cone
        p0 = Point()
        p0.x = 0.0 
        p0.y = 0.0
        p0.z = 0.0

        #lower right
        p1 = Point()
        p1.z = d 
        p1.x = d * w
        p1.y = d * h

        #lower left
        p2 = Point()
        p2.z = d 
        p2.x = -d * w 
        p2.y = d * h

        #upper right
        p3 = Point()
        p3.z = d
        p3.x = d * w
        p3.y = -d * h

        #upper left
        p4 = Point()
        p4.z = d
        p4.x = -d * w
        p4.y = -d * h


        cone_msg.points = []
        cone_msg.points.extend([p0, p1, p3, 
                                p0, p4, p2,
                                p0, p2, p1,
                                p0, p3, p4,
                                p1, p2, p4,
                                p4, p3, p1])        
    
        return cone_msg
    
    def publish_pointcloud_map(self, publisher: Publisher, time: Time):
        self.visual_odom_map.publish_pointcloud_map(publisher, time)


    def get_drawn_keypoints(self):
        return self.ransac_draw_keypoints
        
    def get_position(self) -> Coordinate:
        return self.pos_baselink
    
    def get_theta(self) -> float:
        return self.theta
    
    def get_covariance_P(self) -> NDArray:
        return self.covariance_P
    
    def get_log_weight(self) -> float:
        return self.log_weight
    
    def set_log_weight(self, log_weight: float) -> None:
        self.log_weight = log_weight

    def get_weight(self) -> float:
        try:
            return math.exp(self.log_weight)
        except OverflowError:
            return None
    
    def set_weight(self, weight: float) -> None:
        self.log_weight = math.log(weight)
