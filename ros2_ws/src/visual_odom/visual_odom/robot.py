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

import rclpy
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


from nav_msgs.msg import Odometry

class VisualRobotSample():
    def __init__(self, pos: Coordinate, theta: float, covariance_P: NDArray, matcher: cv2.BFMatcher, odom_publisher:Publisher, map_publisher:Publisher, valid_kp: List[cv2.KeyPoint], valid_des: np.ndarray, valid_kp_depth: np.ndarray, frame_rgb: NDArray, parameters: Parameters, visual_odom_map: VisualOdomMap = VisualOdomMap()):
        self.theta = theta 
        self.pos_baselink = pos

        #create BFMatcher object
        self.bf = matcher

        self.first_iteration = True
        self.visual_odom_map = visual_odom_map
        self.visual_odom_map.add_landmarks_from_kps(covariance_P, valid_kp, valid_des, frame_rgb, valid_kp_depth, self.theta, self.pos_baselink)
    
        self.odometry_msg_publisher = odom_publisher
        self.map_publisher = map_publisher

        self.covariance_P = covariance_P
        self.noise_Q = np.eye(3) * 1e-6
        self.extended_kalman_filter = ExtendedKalmanFilterRobot(State(self.pos_baselink.x, self.pos_baselink.y, self.theta), self.covariance_P, self.noise_Q)
        self.parameters = parameters
        self.weight = 0.0

    def publish_yourself(self, rgb_stamp):
        self.publish_odometry_msg(self.odometry_msg_publisher, self.pos_baselink, self.theta, self.covariance_P, rgb_stamp, VISUAL_ODOM_FRAME_ID, BASE_LINK_FRAME_ID)
        self.publish_pointcloud_map(self.map_publisher, rgb_stamp)
        

    def robot_iteration(self, valid_kp: List[cv2.KeyPoint], valid_des: np.ndarray, valid_kp_depth: np.ndarray, frame_rgb, depth_frame, rgb_stamp):
        self.frame_rgb = frame_rgb
        self.frame_depth = depth_frame
        c = Coordinate()

        if self.first_iteration and len(self.visual_odom_map) < 50:
            self.first_iteration = False
            self.visual_odom_map.add_landmarks_from_kps(self.covariance_P, valid_kp, valid_des, self.frame_rgb, valid_kp_depth, self.theta, self.pos_baselink)
            return
        else:            
            pos_camera = kinect_depth_to_odom(Coordinate(0.0, 0.0, 0.0), self.theta,self.pos_baselink)

            self.visible_landmarks = self.visual_odom_map.get_visible_landmarks(pos_camera, self.theta)

            # Match descriptors.
            matches = self.bf.match(self.visible_landmarks.get_descriptors(), valid_des)

            if len(matches) < self.parameters.min_matches_for_ransac:
                self.visual_odom_map.add_landmarks_from_kps(self.covariance_P, valid_kp, valid_des, self.frame_rgb, valid_kp_depth, self.theta, self.pos_baselink)
                return
            
            ransac_result = self.ransac(self.parameters.ransac_evaluation_tolerance, self.parameters.ransac_iteration, self.parameters.ransac_sample_size, matches, self.visible_landmarks.get_odom_coordinates(), valid_kp, valid_kp_depth)
            ransac_delta_p = ransac_result[0]
            ransac_delta_theta = ransac_result[1]
            ransac_landmark_indices = ransac_result[2]
            self.ransac_draw_keypoints = ransac_result[3]
            ransac_kp_indices = ransac_result[4]
            ransac_not_matched_kp_indices = ransac_result[5]
            ransac_inlier_count = ransac_result[6]

            if float(ransac_inlier_count)/len(matches) < self.parameters.ransac_min_inlier_ratio:  # Weniger als 10% Inlier nach RANSAC
                self.visual_odom_map.add_landmarks_from_kps(self.covariance_P, valid_kp, valid_des, self.frame_rgb, valid_kp_depth, self.theta, self.pos_baselink)
                return

            z_dict = {}
            kp_pos = []
            for i in ransac_kp_indices:
                u, v = valid_kp[i].pt
                u = int(round(u))
                v = int(round(v))

                depth_value = self.frame_depth[v, u]

                kp_pos.append(PixelCoordinate(u, v, depth_value))

            kalman_iteration_result = self.extended_kalman_filter.kalman_iteration(ransac_delta_p, ransac_delta_theta, z_dict, self.visible_landmarks)
            self.pos_baselink = Coordinate(kalman_iteration_result[0].x, kalman_iteration_result[0].y, 0.0)
            self.theta = kalman_iteration_result[0].theta
            self.covariance_P = kalman_iteration_result[1]
            
            self.visible_landmarks.landmark_kalman_iteration(self.pos_baselink, self.theta, ransac_landmark_indices, kp_pos)
            self.weight = self.visible_landmarks.calculate_weight(self.pos_baselink, self.theta, ransac_landmark_indices, kp_pos)

            #self.visual_odom_map.age_and_cleanup_old_landmarks(self.visible_landmarks,ransac_landmark_indices,self.parameters.max_landmark_age,)

            if len(matches) < self.parameters.matches_for_new_landmarks:
                not_matched_kp = []
                not_matched_des = []
                not_matched_kp_depth = []
                for idx in ransac_not_matched_kp_indices:
                    not_matched_kp.append(valid_kp[idx])
                    not_matched_des.append(valid_des[idx])
                    not_matched_kp_depth.append(valid_kp_depth[idx])

                
                #self.visual_odom_map.add_landmarks_from_kps(self.covariance_P, valid_kp, valid_des, self.frame_rgb, valid_kp_depth, self.theta, self.pos_baselink)
                self.visual_odom_map.add_landmarks_from_kps(self.covariance_P, not_matched_kp, not_matched_des, self.frame_rgb, not_matched_kp_depth, self.theta, self.pos_baselink)


    def ransac(self, tolerance: float, iteration: int, n_samples: int, matches: List[cv2.DMatch], landmarks_odom_pos: List[Coordinate], valid_kp: List[cv2.KeyPoint], valid_kp_depth: np.ndarray) -> Tuple[Coordinate, float, List[int], List[int], List[int], List[int], int]:
        #Abbruchbedingung
        
        P = []
        Q = []

        best_inlier_count = 0
        inlier_count = 0
        best_t = Coordinate(0.0, 0.0, 0.0)
        best_theta = 0.0

        landmark_index = [] 

        for m in range(len(matches)):            
            #Matrix mit Koordinaten im kinect frame
            keypoint = valid_kp[matches[m].trainIdx].pt
            
            u, v = keypoint
            u = round(u)
            v = round(v)
            z = valid_kp_depth[matches[m].trainIdx]

            coor_landmark = landmarks_odom_pos[matches[m].queryIdx]
            coor_base_link = odom_to_baselink(coor_landmark, self.theta, self.pos_baselink)
            P.append([coor_base_link.x, coor_base_link.y])

            coor = pixel_to_kinect(PixelCoordinate(u, v, z))
            coor_base_link = kinect_depth_to_baselink(coor)
            Q.append([coor_base_link.x, coor_base_link.y])

            landmark_index.append(matches[m].queryIdx)


        if len(P) < n_samples:
            return Coordinate(0.0, 0.0, 0.0), 0.0, [], [], [], [], 0

        P_array = np.array(P)
        Q_array = np.array(Q)


        best_P_inlier = []
        best_Q_inlier = []
        best_valid_kp_index = []
        best_draw_Q_inlier = []
        best_landmark_index = []
        mean_last_e = 10000.0

        iteration = int(self.parameters.ransac_iteration)

        for iter in range(iteration):
            P_samples = []
            Q_samples = []

            samples = random.sample(range(len(P)), n_samples)
            P_samples = [P[i] for i in samples]
            Q_samples = [Q[i] for i in samples]

            R, t, theta  = kabsch(np.array(P_samples), np.array(Q_samples), self.parameters.max_rotation_angle_deg)
            if R is None:
                continue
   
            e = np.linalg.norm(P_array - ((R @ Q_array.T).T + t.T), axis=1)
            mean_e = np.mean(e)

            inlier_count = np.sum(e < tolerance)

            if inlier_count > best_inlier_count or (inlier_count == best_inlier_count and mean_e < mean_last_e):
                best_inlier_count = inlier_count
                mean_last_e = mean_e

                best_P_inlier = []
                best_Q_inlier = []
                best_valid_kp_index = []
                best_draw_Q_inlier = []
                best_landmark_index = []
                
                for i in range(len(e)):
                    if e[i] < tolerance:
                        best_P_inlier.append(P_array[i])
                        best_Q_inlier.append(Q_array[i])
                        best_valid_kp_index.append(matches[i].trainIdx)
                        best_draw_Q_inlier.append(valid_kp[matches[i].trainIdx])
                        best_landmark_index.append(matches[i].queryIdx) 

        # Kabsch noch einmal mit den endgültigen besten Inliern berechnen
        R, t, theta  = kabsch(np.array(best_P_inlier), np.array(best_Q_inlier), self.parameters.max_rotation_angle_deg)
        if R is None:
            return Coordinate(0.0, 0.0, 0.0), 0.0, [], [], [], [], 0
            
        best_t = Coordinate(float(t[0]), float(t[1]), 0.0)
        best_theta = theta

        valid_set = set(best_valid_kp_index)
        not_matched_kp = [i for i in range(len(valid_kp)) if i not in valid_set]
        
        # Gebe nun exakt die synchronisierten "best_"-Listen zurück
        return best_t, best_theta, best_landmark_index, best_draw_Q_inlier, best_valid_kp_index, not_matched_kp, best_inlier_count

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
    
    def get_weight(self) -> float:
        return self.weight