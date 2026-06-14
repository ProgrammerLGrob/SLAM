#!/usr/bin/env python3
import math

from tf2_ros import Buffer, TransformBroadcaster, TransformListener
from geometry_msgs.msg import TransformStamped, Point
from scipy.spatial.transform import Rotation
from visualization_msgs.msg import Marker

from numpy.typing import NDArray
from rclpy.time import Time
from scipy.spatial import KDTree

from math import pi, sin, cos, tan
import random
from typing import List, Tuple

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
    def __init__(self, pos: Coordinate, theta: float, covariance_P: NDArray, matcher: cv2.BFMatcher, odom_publisher:Publisher, map_publisher:Publisher, cone_publisher:Publisher, valid_kp: List[cv2.KeyPoint], valid_des: np.ndarray, valid_kp_depth: np.ndarray, frame_rgb: NDArray, visual_odom_map: VisualOdomMap = VisualOdomMap()):        
        self.theta = theta 
        self.pos_baselink = pos

        #create BFMatcher object
        self.bf = matcher

        self.first_iteration = True
        self.ransac_delta_p = None
        self.ransac_inlier_ratio = 0.0
        self.visual_odom_map = visual_odom_map
        self.visual_odom_map.add_landmarks_from_kps(covariance_P, valid_kp, valid_des, frame_rgb, valid_kp_depth, self.theta, self.pos_baselink)
        self.odometry_msg_publisher = odom_publisher
        self.map_publisher = map_publisher
        self.cone_publisher = cone_publisher
        self.covariance_P = covariance_P
        self.noise_Q = np.eye(3) * 1e-6
        self.extended_kalman_filter = ExtendedKalmanFilterRobot(State(self.pos_baselink.x, self.pos_baselink.y, self.theta), self.covariance_P, self.noise_Q)
        self.log_weight = 0.0
        self.ransac_draw_keypoints = []

    def publish_yourself(self, rgb_stamp):
        self.publish_odometry_msg(self.odometry_msg_publisher, self.pos_baselink, self.theta, self.covariance_P, rgb_stamp, VISUAL_ODOM_FRAME_ID, BASE_LINK_FRAME_ID)
        self.publish_pointcloud_map(self.map_publisher, rgb_stamp)
        self.publish_vision_cone(self.cone_publisher, rgb_stamp)

    """
    def update_with_wheel_odom(self, state_wheel_odom: State):
        
        
        if self.ransac_delta_p is not None:
            delta_ransac = State(self.ransac_delta_p.x, self.ransac_delta_p.y, self.ransac_delta_theta) 
        else:
            delta_ransac = None

        state, self.covariance_P = self.extended_kalman_filter.kalman_iteration(state_wheel_odom, delta_ransac, self.ransac_inlier_ratio)
        self.pos_baselink = Coordinate(state.x, state.y, 0.0)
        self.theta = state.theta

        if self.ransac_delta_p is not None:
            self.ransac_delta_p = None
            self.ransac_delta_theta = None
            self.ransac_inlier_ratio = 0.0
    """


    def robot_iteration(self, valid_kp: List[cv2.KeyPoint], valid_des: np.ndarray, valid_kp_depth: np.ndarray, frame_rgb, depth_frame, rgb_stamp):
        self.frame_rgb = frame_rgb
        self.frame_depth = depth_frame

        if self.first_iteration and len(self.visual_odom_map) < 50:
            self.first_iteration = False
            self.visual_odom_map.add_landmarks_from_kps(self.covariance_P, valid_kp, valid_des, self.frame_rgb, valid_kp_depth, self.theta, self.pos_baselink)
            return
        else:            
            pos_camera = kinect_depth_to_odom(Coordinate(0.0, 0.0, 0.0), self.theta,self.pos_baselink)

            self.visible_landmarks = self.visual_odom_map.get_visible_landmarks(pos_camera, self.theta)
            #self.visible_landmarks = self.visual_odom_map

            # Match descriptors.
            matches = self.bf.match(self.visible_landmarks.get_descriptors(), valid_des)

            if len(matches) < constants.parameters.min_matches_for_ransac:
                self.visual_odom_map.add_landmarks_from_kps(self.covariance_P, valid_kp, valid_des, self.frame_rgb, valid_kp_depth, self.theta, self.pos_baselink)
                return
            
            ransac_result = self.ransac(matches, self.visible_landmarks.get_odom_coordinates(), valid_kp, valid_kp_depth)
            self.ransac_delta_p = ransac_result[0]
            self.ransac_delta_theta = ransac_result[1]

            c = cos(self.theta)
            s = sin(self.theta)

            R = np.array([[c, -s],
                           [s,  c]])
            delta = R @ np.array([self.ransac_delta_p.x, self.ransac_delta_p.y])
            self.ransac_delta_p = Coordinate(delta[0], delta[1], 0.0)

            ransac_landmark_indices = ransac_result[2]
            self.ransac_draw_keypoints = ransac_result[3]
            ransac_kp_indices = ransac_result[4]
            ransac_not_matched_kp_indices = ransac_result[5]
            ransac_inlier_count = ransac_result[6]
            self.ransac_inlier_ratio = float(ransac_inlier_count)/len(matches) if len(matches) > 0 else 0.0

            if float(ransac_inlier_count)/len(matches) < constants.parameters.ransac_min_inlier_ratio:  # Weniger als x% Inlier nach RANSAC
                self.visual_odom_map.add_landmarks_from_kps(self.covariance_P, valid_kp, valid_des, self.frame_rgb, valid_kp_depth, self.theta, self.pos_baselink)
                rclpy.logging.get_logger("RANSAC").info(f"RANSAC result rejected due to low inlier ratio: {float(ransac_inlier_count)/len(matches):.2f} with {ransac_inlier_count} inliers out of {len(matches)} matches.")
                return
#
            kp_pos = []
            for i in ransac_kp_indices:
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
                for idx in ransac_not_matched_kp_indices:
                    not_matched_kp.append(valid_kp[idx])
                    not_matched_des.append(valid_des[idx])
                    not_matched_kp_depth.append(valid_kp_depth[idx])

                
                #self.visual_odom_map.add_landmarks_from_kps(self.covariance_P, valid_kp, valid_des, self.frame_rgb, valid_kp_depth, self.theta, self.pos_baselink)
                self.visual_odom_map.add_landmarks_from_kps(self.covariance_P, not_matched_kp, not_matched_des, self.frame_rgb, not_matched_kp_depth, self.theta, self.pos_baselink)


    def ransac(self, matches: List[cv2.DMatch], landmarks_odom_pos: List[Coordinate], valid_kp: List[cv2.KeyPoint], valid_kp_depth: np.ndarray) -> Tuple[Coordinate, float, List[int], List[int], List[int], List[int], int]:
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

        P_array = np.array(P)
        Q_array = np.array(Q)


        best_P_inlier = []
        best_Q_inlier = []
        best_valid_kp_index = []
        best_draw_Q_inlier = []
        best_landmark_index = []
        mean_last_e = 10000.0

        iteration = int(constants.parameters.ransac_iteration)

        #p = 0.99          # gewünschte Erfolgswahrscheinlichkeit
        #w = 0.7           # geschätzter Inlier-Anteil
        #s = 3             # Anzahl Samples pro RANSAC-Hypothese

        #k = int(math.log(1 - p) / math.log(1 - w**s))+1

        for iter in range(iteration): # k statt iteration
            P_samples = []
            Q_samples = []

            samples = random.sample(range(len(P)), constants.parameters.ransac_sample_size)
            P_samples = [P[i] for i in samples]
            Q_samples = [Q[i] for i in samples]

            R, t, theta  = kabsch(np.array(P_samples), np.array(Q_samples), constants.parameters.max_rotation_angle_deg)
            if R is None:
                #rclpy.logging.get_logger("RANSAC").info(f"RANSAC at iteration {iter} failed to compute transformation")
                continue
   
            e = np.linalg.norm(P_array - ((R @ Q_array.T).T + t.T), axis=1)
            mean_e = np.mean(e)

            inlier_count = np.sum(e < constants.parameters.ransac_evaluation_tolerance)

            if inlier_count > best_inlier_count or (inlier_count == best_inlier_count and mean_e < mean_last_e):
                best_inlier_count = inlier_count
                mean_last_e = mean_e

                best_P_inlier = []
                best_Q_inlier = []
                best_valid_kp_index = []
                best_draw_Q_inlier = []
                best_landmark_index = []
                
                for i in range(len(e)):
                    if e[i] < constants.parameters.ransac_evaluation_tolerance:
                        best_P_inlier.append(P_array[i])
                        best_Q_inlier.append(Q_array[i])
                        best_valid_kp_index.append(matches[i].trainIdx)
                        best_draw_Q_inlier.append(valid_kp[matches[i].trainIdx])
                        best_landmark_index.append(matches[i].queryIdx) 

            if (best_inlier_count/ len(matches)) > 0.80 or mean_e < constants.parameters.ransac_evaluation_tolerance*4:
                #rclpy.logging.get_logger("RANSAC").info(f"RANSAC early break at iteration {iter} with mean error {mean_e} and inlier count {inlier_count} and length matches {len(matches)}")
                break

            #if(iter == iteration-1):
                #rclpy.logging.get_logger("RANSAC").info(f"RANSAC finished all iterations. Best mean error: {mean_last_e} with inlier count {best_inlier_count} out of {len(matches)} matches.")

        if len(best_P_inlier) <  constants.parameters.min_matches_for_ransac:
            rclpy.logging.get_logger("RANSAC").info(f"RANSAC failed to find a valid transformation with enough inliers. Best inlier count: {best_inlier_count} out of {len(matches)} matches.")
            return Coordinate(0.0, 0.0, 0.0), 0.0, [], [], [], [], 0
        
        # Kabsch noch einmal mit den endgültigen besten Inliern berechnen
        R, t, theta  = kabsch(np.array(best_P_inlier), np.array(best_Q_inlier), constants.parameters.max_rotation_angle_deg)
            
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