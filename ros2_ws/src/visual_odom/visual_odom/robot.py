#!/usr/bin/env python3
import copy
import math
import random
import cv2

import numpy as np

from math import sin, cos, tan
from geometry_msgs.msg import Pose2D,  Point
from tf2_ros import Buffer, Header, TransformBroadcaster, TransformListener
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from rclpy.node import Publisher
from rclpy.time import Time

from scipy.spatial.transform import Rotation
from numpy.typing import NDArray
from typing import List

from visual_odom.landmark import *
from visual_odom.visual_odom_map import *
from visual_odom.tf_methods import *
from visual_odom.constants import *
from visual_odom.ekf_robot import *
import visual_odom.constants as constants


class VisualRobotSample():
    def __init__(self, pos: Coordinate, theta: float, covariance_P: NDArray, matcher: cv2.BFMatcher, odom_publisher:Publisher, map_publisher:Publisher, cone_publisher:Publisher, total_publisher:Publisher, valid_kp: List[cv2.KeyPoint], valid_des: np.ndarray, valid_kp_depth: np.ndarray, frame_rgb: NDArray, index:int, map = None):        
        # Initialize argument values
        self.theta = theta 
        self.pos_baselink = pos
        self.index = index
        if map is None:
            map = VisualOdomMap()
        self.visual_odom_map = map
        self.bf = matcher
        self.odometry_msg_publisher = odom_publisher
        self.map_publisher = map_publisher
        self.cone_publisher = cone_publisher
        self.covariance_P = covariance_P
        self.total_publisher = total_publisher

        # Initialize start values
        self.first_iteration = True
        self.ransac_delta_p = None
        self.ransac_inlier_ratio = 0.0
        self.last_wheel_odom = None
        self.log_weight = 0.0
        self.noise_Q = np.eye(3) * 1e-6
        self.ransac_draw_keypoints = []
        self.path = []
        self.accumulated_pixels = []
        self.delta_wheel_odom_ransac = State(0.0, 0.0, 0.0)
        self.acc_ransac_noise_delta = State(0.0, 0.0, 0.0)

        self.pos_visual_odom = State(self.pos_baselink.x, self.pos_baselink.y, self.theta)
        self.ekf = ExtendedKalmanFilterRobot(State(self.pos_baselink.x, self.pos_baselink.y, self.theta), self.covariance_P, self.noise_Q)
        self.visual_odom_map.add_landmarks_from_kps(covariance_P, valid_kp, valid_des, frame_rgb, valid_kp_depth, self.theta, self.pos_baselink)
        
        marker = Marker()
        # Initialize random color for path markers; red is reserved for best robot
        marker.color.r = random.uniform(0.0, 0.8)
        marker.color.g = random.uniform(0.3, 1.0)
        marker.color.b = random.uniform(0.3, 1.0)

        marker.color.a = 1.0
        self.path_color = marker.color


    def publish_yourself(self, rgb_stamp):
        """!
        @brief Publishes all visualization and odometry messages for the particle.

        @param rgb_stamp Timestamp of the current frame.
        @return None
        """
        self.publish_odometry_msg(self.odometry_msg_publisher, self.pos_baselink, self.theta, self.covariance_P, rgb_stamp, VISUAL_ODOM_FRAME_ID, BASE_LINK_FRAME_ID)
        self.publish_pointcloud_map(self.map_publisher, rgb_stamp)
        self.publish_vision_cone(self.cone_publisher, rgb_stamp)
        self.publish_pixels(self.frame_depth, self.frame_rgb, self.total_publisher, rgb_stamp)


    def predict_with_wheel_odom(self, state_wheel_odom: State):
        """!
        @brief Updates the EKF prediction step using wheel odometry increments.

        @param state_wheel_odom Current wheel odometry state.
        @return None
        """ 
        if self.last_wheel_odom is None:
            self.last_wheel_odom = state_wheel_odom
            return
        delta_wheel_odom = state_wheel_odom - self.last_wheel_odom
        self.ekf.prediction(delta_wheel_odom)
        self.last_wheel_odom = state_wheel_odom

        self.delta_wheel_odom_ransac += delta_wheel_odom


    def robot_iteration(self, valid_kp: List[cv2.KeyPoint], valid_des: np.ndarray, valid_kp_depth: np.ndarray, frame_rgb, depth_frame, rgb_stamp, ransac_result):
        """
        @brief Executes one visual odometry iteration including feature matching, RANSAC validation, EKF fusion, and landmark map updates.

        @param valid_kp List of valid ORB keypoints extracted from the current RGB frame.
        @param valid_des Descriptor matrix corresponding to the detected keypoints.
        @param valid_kp_depth Depth values for each detected keypoint.
        @param frame_rgb Current RGB image frame.
        @param depth_frame Current aligned depth image.
        @param rgb_stamp ROS timestamp of the current RGB frame.
        @param ransac_result Precomputed RANSAC motion estimation and inlier information.
        @return None. Updates robot pose, covariance, landmark map, and trajectory in-place.
        """
        
        self.frame_rgb = frame_rgb
        self.frame_depth = depth_frame
        # Skip iteration if not enough landmarks are in the map
        if self.first_iteration and len(self.visual_odom_map) < INITIAL_LANDMARK_COUNT:
            self.visual_odom_map.add_landmarks_from_kps(self.covariance_P, valid_kp, valid_des, self.frame_rgb, valid_kp_depth, self.theta, self.pos_baselink)
            return
        else:       
            self.first_iteration = False   
            pos_camera = kinect_depth_to_odom(Coordinate(0.0, 0.0, 0.0), self.theta,self.pos_baselink)
            self.visible_landmarks = self.visual_odom_map.get_visible_landmarks(pos_camera, self.theta)
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
            
            # Extract RANSAC results
            self.ransac_delta_p = ransac_result[0]
            self.ransac_delta_theta = ransac_result[1]
            self.ransac_draw_keypoints = ransac_result[2]
            ransac_inlier_count = ransac_result[3]
            self.ransac_inlier_ratio = float(ransac_inlier_count)/len(matches) if len(matches) > 0 else 0.0
            ransac_landmark_indices = landmark_indices

            # Skip iteration if visual odom is not sufficient
            if self.ransac_inlier_ratio < constants.parameters.ransac_min_inlier_ratio:
                self.visual_odom_map.add_landmarks_from_kps(self.covariance_P, valid_kp, valid_des, self.frame_rgb, valid_kp_depth, self.theta, self.pos_baselink)
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
            
            # Add noise offset to x-y-state after certain translation 
            if math.sqrt(self.acc_ransac_noise_delta.x**2 + self.acc_ransac_noise_delta.y**2) > NOISE_TRANSLATION_THRESHOLD:
                noise_x = np.random.normal(0.0, GAUSS_NOISE_X_SIGMA)
                noise_y = np.random.normal(0.0, GAUSS_NOISE_Y_SIGMA)
                self.delta_ransac_state.x += noise_x
                self.delta_ransac_state.y += noise_y

                self.acc_ransac_noise_delta.x = 0.0
                self.acc_ransac_noise_delta.y = 0.0
            
            # Add noise offset to x-y-state after certain rotation 
            if abs(self.acc_ransac_noise_delta.theta) > NOISE_ROTATION_THRESHOLD:
                noise_theta = np.random.normal(0.0, GAUSS_NOISE_THETA_SIGMA)
                self.delta_ransac_state.theta += noise_theta
                self.acc_ransac_noise_delta.theta = 0.0

            # Assemble new state and update with Kalman-Filter
            new_state = self.pos_visual_odom + self.delta_ransac_state
            new_state.theta = normalize_angle(new_state.theta)
            self.pos_visual_odom = new_state

            self.ekf.update(self.pos_visual_odom)
            self.delta_wheel_odom_ransac = State(0.0, 0.0, 0.0)
            state = self.ekf.get_state()
            self.pos_baselink = Coordinate(state.x, state.y, 0.0)
            self.theta = state.theta
            self.covariance_P = self.ekf.get_covariance_p()

            kp_pos = []
            for i in matched_kp_index:
                u, v = valid_kp[i].pt
                u = int(round(u))
                v = int(round(v))
                
                depth_value = self.frame_depth[v, u]
                kp_pos.append(PixelCoordinate(u, v, depth_value))
            
            # Prepare for next iteration with weight calculation for particle filter, landmark kalman filtering and map maintenance
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
            
            # Fill map with landmarks and add path marker
            self.accumulate_pixels(self.frame_depth, self.frame_rgb)
            self.visual_odom_map.add_landmarks_from_kps(self.covariance_P, not_matched_kp, not_matched_des, self.frame_rgb, not_matched_kp_depth, self.theta, self.pos_baselink)
            self.path.append(Pose2D(x=self.pos_baselink.x, y=self.pos_baselink.y))

    def ransac_failed(self):
        """
        @brief Fallback EKF update when RANSAC validation fails.

        @return None. Updates pose, covariance, and internal state in-place.
        """
        new_state = self.pos_visual_odom + self.delta_wheel_odom_ransac
        new_state.theta = normalize_angle(new_state.theta)
        self.pos_visual_odom = new_state

        self.ekf.add_noise_to_R()
        self.ekf.update(self.pos_visual_odom)
        state = self.ekf.get_state()
        self.covariance_P = self.ekf.get_covariance_p()
        self.pos_baselink = Coordinate(state.x, state.y, 0.0)
        self.theta = state.theta
        
        self.delta_wheel_odom_ransac = State(0.0, 0.0, 0.0)


    def publish_odometry_msg(self, publisher, pos: Coordinate, theta: float, covariance: NDArray, timestamp, parent_frame_id: str, child_frame_id: str):
        """
        @brief Publishes the robot pose and covariance as a ROS odometry message.

        @param publisher ROS publisher used to publish the odometry message.
        @param pos Current robot position in odometry coordinates.
        @param theta Current robot yaw orientation.
        @param covariance 3x3 robot pose covariance matrix.
        @param timestamp ROS timestamp for the odometry message.
        @param parent_frame_id Parent coordinate frame ID.
        @param child_frame_id Child coordinate frame ID.
        @return None.
        """
        msg = Odometry()

        msg.header.stamp = timestamp
        msg.header.frame_id = parent_frame_id
        msg.child_frame_id = child_frame_id

        msg.pose.pose.position.x = pos.x
        msg.pose.pose.position.y = pos.y
        msg.pose.pose.position.z = pos.z

        euler = Rotation.from_euler('z', float(theta))
        quat = euler.as_quat(canonical=True)
        msg.pose.pose.orientation.x = quat[0]
        msg.pose.pose.orientation.y = quat[1]
        msg.pose.pose.orientation.z = quat[2]
        msg.pose.pose.orientation.w = quat[3]

        msg.pose.covariance = [
           covariance[0, 0], covariance[0, 1],          0.0,    0.0,    0.0,   covariance[0, 2],
            covariance[1, 0], covariance[1, 1],         0.0,    0.0,    0.0,    covariance[1, 2],
            0.0,                0.0,                    0.0,    0.0,    0.0,    0.0,
            0.0,                0.0,                    0.0,    0.0,    0.0,    0.0,
            0.0,                0.0,                    0.0,    0.0,    0.0,    0.0,
            covariance[2, 0],    covariance[2, 1],      0.0,    0.0,    0.0,    covariance[2, 2]
        ]
        publisher.publish(msg)

    def publish_vision_cone(self, cone_publisher:Publisher, rgb_stamp:Time):
        """
        @brief Publishes the camera field-of-view cone as a visualization marker.

        @param cone_publisher ROS publisher for the vision cone marker.
        @param rgb_stamp Timestamp of the current RGB frame.
        """
        cone_msg = self.init_camera_cone(rgb_stamp)
        cone_publisher.publish(cone_msg)

    def init_camera_cone(self, rgb_stamp):
        """
        @brief Creates a triangular marker representing the camera vision cone.

        @param rgb_stamp Timestamp used for marker synchronization.
        @return ROS marker imitating the camera FOV.
        """
        cone_msg = Marker()
        cone_msg.header.frame_id = "kinect_depth"
        cone_msg.header.stamp = rgb_stamp
        cone_msg.ns = "vision_cone"
        cone_msg.id = 0

        cone_msg.type = Marker.TRIANGLE_LIST
        cone_msg.action = Marker.ADD
        cone_msg.scale.x = 1.0
        cone_msg.scale.y = 1.0
        cone_msg.scale.z = 1.0

        cone_msg.color.r = 0.0
        cone_msg.color.g = 1.0
        cone_msg.color.b = 0.0
        cone_msg.color.a = 0.3
              
        h = tan(CAMERA_ANGLE_VER_RAD / 2)
        w = tan(CAMERA_ANGLE_HOR_RAD / 2)
        d = constants.parameters.max_depth/1000
        
        #Center Point of Cone
        p0 = Point()
        p0.x = 0.0 
        p0.y = 0.0
        p0.z = 0.0

        #Lower Right Point
        p1 = Point()
        p1.z = d 
        p1.x = d * w
        p1.y = d * h

        #Lower Left Point
        p2 = Point()
        p2.z = d 
        p2.x = -d * w 
        p2.y = d * h

        #Upper Right Point
        p3 = Point()
        p3.z = d
        p3.x = d * w
        p3.y = -d * h

        #Upper Left Point
        p4 = Point()
        p4.z = d
        p4.x = -d * w
        p4.y = -d * h

        #Fill array in triples for cone construction made of triangles
        cone_msg.points = []
        cone_msg.points.extend([p0, p1, p3, 
                                p0, p4, p2,
                                p0, p2, p1,
                                p0, p3, p4,
                                p1, p2, p4,
                                p4, p3, p1])        
    
        return cone_msg
    
    def publish_pointcloud_map(self, publisher: Publisher, time: Time):
        """
        @brief Calls function to publish active landmarks as a standard PointCloud2 sensor message for RViz.

        @param publisher ROS publisher for the point cloud map.
        @param time Timestamp used for point cloud synchronization.
        """
        self.visual_odom_map.publish_pointcloud_map(publisher, time)
    
    def publish_pixels(self, frame_depth: NDArray, frame_rgb: NDArray, publisher: rclpy.publisher.Publisher, time: Time) -> None:
        """!
        @brief Downsamples and formats dense pixel coordinates into PointCloud2 structures.

        @param frame_depth Input depth reference image.
        @param frame_rgb Color reference frame image.
        @param publisher Target ROS 2 PointCloud2 publisher channel.
        @param time Active system timestamp.
        @param frame_id Relative spatial coordinate frame identifier.
        """
        calculated_point_coordinates = []
        divisor = 3 # Only publish every "divisor" Pixel
        
        for pix_u in range(frame_depth.shape[1] // divisor):
            for pix_v in range(frame_depth.shape[0] // divisor):
                depth_value = frame_depth[pix_v * divisor, pix_u * divisor]
                if MIN_DEPTH < depth_value < constants.parameters.max_depth:
                    pos = pixel_to_kinect(PixelCoordinate(pix_u * divisor, pix_v * divisor, depth_value))
                    b, g, r = frame_rgb[pix_v * divisor, pix_u * divisor]
                    rgb = (int(r) << 16) | (int(g) << 8) | int(b)
                    calculated_point_coordinates.append((pos.x, pos.y, pos.z, rgb))

        h = Header()
        h.stamp = time
        h.frame_id = KINECT_FRAME_ID

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
        #self.get_logger().info(f"PointCloud with {len(calculated_point_coordinates)} points sent!")

    def publish_accumulated_pixels(self, publisher: Publisher, time: Time):
        
        h = Header()
        h.stamp = time
        h.frame_id = KINECT_FRAME_ID

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
        ]

        msg = point_cloud2.create_cloud(
            header=h,
            fields=fields,
            points = self.accumulated_pixels
        )
        publisher.publish(msg)

    def accumulate_pixels(self, frame_depth: NDArray, frame_rgb: NDArray) -> None:
        """!
        @brief Accumulate pixels for every robot to publish depending on best robot

        @param frame_depth Input depth reference image.
        @param frame_rgb Color reference frame image.
        @param publisher Target ROS 2 PointCloud2 publisher channel.
        @param time Active system timestamp.
        """
        divisor = 7 # Only publish every "divisor" Pixel
        
        for pix_u in range(frame_depth.shape[1] // divisor):
            for pix_v in range(frame_depth.shape[0] // divisor):
                depth_value = frame_depth[pix_v * divisor, pix_u * divisor]
                if MIN_DEPTH < depth_value < constants.parameters.max_depth:
                    pos = pixel_to_kinect(PixelCoordinate(pix_u * divisor, pix_v * divisor, depth_value))
                    b, g, r = frame_rgb[pix_v * divisor, pix_u * divisor]
                    rgb = (int(r) << 16) | (int(g) << 8) | int(b)
                    self.accumulated_pixels.append((pos.x, pos.y, pos.z, rgb))

    def get_drawn_keypoints(self):
        """
        @brief Returns the keypoints used as RANSAC inliers for visualization.

        @return List of matched keypoints.
        """
        return self.ransac_draw_keypoints
        
    def get_position(self) -> Coordinate:
        """
        @brief Returns the current robot position.

        @return Current base_link position.
        """
        return self.pos_baselink
    
    def get_theta(self) -> float:
        """
        @brief Returns the current robot heading.

        @return Current yaw angle in radians.
        """
        return self.theta
    
    def get_covariance_P(self) -> NDArray:
        """
        @brief Returns the current pose covariance matrix.

        @return 3x3 covariance matrix.
        """
        return self.covariance_P
    
    def get_log_weight(self) -> float:
        """
        @brief Returns the particle log weight.

        @return Current log weight value.
        """
        return self.log_weight
    
    def set_log_weight(self, log_weight: float) -> None:
        """
        @brief Sets the particle log weight.

        @param log_weight New log weight value.
        @return None.
        """
        self.log_weight = log_weight

    def get_weight(self) -> float:
        """
        @brief Returns the particle weight in linear scale.

        @return Particle weight or None if overflow occurs.
        """
        try:
            return math.exp(self.log_weight)
        except OverflowError:
            return None
    
    def set_weight(self, weight: float) -> None:
        """
        @brief Sets the particle weight using linear scale.

        @param weight New particle weight.
        @return None.
        """
        self.log_weight = math.log(weight)

    def set_state(self, pos: Coordinate, theta: float) -> None:
        """!
        @brief Sets the current robot base tracking pose.

        @param pos New estimated position of the robot base.
        @param theta New estimated heading of the robot base.
        """
        self.pos_baselink = State(pos.x, pos.y, theta)
        self.pos_baselink.z = 0.0
        self.theta = theta
        self.ekf.set_state(State(pos.x, pos.y, theta))
