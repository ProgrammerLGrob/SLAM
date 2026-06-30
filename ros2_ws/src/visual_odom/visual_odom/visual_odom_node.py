#!/usr/bin/env python3

"""!
@file visual_odom_node.py
@package visual_odom.visual_odom_node
@brief Main ROS 2 execution node managing the particle filter localization and visual odometry loop.
"""

import math

import builtin_interfaces.msg
from tf2_ros import Buffer, Header, TransformBroadcaster, TransformListener
from geometry_msgs.msg import Point
from scipy.spatial.transform import Rotation
from numpy.typing import NDArray
from rclpy.time import Time
from visualization_msgs.msg import Marker, MarkerArray
from math import cos, pi, sin
import random
from typing import List, Tuple
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from sensor_msgs_py import point_cloud2
import cv2
from cv_bridge import CvBridge
import numpy as np

from visual_odom.landmark import kabsch
from visual_odom.tf_methods import pixel_to_kinect, pixel_to_kinect_, kinect_depth_to_baselink, calculate_tf, kinect_depth_to_odom, odom_to_baselink
from visual_odom.constants import (
    Coordinate, PixelCoordinate, State, Parameters,
    RGB_IMAGE_TOPIC, DEPTH_IMAGE_TOPIC, WHEEL_ODOMETRY_TOPIC, VISUAL_ODOM_PATH_TOPIC,
    KEYPOINT_POINTCLOUD_FRAME_TOPIC, POINTCLOUD_FRAME_TOPIC,
    VISION_CONE_TOPIC, KP_IMAGE_TOPIC, VISUAL_ODOM_FRAME_ID,
    BASE_LINK_FRAME_ID, MIN_DEPTH, MAX_DEPTH, RANSAC_EVALUATION_TOLERANCE,
    RANSAC_ITERATION, RANSAC_SAMPLE_SIZE, RGB_DEPTH_SYNC_TOLERANCE_SEC,
    PIXEL_TOLERANCE, MATCHES_FOR_NEW_LANDMARKS, MIN_MATCHES_FOR_RANSAC,
    MAX_ROTATION_ANGLE_DEG, MIN_LANDMARK_TRUST, RANSAC_MIN_INLIER_RATIO,
    N_ROBOT_SAMPLES, VISUAL_ODOM_MSG_TOPIC, RESAMPLE_THETA_TOLERANCE, RESAMPLE_TIME_TOLERANCE, RESAMPLE_POS_TOLERANCE, normalize_angle
)
from visual_odom.robot import VisualRobotSample
import visual_odom.constants as constants

from nav_msgs.msg import Odometry
from rcl_interfaces.msg import SetParametersResult


class VisualOdom(Node):
    """!
    @brief ROS 2 Visual Odometry node managing features extraction, RANSAC estimation, and particle weight selection.
    """

    def __init__(self):
        """!
        @brief Initializes the visual odometry node, subscribers, publishers, and ORB keypoint detectors.
        """
        super().__init__('visual_odom')

        # Load parameters from parameter file
        constants.parameters = self.parameter_initialization()
        self.add_on_set_parameters_callback(self.parameter_callback)
        
        # Initilize ORB feature detector
        self.bridge = CvBridge()
        self.orb = cv2.ORB_create(
            nfeatures=1500,        # Maximum number of keypoints to detect
            edgeThreshold=30,      # Safety border threshold
            patchSize=30,          # Size of the patch for descriptors
            fastThreshold=5,       # FAST detector sensitivity
            scoreType=cv2.ORB_HARRIS_SCORE
        )
        
        # Create BFMatcher object
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Initialize subscribers
        self.subscription_rgb = self.create_subscription(Image, RGB_IMAGE_TOPIC, self.listener_rgb_callback, 10)
        self.subscription_depth = self.create_subscription(Image, DEPTH_IMAGE_TOPIC, self.listener_depth_callback, 10)
        self.subscription_wheel_odom = self.create_subscription(Odometry, WHEEL_ODOMETRY_TOPIC, self.listener_wheel_odom_callback, 10)

        # Initialize publishers for ROS topics
        self.publisher_keypoints_3d = self.create_publisher(PointCloud2, KEYPOINT_POINTCLOUD_FRAME_TOPIC, 10)
        self.total_publisher = self.create_publisher(PointCloud2, POINTCLOUD_FRAME_TOPIC, 10)
        self.publisher_visual_odometry_msg = self.create_publisher(Odometry, constants.parameters.topic_visual_odometry_msg, 10)
        self.publisher_cone = self.create_publisher(Marker, VISION_CONE_TOPIC, 10)
        self.publisher_image = self.create_publisher(Image, KP_IMAGE_TOPIC, 10)
        self.publisher_accumulated_pixels = self.create_publisher(Image, KP_IMAGE_TOPIC, 10)
        self.visual_odom_path = self.create_publisher(MarkerArray, VISUAL_ODOM_PATH_TOPIC, 10)

        # Initialize transform parameters
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Initialize global parameters
        self.frame_depth = None
        self.frame_depth_stamp = None
        self.frame_rgb = None
        self.theta = 0.0 
        self.valid_kp_last = None
        self.valid_des_last = None
        self.valid_kp_depth_last = None
        self.valid_kp = None
        self.valid_des = None
        self.first_kp = None
        self.first_des = None
        self.pos_baselink = Coordinate(0.0, 0.0, 0.0) # Robot position in odom frame
        self.robots: List[VisualRobotSample] = []
        self.best_robot_idx = 0
        self.best_weight = 0.0

        # Initial value allocation
        self.first_iteration = True
        self.position_initialized = False
        sigma_x0 = 0.1   # X translation initial uncertainty in meters
        sigma_y0 = 0.1   # Y translation initial uncertainty in meters
        sigma_th0 = 0.05  # Yaw rotation initial uncertainty in radians (~3 deg)
        self.covariance_P = np.diag([sigma_x0**2, sigma_y0**2, sigma_th0**2])
        
        # Completion of node init
        self.get_logger().info("Visual Odometry Node initialized and ready to receive RGB-D camera frames.")
        self.frame_stamp = builtin_interfaces.msg.Time()

    def listener_wheel_odom_callback(self, msg: Odometry) -> None:
        """!
        @brief Receives wheel odometry pose measurements and initializes baseline tracking.

        @param msg Incoming wheel odometry message.
        """

        # Extract orientation and pose from wheel_msg 
        rot = Rotation.from_quat([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
        state_wheel_odom = State(msg.pose.pose.position.x, msg.pose.pose.position.y, rot.as_euler('xyz')[2])
        
        # Set state on frist iteration, else predict next state
        if not self.position_initialized:
            self.position_initialized = True
            self.pos_baselink = Coordinate(state_wheel_odom.x, state_wheel_odom.y, 0.0)
            self.theta = state_wheel_odom.theta         
            return
        else:
            if not self.first_iteration:
                for robot in self.robots:
                    robot.predict_with_wheel_odom(state_wheel_odom)

    def listener_rgb_callback(self, msg: Image) -> None:
        """!
        @brief Handles primary RGB camera frames. Triggers visual processing loops and publishes transforms.

        @param msg Incoming RGB reference image.
        """
        # Ignore update with RGB until state is initialized in listener_wheel_odom_callback
        if not self.position_initialized:
            return
        
        # Return if messages are out of sync
        if self._stamp_to_sec(msg.header.stamp) - self._stamp_to_sec(self.frame_stamp) > 0.000:
            self.frame_stamp = msg.header.stamp
        else:
            return
        
        # Extract image data from RGB image
        self.valid_kp, self.valid_des, self.valid_kp_depth, rgb_stamp = self.img_to_kp_des_filtered(msg)
        if self.valid_kp is None or self.valid_des is None or self.valid_kp_depth is None:
            return 

        # If no data to compare to, save current values for next frame and skip further steps
        if self.valid_kp_last is None and self.valid_des_last is None:
            self.valid_kp_last = self.valid_kp
            self.valid_des_last = self.valid_des
            self.valid_kp_depth_last = self.valid_kp_depth

            self.first_kp = self.valid_kp
            self.first_des = self.valid_des
            self.first_kp_depth = self.valid_kp_depth
            self.first_frame_stamp = rgb_stamp
            return

        #On first iteration, initialize robot particles in starting coordinates
        if self.first_iteration:
            self.first_iteration = False
            odom_to_base_footprint = calculate_tf(self.pos_baselink, self.theta, rgb_stamp, VISUAL_ODOM_FRAME_ID, BASE_LINK_FRAME_ID)
            self.tf_broadcaster.sendTransform(odom_to_base_footprint)

            self.first_theta = self.theta
            self.first_pos = self.pos_baselink

            for i in range(constants.parameters.n_robot_samples):
                self.robots.append(VisualRobotSample(self.pos_baselink, self.theta, self.covariance_P, self.bf, self.publisher_visual_odometry_msg, self.publisher_keypoints_3d, self.publisher_cone,self.total_publisher, self.valid_kp, self.valid_des, self.valid_kp_depth, self.frame_rgb, i))
        else:
            matches = self.bf.match(self.valid_des_last, self.valid_des)
            ransac_result = self.ransac_frame_to_frame(matches, self.valid_kp, self.valid_kp_depth, self.valid_kp_last, self.valid_kp_depth_last)

            self.iteration(self.valid_kp, self.valid_des, self.valid_kp_depth, self.frame_rgb, self.frame_depth, rgb_stamp, ransac_result)  
            
            # Get best position data, format to messages and publish to RViz
            self.pos_baselink = self.best_robot.get_position()
            self.theta = self.best_robot.get_theta()
            self.covariance_P = self.best_robot.get_covariance_P()
            
            odom_to_base_footprint = calculate_tf(self.pos_baselink, self.theta, rgb_stamp, VISUAL_ODOM_FRAME_ID, BASE_LINK_FRAME_ID)
            self.tf_broadcaster.sendTransform(odom_to_base_footprint)          
            frame_rgb_drawn = cv2.drawKeypoints(self.frame_rgb, self.best_robot.get_drawn_keypoints(), None, color=(0, 255, 0), flags=0)
            msg_out = self.bridge.cv2_to_imgmsg(frame_rgb_drawn, encoding='bgr8')
            
            self.best_robot.publish_yourself(rgb_stamp) 
            self.publisher_image.publish(msg_out)
            self.visual_odom_path.publish(create_particle_path_markers(self, self.robots, self.best_robot_idx))
            
            # Preparations for next iteration
            self.valid_kp_last = self.valid_kp
            self.valid_des_last = self.valid_des
            self.valid_kp_depth_last = self.valid_kp_depth

            self.possible_resample(self.robots, self.best_robot_idx, self.first_frame_stamp)
            

            

    def possible_resample(self, robots: List[VisualRobotSample], best_robot_idx: int, first_frame_stamp: float) -> None:
        """!
        @brief Performs resampling by matching visible landmarks to the first frame.

        @param robots List of active robot particles.
        @param best_robot_idx Index of the best particle hypothesis.
        @param first_frame_stamp Timestamp of the first recorded frame.
        @return None
        """
        # Get best robot and check for resampling criteria
        best_robot = robots[best_robot_idx]
        #rclpy.logging.get_logger("VisualOdom").info(f"abs(best_robot.theta-self.first_theta) = {abs(best_robot.theta-self.first_theta)} and time since first frame: {self._stamp_to_sec(self.frame_stamp) - self._stamp_to_sec(first_frame_stamp)}")
        
        if abs(best_robot.theta-self.first_theta) < RESAMPLE_THETA_TOLERANCE*pi/180.0 and self._stamp_to_sec(self.frame_stamp) - self._stamp_to_sec(first_frame_stamp) > RESAMPLE_TIME_TOLERANCE and math.sqrt((best_robot.pos_baselink.x-self.first_pos.x)**2 + (best_robot.pos_baselink.y-self.first_pos.y)**2) > RESAMPLE_POS_TOLERANCE:
            # Compare visible landmarks from map to current frame
            pos_camera = kinect_depth_to_odom(Coordinate(0.0, 0.0, 0.0), best_robot.theta, best_robot.pos_baselink)
            self.visible_landmarks = best_robot.visual_odom_map.get_visible_landmarks(pos_camera, best_robot.theta)
            matches = self.bf.match(self.visible_landmarks.get_descriptors(), self.first_des)
            ransac_result = self.ransac_frame_to_map(matches, self.visible_landmarks.get_odom_coordinates(), self.first_kp, self.first_kp_depth)

            # Calculate delta values
            ransac_delta_p = ransac_result[0]
            c = cos(best_robot.theta)
            s = sin(best_robot.theta)
            R = np.array([[c, -s],
                           [s,  c]])
            delta = R @ np.array([ransac_delta_p.x, ransac_delta_p.y])
            ransac_delta_p = Coordinate(delta[0], delta[1], 0.0)

            # Calculate new best robot pose and publish
            best_robot.pos_baselink += ransac_delta_p
            best_robot.theta += ransac_result[1]
            best_robot.theta = normalize_angle(best_robot.theta)
            
            best_robot.set_state(best_robot.pos_baselink, best_robot.theta)
            best_robot.publish_yourself(self.frame_stamp)
            best_robot.publish_accumulated_pixels(self.publisher_accumulated_pixels, self.frame_stamp)
            rclpy.logging.get_logger("VisualOdom").info(f"Resampling robot particles based on RANSAC alignment with first frame. delta position: {ransac_delta_p}, delta theta: {ransac_result[1]}, New position: {best_robot.pos_baselink}, New theta: {best_robot.theta}")

            self.first_frame_stamp = self.frame_stamp

    def ransac_frame_to_map(self, matches: List[cv2.DMatch], landmarks_odom_pos: List[Coordinate], valid_kp: List[cv2.KeyPoint], valid_kp_depth: np.ndarray) -> Tuple[Coordinate, float, List[int], int]:
        """!
        @brief Estimates the transformation between map landmarks and the current frame using RANSAC.

        @param matches Descriptor matches between map landmarks and current frame features.
        @param landmarks_odom_pos Landmark positions in odom frame.
        @param valid_kp Valid keypoints from current frame.
        @param valid_kp_depth Depth values of current frame keypoints.
        @return Estimated translation, rotation, inlier keypoints, and number of inliers.
        """
            # Variable Initialization
        P = []
        Q = []
        landmark_index = []

        best_inlier_count = 0
        best_t = Coordinate(0.0, 0.0, 0.0)
        best_theta = 0.0 

        # Calculate P and Q values for every match
        for m in range(len(matches)):            

            keypoint = valid_kp[matches[m].trainIdx].pt
            u, v = keypoint
            u = round(u)
            v = round(v)
            z = valid_kp_depth[matches[m].trainIdx]

            coor_landmark = landmarks_odom_pos[matches[m].queryIdx]
            P.append([coor_landmark.x, coor_landmark.y])

            coor = pixel_to_kinect(PixelCoordinate(u, v, z))
            coor_base_link = kinect_depth_to_baselink(coor)
            Q.append([coor_base_link.x, coor_base_link.y])

            landmark_index.append(matches[m].queryIdx)
        
        # Execute RANSAC
        best_t, best_theta, best_draw_Q_inlier, best_inlier_count = self.ransac_calculation(P, Q, matches, valid_kp)
        
        # Return best values form RANSAC Calculation
        return best_t, best_theta, best_draw_Q_inlier, best_inlier_count

    def ransac_frame_to_frame(self, matches: List[cv2.DMatch], valid_kp: List[cv2.KeyPoint], valid_kp_depth: np.ndarray, valid_kp_last: List[cv2.KeyPoint], valid_kp_depth_last: np.ndarray) -> Tuple[Coordinate, float, List[cv2.KeyPoint], int]:
        """!
        @brief Estimates translation and rotation between consecutive frames using RANSAC and 2D Kabsch point set alignment.

        @param matches Vector list of descriptor matches between consecutive visual frames.
        @param valid_kp Feature coordinates extracted from the current frame.
        @param valid_kp_depth Pixel depth values matching current features.
        @param valid_kp_last Feature coordinates extracted from the last frame.
        @param valid_kp_depth_last Pixel depth values matching last frame features.
        @return A tuple of (estimated Coordinate translation, rotation yaw, visual inlier keypoints, total inliers).
        """

        # Variable Initialization
        P = []
        Q = []
        landmark_index = [] 

        best_inlier_count = 0
        best_t = Coordinate(0.0, 0.0, 0.0)
        best_theta = 0.0

        # Calculate P and Q values for every match
        for m in range(len(matches)): 
            keypoint = valid_kp[matches[m].trainIdx].pt
            keypoint_last = valid_kp_last[matches[m].queryIdx].pt
            
            u, v = keypoint_last
            u = round(u)
            v = round(v)
            z = valid_kp_depth_last[matches[m].queryIdx]

            coor = pixel_to_kinect_(u, v, z)
            coor = Coordinate(coor[0], coor[1], coor[2])
            coor_base_link = kinect_depth_to_baselink(coor)
           
            P.append([coor_base_link.x, coor_base_link.y])

            u, v = keypoint
            u = round(u)
            v = round(v)
            z = valid_kp_depth[matches[m].trainIdx]

            coor = pixel_to_kinect_(u, v, z)
            coor = Coordinate(coor[0], coor[1], coor[2])
            coor_base_link = kinect_depth_to_baselink(coor)

            Q.append([coor_base_link.x, coor_base_link.y])
            landmark_index.append(matches[m].queryIdx)
        
        # Execute RANSAC
        best_t, best_theta, best_draw_Q_inlier, best_inlier_count = self.ransac_calculation(P, Q, matches, valid_kp)
        
        # Return best values form RANSAC Calculation
        return best_t, best_theta, best_draw_Q_inlier, best_inlier_count

    def ransac_calculation(self, P: List, Q: List, matches: List[cv2.DMatch], valid_kp: List[cv2.KeyPoint]):
        """!
        @brief Executes the core RANSAC loop and computes the best transformation using Kabsch.

        @param P Reference point set.
        @param Q Observed point set.
        @param matches Descriptor matches corresponding to P/Q pairs.
        @param valid_kp Valid keypoints used for visualization of inliers.
        @return Best translation, rotation, inlier keypoints, and number of inliers.
        """
        # Variable Initialization
        iteration = int(constants.parameters.ransac_iteration)
        mean_last_e = 10000.0
        
        P_array = np.array(P)
        Q_array = np.array(Q)

        best_P_inlier = []
        best_Q_inlier = []
        best_draw_Q_inlier = []
        best_inlier_count = 0

        # Calculate Kabsch for random samples (random P/Q pair) and reproject, saving best translation and rotation
        for iter in range(iteration):
            samples = random.sample(range(len(P)), constants.parameters.ransac_sample_size)
            P_samples = [P[i] for i in samples]
            Q_samples = [Q[i] for i in samples]

            R, t, theta = kabsch(np.array(P_samples), np.array(Q_samples), constants.parameters.max_rotation_angle_deg)
            if R is None:
                continue
   
            e = np.linalg.norm(P_array - ((R @ Q_array.T).T + t.T), axis=1)
            mean_e = np.mean(e)
            inlier_count = np.sum(e < constants.parameters.ransac_evaluation_tolerance)

            if inlier_count > best_inlier_count or (inlier_count == best_inlier_count and mean_e < mean_last_e):
                best_inlier_count = inlier_count
                mean_last_e = mean_e

                best_P_inlier = []
                best_Q_inlier = []
                best_draw_Q_inlier = []
                
                for i in range(len(e)):
                    if e[i] < constants.parameters.ransac_evaluation_tolerance:
                        best_P_inlier.append(P_array[i])
                        best_Q_inlier.append(Q_array[i])
                        best_draw_Q_inlier.append(valid_kp[matches[i].trainIdx])

            if (best_inlier_count / len(matches)) > constants.parameters.ransac_min_inlier_ratio or mean_e < constants.parameters.ransac_evaluation_tolerance * 4.0:
                break

        if len(best_P_inlier) < constants.parameters.min_matches_for_ransac:
            return Coordinate(0.0, 0.0, 0.0), 0.0, [], 0.0
        
        # Re-compute optimal Kabsch transformation matrix parameters using the only the inliers from the best solution
        R, t, theta = kabsch(np.array(best_P_inlier), np.array(best_Q_inlier), constants.parameters.max_rotation_angle_deg)
            
        best_theta = theta
        best_t = Coordinate(float(t[0]), float(t[1]), 0.0)

        return best_t, best_theta, best_draw_Q_inlier, best_inlier_count
    
    def listener_depth_callback(self, msg: Image) -> None:
        """!
        @brief Subscribes to raw depth frames and registers active depth timestamps.

        @param msg Incoming raw depth image.
        """
        self.frame_depth = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
        self.frame_depth_stamp = msg.header.stamp

    def _stamp_to_sec(self, stamp: builtin_interfaces.msg.Time) -> float:
        """!
        @brief Converts standard ROS timeline stamps to floating point seconds.

        @param stamp ROS timeline stamp.
        @return floating point seconds.
        """
        return stamp.sec + stamp.nanosec * 1e-9

    def img_to_kp_des_filtered(self, msg: Image) -> Tuple[List[cv2.KeyPoint], np.ndarray, np.ndarray, Time]:
        """!
        @brief Detects ORB features and filters out close feature duplicates.

        Ensure that features have valid depth observations within range thresholds.

        @param msg Incoming color image message.
        @return A tuple of (valid keypoints list, descriptors, depth values, timestamp).
        """
        # convert RGB message and feature detection
        self.frame_rgb = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        kp = self.orb.detect(self.frame_rgb, None)
        rgb_stamp = msg.header.stamp
        
        # Skip, if values are not valid 
        if self.frame_depth is None or self.frame_depth_stamp is None:
            return None, None, None, None  
        
        if abs(self._stamp_to_sec(rgb_stamp) - self._stamp_to_sec(self.frame_depth_stamp)) > constants.parameters.rgb_depth_sync_tolerance_sec:
            self.get_logger().debug("RGB/Depth timestamps are not synchronized - Skipping Frame.")
            return None, None, None, None

        h, w = self.frame_depth.shape[:2]
        occupied_pixels = {} 
        valid_kp = []
        valid_kp_depth = []
        valid_des = []

        # Save all feature attributes in arrays and filter for valid keypoints only        
        for p in kp:
            u, v = p.pt
            u = int(round(u))
            v = int(round(v))

            if not (0 <= u < w and 0 <= v < h):
                continue

            depth_value = self.frame_depth[v, u]
            if MIN_DEPTH < depth_value < constants.parameters.max_depth:
                is_too_close = False
                
                # Perform fast pixel occupancy neighborhood queries
                for neighbor_u in range(u - constants.parameters.pixel_tolerance, u + constants.parameters.pixel_tolerance + 1):
                    for neighbor_v in range(v - constants.parameters.pixel_tolerance, v + constants.parameters.pixel_tolerance + 1):
                        if (neighbor_u, neighbor_v) in occupied_pixels:
                            is_too_close = True
                            break
                    if is_too_close:
                        break
                
                if not is_too_close:
                    valid_kp.append(p)
                    valid_kp_depth.append(depth_value)
                    occupied_pixels[(u, v)] = True

        # Calculate descriptors only for valid keypoints
        valid_kp, valid_des = self.orb.compute(self.frame_rgb, valid_kp)

        return valid_kp, valid_des, np.array(valid_kp_depth), rgb_stamp
 
    def iteration(self, valid_kp: List[cv2.KeyPoint], valid_des: np.ndarray, valid_kp_depth: np.ndarray, frame_rgb: NDArray, frame_depth: NDArray, rgb_stamp: Time, ransac_result: Tuple[Coordinate, float, List[cv2.KeyPoint], int]) -> None:
        """!
        @brief Runs a single localization iteration over all particles.

        Calculates normalized weights and selects the best candidate particle estimate.

        @param valid_kp Set of visual features.
        @param valid_des Binary visual descriptors.
        @param valid_kp_depth Feature depth values.
        @param frame_rgb Color reference frame image.
        @param frame_depth Depth reference frame image.
        @param rgb_stamp Timeline synchronization timestamp.
        @param ransac_result Estimated relative rigid transformation data.
        """
        log_weights = []
        for robot in self.robots:
            robot.robot_iteration(valid_kp, valid_des, valid_kp_depth, frame_rgb, frame_depth, rgb_stamp, ransac_result)
            log_weights.append(robot.get_log_weight())

        log_weights = np.array(log_weights)
        log_weights -= np.max(log_weights)
        weights = np.exp(log_weights)

        sum_w = np.sum(weights)
        normalized_weights = weights / sum_w if sum_w > 0.0 else np.ones_like(weights) / len(weights)
    
        self.best_robot_idx = np.argmax(normalized_weights)
        self.best_weight = normalized_weights[self.best_robot_idx]
        self.best_robot = self.robots[self.best_robot_idx]

    def parameter_initialization(self) -> Parameters:
        """!
        @brief Declares and loads configuration parameters from the ROS 2 parameters server.

        @return Initialized Parameters data package.
        """
        parameters = Parameters(
            max_depth = int(self.declare_parameter('max_depth', MAX_DEPTH).value),
            ransac_evaluation_tolerance = float(self.declare_parameter('ransac.evaluation_tolerance', RANSAC_EVALUATION_TOLERANCE).value),
            ransac_iteration = int(self.declare_parameter('ransac.iterations', RANSAC_ITERATION).value),
            ransac_sample_size = int(self.declare_parameter('ransac.sample_size', RANSAC_SAMPLE_SIZE).value),
            rgb_depth_sync_tolerance_sec = float(self.declare_parameter('rgb_depth_sync_tolerance_sec', RGB_DEPTH_SYNC_TOLERANCE_SEC).value),
            pixel_tolerance = int(self.declare_parameter('pixel_tolerance', PIXEL_TOLERANCE).value),
            matches_for_new_landmarks = int(self.declare_parameter('matches_for_new_landmarks', MATCHES_FOR_NEW_LANDMARKS).value),
            min_matches_for_ransac = int(self.declare_parameter('ransac.min_matches_for_ransac', MIN_MATCHES_FOR_RANSAC).value),
            max_rotation_angle_deg = float(self.declare_parameter('ransac.max_rotation_angle_deg', MAX_ROTATION_ANGLE_DEG).value),
            min_landmark_trust = float(self.declare_parameter('min_landmark_trust', MIN_LANDMARK_TRUST).value),
            ransac_min_inlier_ratio = float(self.declare_parameter('ransac.min_inlier_ratio', RANSAC_MIN_INLIER_RATIO).value),
            n_robot_samples = int(self.declare_parameter('n_robot_samples', N_ROBOT_SAMPLES).value),
            topic_visual_odometry_msg = self.declare_parameter('topics.visual_odometry_msg', VISUAL_ODOM_MSG_TOPIC).value
        )
        return parameters

    def parameter_callback(self, params: List[rclpy.parameter.Parameter]) -> SetParametersResult:
        """!
        @brief Dynamic parameter callback handling updates during node executions.

        @param params Updated parameters list.
        @return ROS 2 SetParametersResult status indicator.
        """
        mapping = {
            'max_depth': ('max_depth', int),
            'ransac.evaluation_tolerance': ('ransac_evaluation_tolerance', float),
            'ransac.iterations': ('ransac_iteration', int),
            'ransac.sample_size': ('ransac_sample_size', int),
            'rgb_depth_sync_tolerance_sec': ('rgb_depth_sync_tolerance_sec', float),
            'pixel_tolerance': ('pixel_tolerance', int),
            'matches_for_new_landmarks': ('matches_for_new_landmarks', int),
            'ransac.min_matches_for_ransac': ('min_matches_for_ransac', int),
            'ransac.max_rotation_angle_deg': ('max_rotation_angle_deg', float),
            'min_landmark_trust': ('min_landmark_trust', float),
            'ransac.min_inlier_ratio': ('ransac_min_inlier_ratio', float),
            'n_robot_samples': ('n_robot_samples', int),
            'topics.visual_odometry_msg': ('topic_visual_odometry_msg', str),
        }

        for param in params:
            if param.name in mapping:
                attr, cast = mapping[param.name]
                setattr(constants.parameters, attr, cast(param.value))
                self.get_logger().info(f"{param.name} -> {param.value}")

        return SetParametersResult(successful=True)


def create_particle_path_markers(self, particles:List[VisualRobotSample], best_particle_idx:int):
    """!
    @brief Generates visualization markers for particle trajectories.

    @param particles Active particle set.
    @param best_particle_idx Best particle index.
    @return MarkerArray for RViz visualization.
    """
    marker_array = MarkerArray()

    for particle in particles:
        marker = Marker()

        marker.header.frame_id = "odom"
        marker.header.stamp = self.get_clock().now().to_msg()

        marker.ns = "particle_paths"
        marker.id = particle.index

        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        # Width of markers
        marker.scale.x = 0.002

        # Best particle red and wide
        if particle.index == best_particle_idx:
            marker.color.r = 1.0    
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker.scale.x = 0.005
            particle.path_color.a += 0.05
        else:
            marker.color.r = particle.path_color.r     
            marker.color.g = particle.path_color.g
            marker.color.b = particle.path_color.b
            particle.path_color.a *= 0.99
            marker.color.a = particle.path_color.a


        # Particle points
        for pose in particle.path:
            p = Point()
            p.x = pose.x
            p.y = pose.y
            p.z = 0.0
            marker.points.append(p)

        marker_array.markers.append(marker)

    return marker_array



def main():
    """!
    @brief Core node runtime initialization routine.
    """
    rclpy.init()
    node = VisualOdom()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()