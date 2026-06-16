#!/usr/bin/env python3
import builtin_interfaces
from sensor_msgs import msg
from tf2_ros import Buffer, Duration, TransformBroadcaster, TransformListener
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation
from numpy.typing import NDArray
from rclpy.time import Time
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as R

from visualization_msgs.msg import Marker

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
import visual_odom.constants as constants

from std_msgs.msg import Header

import cProfile
import pstats
import io
import shutil


from nav_msgs.msg import Odometry

from rcl_interfaces.msg import SetParametersResult


class VisualOdom(Node):
    def __init__(self):
        super().__init__('visual_odom')

        constants.parameters = self.parameter_initialization()
        self.add_on_set_parameters_callback(self.parameter_callback)
        
        self.bridge = CvBridge()
        self.orb = cv2.ORB_create(
            nfeatures=1500,        # maximale Anzahl an zu detektierenden Keypoints
            edgeThreshold=30,      # Mindestabstand eines Keypoints vom Bildrand
            patchSize=30,          # Größe des Bereichs zur Descriptor-Berechnung
            fastThreshold=5,      # Schwellwert für FAST-Feature-Erkennung (Empfindlichkeit)
            scoreType=cv2.ORB_HARRIS_SCORE  # Methode zur Bewertung der Keypoint-Qualität
        )
        #create BFMatcher object
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self.subscription_rgb = self.create_subscription(Image, RGB_IMAGE_TOPIC, self.listener_rgb_callback, 10)
        self.subscription_depth = self.create_subscription(Image, DEPTH_IMAGE_TOPIC, self.listener_depth_callback, 10)
        self.subscription_wheel_odom = self.create_subscription(Odometry, WHEEL_ODOMETRY_TOPIC, self.listener_wheel_odom_callback, 10)


        self.publisher_keypoints_3d = self.create_publisher(PointCloud2, KEYPOINT_POINTCLOUD_FRAME_ID, 10)
        self.publisher_3d = self.create_publisher(PointCloud2, POINTCLOUD_FRAME_ID, 10)
        self.publisher_visual_odometry_msg = self.create_publisher(Odometry, constants.parameters.topic_visual_odometry_msg, 10)
        self.publisher_cone = self.create_publisher(Marker, VISION_CONE_TOPIC, 10)
        self.publisher_image = self.create_publisher(Image, KP_IMAGE_TOPIC, 10)

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
        self.position_initialized = False
       

		# self.P = self.Q:
        sigma_x0   = 0.1   # 10cm Anfangsunsicherheit in x
        sigma_y0   = 0.1   # 10cm in y
        sigma_th0  = 0.05  # ~3° in theta

        self.covariance_P =  np.diag([sigma_x0**2, sigma_y0**2, sigma_th0**2])

        self.robots: List[VisualRobotSample] = []
        self.best_robot_idx = 0
        self.best_weight = 0.0

        self.valid_kp_last = None
        self.valid_des_last = None
        self.valid_kp_depth_last = None
        self.valid_kp = None
        self.valid_des = None

        self.get_logger().info("Visual Odometry Node gestartet und bereit für die Verarbeitung von RGB-D Daten.")
        self.frame_stamp = builtin_interfaces.msg.Time()

    def listener_wheel_odom_callback(self, msg: Odometry):
        rot = Rotation.from_quat([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
        state_wheel_odom = State(msg.pose.pose.position.x, msg.pose.pose.position.y, rot.as_euler('xyz')[2])

        if self.position_initialized == False:
            self.position_initialized = True
            self.pos_baselink = Coordinate(state_wheel_odom.x, state_wheel_odom.y, 0.0)
            self.theta = state_wheel_odom.theta         
            return
        else:
            if self.first_iteration == False:
                for robot in self.robots:
                    robot.predict_with_wheel_odom(state_wheel_odom)

        

    def listener_rgb_callback(self,msg):
        

        if self.position_initialized == False:
            return
        
        if self._stamp_to_sec(msg.header.stamp) - self._stamp_to_sec(self.frame_stamp) > 0.000:
            self.frame_stamp = msg.header.stamp
        else:
            return
        
        self.valid_kp, self.valid_des, self.valid_kp_depth, rgb_stamp = self.img_to_kp_des_filtered(msg)
        if self.valid_kp is None or self.valid_des is None or self.valid_kp_depth is None:
            return 

        if self.valid_kp_last is None and self.valid_des_last is None:
            self.valid_kp_last = self.valid_kp
            self.valid_des_last = self.valid_des
            self.valid_kp_depth_last = self.valid_kp_depth
            return

        if self.first_iteration:
            self.first_iteration = False
            odom_to_base_footprint = calculate_tf(self.pos_baselink, self.theta, rgb_stamp,  VISUAL_ODOM_FRAME_ID, BASE_LINK_FRAME_ID)
            self.tf_broadcaster.sendTransform(odom_to_base_footprint)

            for _ in range(constants.parameters.n_robot_samples):
                self.robots.append(VisualRobotSample(self.pos_baselink, self.theta, self.covariance_P, self.bf, self.publisher_visual_odometry_msg, self.publisher_keypoints_3d,self.publisher_cone,self.valid_kp, self.valid_des, self.valid_kp_depth, self.frame_rgb))
        else:
            #rclpy.logging.get_logger("VisualOdom").info(f"Starting RANSAC with {len(self.bf.match(self.valid_des, self.valid_des_last))} matches between current and last frame.")
            matches = self.bf.match(self.valid_des_last,self.valid_des)
            ransac_result = self.ransac(matches, self.valid_kp, self.valid_kp_depth, self.valid_kp_last, self.valid_kp_depth_last)

            self.iteration(self.valid_kp, self.valid_des, self.valid_kp_depth, self.frame_rgb, self.frame_depth, rgb_stamp, ransac_result)  
            
            self.pos_baselink = self.best_robot.get_position()
            self.theta = self.best_robot.get_theta()
            self.covariance_P = self.best_robot.get_covariance_P()
            
            odom_to_base_footprint = calculate_tf(self.pos_baselink, self.theta, rgb_stamp, VISUAL_ODOM_FRAME_ID, BASE_LINK_FRAME_ID)
            self.tf_broadcaster.sendTransform(odom_to_base_footprint)


            self.best_robot.publish_yourself(rgb_stamp)           
        
            frame_rgb_drawn = cv2.drawKeypoints(self.frame_rgb, self.best_robot.get_drawn_keypoints(), None, color=(0,255,0), flags=0)
            
            msg = self.bridge.cv2_to_imgmsg(frame_rgb_drawn, encoding='bgr8')
            self.publisher_image.publish(msg)

            #cv2.imshow("second RGB Image", frame_rgb_drawn)
            #cv2.waitKey(1)
            self.valid_kp_last = self.valid_kp
            self.valid_des_last = self.valid_des
            self.valid_kp_depth_last = self.valid_kp_depth

    def ransac(self, matches: List[cv2.DMatch],  valid_kp: List[cv2.KeyPoint], valid_kp_depth: np.ndarray , valid_kp_last: List[cv2.KeyPoint], valid_kp_depth_last: np.ndarray) -> Tuple[Coordinate, float, List[int], List[int], List[int], List[int], int]:
        #Abbruchbedingung
        #t0 = time.perf_counter()
        P = []
        Q = []

        best_inlier_count = 0
        inlier_count = 0
        best_t = Coordinate(0.0, 0.0, 0.0)
        best_theta = 0.0

        landmark_index = [] 


        for m in range(len(matches)): 
            #t1 = time.perf_counter()           
            #Matrix mit Koordinaten im kinect frame
            keypoint = valid_kp[matches[m].trainIdx].pt
            keypoint_last = valid_kp_last[matches[m].queryIdx].pt
            #t2 = time.perf_counter()
            #time1 = 1000000*(t2-t1)

            
            u, v = keypoint_last
            u = round(u)
            v = round(v)
            z = valid_kp_depth_last[matches[m].queryIdx]

            coor = pixel_to_kinect_(u, v, z)
            coor = Coordinate(coor[0], coor[1], coor[2])
            coor_base_link = kinect_depth_to_baselink(coor)
           
            P.append([coor_base_link.x, coor_base_link.y])
            #t6 = time.perf_counter()
            #time5 = 1000000*(t6-t5)

            u, v = keypoint
            u = round(u)
            v = round(v)
            z = valid_kp_depth[matches[m].trainIdx]

            coor = pixel_to_kinect_(u, v, z)
            coor = Coordinate(coor[0], coor[1], coor[2])
            #time6 = 1000000*(t7-t6)
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

            R, delta_t, delta_theta  = kabsch(np.array(P_samples), np.array(Q_samples), constants.parameters.max_rotation_angle_deg)
            if R is None:
                #rclpy.logging.get_logger("RANSAC").info(f"RANSAC at iteration {iter} failed to compute transformation")
                continue
   
            e = np.linalg.norm(P_array - ((R @ Q_array.T).T + delta_t.T), axis=1)
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

            if (best_inlier_count/ len(matches)) > constants.parameters.ransac_min_inlier_ratio or mean_e < constants.parameters.ransac_evaluation_tolerance*4:
                #rclpy.logging.get_logger("RANSAC").info(f"RANSAC early break at iteration {iter} with mean error {mean_e} and inlier count {inlier_count} and length matches {len(matches)}")
                break

            #if(iter == iteration-1):
                #rclpy.logging.get_logger("RANSAC").info(f"RANSAC finished all iterations. Best mean error: {mean_last_e} with inlier count {best_inlier_count} out of {len(matches)} matches.")

        if len(best_P_inlier) <  constants.parameters.min_matches_for_ransac:
            #rclpy.logging.get_logger("RANSAC").info(f"RANSAC failed to find a valid transformation with enough inliers. Best inlier count: {best_inlier_count} out of {len(matches)} matches.")
            return Coordinate(0.0, 0.0, 0.0), 0.0, [], 0.0
        
        # Kabsch noch einmal mit den endgültigen besten Inliern berechnen
        R, delta_t, delta_theta  = kabsch(np.array(best_P_inlier), np.array(best_Q_inlier), constants.parameters.max_rotation_angle_deg)
            
        best_theta = delta_theta

        best_t = Coordinate(float(delta_t[0]), float(delta_t[1]), 0.0)

        
        # Gebe nun exakt die synchronisierten "best_"-Listen zurück
        return best_t, best_theta, best_draw_Q_inlier, best_inlier_count
                

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
                if depth_value > MIN_DEPTH and depth_value < constants.parameters.max_depth:
                    pos = pixel_to_kinect(PixelCoordinate(pix_u*divisor, pix_v*divisor, depth_value))
                    b, g, r = frame_rgb[pix_v*divisor, pix_u*divisor]
                    rgb = (int(r) << 16) | (int(g) << 8) | int(b)

                    calculated_point_coordinates.append((pos.x, pos.y, pos.z, rgb))

       
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

        
        if abs(self._stamp_to_sec(rgb_stamp) - self._stamp_to_sec(self.frame_depth_stamp)) > constants.parameters.rgb_depth_sync_tolerance_sec:
            self.get_logger().debug("RGB/Depth nicht ausreichend synchron - Frame wird uebersprungen.")
            return None, None, None, None

        # Evaluation of the detected keypoints: Only keypoints with valid depth values are kept for further processing
        h, w = self.frame_depth.shape[:2]

        occupied_pixels = {} 
        all_kp = []
        valid_kp = []
        all_kp_depth = []
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
            if MIN_DEPTH < depth_value < constants.parameters.max_depth:
                all_kp.append(p)
                all_kp_depth.append(depth_value)

                is_too_close = False
                
                # Super-schneller Lookup im Dictionary
                for neighbor_u in range(u - constants.parameters.pixel_tolerance, u + constants.parameters.pixel_tolerance + 1):
                    for neighbor_v in range(v - constants.parameters.pixel_tolerance, v + constants.parameters.pixel_tolerance + 1):
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
 
    def iteration(self, valid_kp: List[cv2.KeyPoint], valid_des: np.ndarray, valid_kp_depth: np.ndarray, frame_rgb: NDArray, frame_depth: NDArray, rgb_stamp: Time, ransac_result):
        log_weights = []
        for robot in self.robots:
            robot.robot_iteration(valid_kp, valid_des, valid_kp_depth, frame_rgb, frame_depth, rgb_stamp, ransac_result)
            log_weights.append(robot.get_log_weight())

        log_weights = np.array(log_weights)
        log_weights -= np.max(log_weights)
        weights = np.exp(log_weights)

        sum = np.sum(weights)
        
        normalized_weights = weights/sum
    
        self.best_robot_idx = np.argmax(normalized_weights)
        #rclpy.logging.get_logger("VisualOdom").info(f"Best robot index: {self.best_robot_idx} with weight {normalized_weights[self.best_robot_idx]:.4f}")
        self.best_weight = normalized_weights[self.best_robot_idx]
        self.best_robot = self.robots[self.best_robot_idx]


    def parameter_initialization(self) -> Parameters:
        # Load parameters from config file with explicit type casting
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

    def parameter_callback(self, params):
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
                self.get_logger().info(f"{param.name} → {param.value}")

        return SetParametersResult(successful=True)

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
    node = VisualOdom()
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        profiler.disable()
        
        # Als Text ausgeben
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.sort_stats('cumulative')
        stats.print_stats(30)
        print(stream.getvalue())
        
        # Als Datei speichern und nach Windows kopieren
        profiler.dump_stats('/tmp/profile.prof')
        shutil.copy('/tmp/profile.prof', '/mnt/c/Users/lukas/Desktop/profile.prof')
        print("Profiling-Daten gespeichert: /mnt/c/Users/lukas/Desktop/profile.prof")
        
        node.destroy_node()
        rclpy.shutdown()
