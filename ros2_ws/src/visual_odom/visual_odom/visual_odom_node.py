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


from nav_msgs.msg import Odometry

class VisualOdom(Node):
    def __init__(self):
        super().__init__('visual_odom')
        
        # Load parameters from config file with explicit type casting
        self.min_depth: int = int(self.declare_parameter('min_depth', MIN_DEPTH).value)  # type: ignore
        self.max_depth: int = int(self.declare_parameter('max_depth', MAX_DEPTH).value)  # type: ignore
        self.ransac_evaluation_tolerance: float = float(self.declare_parameter('ransac.evaluation_tolerance', RANSAC_EVALUATION_TOLERANCE).value)  # type: ignore
        self.ransac_iteration: int = int(self.declare_parameter('ransac.iterations', RANSAC_ITERATION).value)  # type: ignore
        self.ransac_sample_size: int = int(self.declare_parameter('ransac.sample_size', RANSAC_SAMPLE_SIZE).value)  # type: ignore
        self.rgb_depth_sync_tolerance_sec: float = float(self.declare_parameter('rgb_depth_sync_tolerance_sec', RGB_DEPTH_SYNC_TOLERANCE_SEC).value)  # type: ignore
        self.pixel_tolerance: int = int(self.declare_parameter('pixel_tolerance', PIXEL_TOLERANCE).value)  # type: ignore
        self.matches_for_new_landmarks: int = int(self.declare_parameter('matches_for_new_landmarks', MATCHES_FOR_NEW_LANDMARKS).value)  # type: ignore
        self.min_matches_for_ransac: int = int(self.declare_parameter('ransac.min_matches_for_ransac', MIN_MATCHES_FOR_RANSAC).value) #type: ignore
        self.max_rotation_angle_deg: float = float(self.declare_parameter('ransac.max_rotation_angle_deg', MAX_ROTATION_ANGLE_DEG).value) #type: ignore
        self.max_landmark_age: float = float(self.declare_parameter('max_landmark_age', MAX_LANDMARK_AGE).value)  # type: ignore
        self.ransac_min_inlier_ratio: float = float(self.declare_parameter('ransac.min_inlier_ratio', RANSAC_MIN_INLIER_RATIO).value)  # type: ignore
        self.get_logger().info(f"pixel_tolerance: {self.pixel_tolerance}")

        self.topic_visual_odometry_msg: str = self.declare_parameter('topics.visual_odometry_msg', VISUAL_ODOM_MSG_TOPIC).value  # type: ignore

        self.bridge = CvBridge()
        # Initiate ORB detector
        self.orb = cv2.ORB_create()
        #create BFMatcher object
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self.subscription_rgb = self.create_subscription(Image,'/serf01/nav_rgbd_1/rgb/image_raw', self.listener_rgb_callback, 10)
        self.subscription_depth = self.create_subscription(Image,'/serf01/nav_rgbd_1/depth/image_raw', self.listener_depth_callback, 10)

        self.publisher_keypoints_3d = self.create_publisher(PointCloud2, KEYPOINT_POINTCLOUD_FRAME_ID, 10)
        self.publisher_3d = self.create_publisher(PointCloud2, POINTCLOUD_FRAME_ID, 10)
        self.publisher_visual_odometry_msg = self.create_publisher(Odometry, self.topic_visual_odometry_msg, 10)


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
        self.visual_odom_map = VisualOdomMap()


		# self.P = self.Q:
        sigma_x0   = 0.1   # 10cm Anfangsunsicherheit in x
        sigma_y0   = 0.1   # 10cm in y
        sigma_th0  = 0.05  # ~3° in theta
        self.covariance_P = np.diag([sigma_x0**2, sigma_y0**2, sigma_th0**2])
        self.extended_kalman_filter = ExtendedKalmanFilterRobot(State(self.pos_baselink.x, self.pos_baselink.y, self.theta), self.covariance_P)

        self.get_logger().info("Visual Odometry Node gestartet und bereit für die Verarbeitung von RGB-D Daten.")

        

    def listener_rgb_callback(self,msg):
        self.frame_rgb=self.bridge.imgmsg_to_cv2(msg,'bgr8')
        self.kp = self.orb.detect(self.frame_rgb, None)

        if self.frame_depth is None or self.frame_depth_stamp is None:
            return  

        rgb_stamp = msg.header.stamp
        if abs(self._stamp_to_sec(rgb_stamp) - self._stamp_to_sec(self.frame_depth_stamp)) > self.rgb_depth_sync_tolerance_sec:
            self.get_logger().debug("RGB/Depth nicht ausreichend synchron - Frame wird uebersprungen.")
            return

        # Evaluation of the detected keypoints: Only keypoints with valid depth values are kept for further processing
        valid_kp = []
        valid_des = []
        valid_kp_depth = []
        h, w = self.frame_depth.shape[:2]
        calculated_keypoint_coordinates = []
        # 'pixel' Dictionary nutzen wir als super-schnellen Hash-Map Speicher
        occupied_pixels = {} 
        all_kp = []
        valid_kp = []
        valid_kp_depth = []

        for p in self.kp:
            u, v = p.pt
            u = int(round(u))
            v = int(round(v))

            # Bildgrenzen prüfen
            if not (0 <= u < w and 0 <= v < h):
                continue

            depth_value = self.frame_depth[v, u]

            # Tiefenwert prüfen
            if self.min_depth < depth_value < self.max_depth:
                all_kp.append(p)
                
                is_too_close = False
                
                # Super-schneller Lookup im Dictionary
                for neighbor_u in range(u - self.pixel_tolerance, u + self.pixel_tolerance + 1):
                    for neighbor_v in range(v - self.pixel_tolerance, v + self.pixel_tolerance + 1):
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

        #self.get_logger().info(f"Anzahl gültiger Keypoints mit Tiefeninformation: {len(valid_kp)}; Anzahl aller Keypoints: {len(all_kp)}")
        
        if self.first_iteration or len(self.visual_odom_map) < 50:
            odom_to_base_footprint = calculate_tf(self.pos_baselink, self.theta, rgb_stamp,  VISUAL_ODOM_FRAME_ID, BASE_LINK_FRAME_ID)
            self.tf_broadcaster.sendTransform(odom_to_base_footprint)

            self.first_iteration = False
            self.visual_odom_map.add_landmarks_from_kps(self.covariance_P, valid_kp, valid_des, self.frame_rgb, valid_kp_depth, self.theta, self.pos_baselink)
            #self.get_logger().info(f"Erste Iteration: {len(valid_kp)} Landmarken zur Karte hinzugefügt.")
            return
        else:            
            pos_camera = kinect_depth_to_odom(Coordinate(0.0, 0.0, 0.0), self.theta,self.pos_baselink)

            visible_landmarks = self.visual_odom_map.get_visible_landmarks(pos_camera, self.theta)
            #self.get_logger().info(f"Anzahl sichtbarer Landmarken: {len(visible_landmarks)}")
           
            # Match descriptors.
            matches = self.bf.match(visible_landmarks.get_descriptors(), valid_des)

            self.get_logger().info(f"Anzahl der Matches: {len(matches)}")

            if len(matches) < self.min_matches_for_ransac:
                self.get_logger().warn(f"Zu wenige Matches fuer RANSAC: {len(matches)}")
                self.visual_odom_map.add_landmarks_from_kps(self.covariance_P, valid_kp, valid_des, self.frame_rgb, valid_kp_depth, self.theta, self.pos_baselink)
                return
            
            ransac_result = self.ransac(self.ransac_evaluation_tolerance, self.ransac_iteration, self.ransac_sample_size, matches, visible_landmarks.get_odom_coordinates(), valid_kp, valid_kp_depth)
            ransac_delta_p = ransac_result[0]
            ransac_delta_theta = ransac_result[1]
            ransac_landmark_indices = ransac_result[2]
            ransac_draw_keypoints = ransac_result[3]
            ransac_kp_indices = ransac_result[4]
            ransac_not_matched_kp_indices = ransac_result[5]
            ransac_inlier_count = ransac_result[6]

            if float(ransac_inlier_count)/len(matches) < self.ransac_min_inlier_ratio:  # Weniger als 10% Inlier nach RANSAC
                self.get_logger().warn(f"RANSAC-Ergebnis hat zu wenige Inlier: {ransac_inlier_count} von {len(matches)} Matches. Hinzufügen neuer Landmarken basierend auf den aktuellen Keypoints.")
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

            kalman_iteration_result = self.extended_kalman_filter.kalman_iteration(ransac_delta_p, ransac_delta_theta, z_dict, visible_landmarks)
            self.pos_baselink = Coordinate(kalman_iteration_result[0].x, kalman_iteration_result[0].y, 0.0)
            self.theta = kalman_iteration_result[0].theta
            self.covariance_P = kalman_iteration_result[1]
            

            visible_landmarks.kalman_iteration(self.pos_baselink, self.theta, ransac_landmark_indices, kp_pos)


            odom_to_base_footprint = calculate_tf(self.pos_baselink, self.theta, rgb_stamp, VISUAL_ODOM_FRAME_ID, BASE_LINK_FRAME_ID)
            self.tf_broadcaster.sendTransform(odom_to_base_footprint)

            

            self.publish_odometry_msg(self.publisher_visual_odometry_msg, self.pos_baselink, self.theta, self.covariance_P, rgb_stamp, VISUAL_ODOM_FRAME_ID, BASE_LINK_FRAME_ID)

            #self.get_logger().info(f"RANSAC Result: Rotation um z Achse in ° {ransac_result[1]*180.0/pi}")
            #self.get_logger().info(f"RANSAC Result: Translation  in m {ransac_result[0].x}, {ransac_result[0].y}")
            #self.get_logger().info(f"RANSAC Result: Theta in ° {self.theta*180/pi}")
            
          
            #self.publish_pixels(self.frame_depth, self.frame_rgb, self.publisher_3d, rgb_stamp, KINECT_FRAME_ID)
            #visible_landmarks.publish_pointcloud_map(self.publisher_keypoints_3d, rgb_stamp)
            self.visual_odom_map.publish_pointcloud_map(self.publisher_keypoints_3d, rgb_stamp)

            #self.visual_odom_map.age_and_cleanup_old_landmarks(visible_landmarks,ransac_landmark_indices,self.max_landmark_age,)
            #self.get_logger().info(f"Anzahl der Matches: {len(matches)}")

            if len(matches) < self.matches_for_new_landmarks:
                not_matched_kp = []
                not_matched_des = []
                not_matched_kp_depth = []
                for idx in ransac_not_matched_kp_indices:
                    not_matched_kp.append(valid_kp[idx])
                    not_matched_des.append(valid_des[idx])
                    not_matched_kp_depth.append(valid_kp_depth[idx])

                
                #self.visual_odom_map.add_landmarks_from_kps(self.covariance_P, valid_kp, valid_des, self.frame_rgb, valid_kp_depth, self.theta, self.pos_baselink)
                self.visual_odom_map.add_landmarks_from_kps(self.covariance_P, not_matched_kp, not_matched_des, self.frame_rgb, not_matched_kp_depth, self.theta, self.pos_baselink)
                #self.get_logger().info(f"Zu wenige Matches ({len(matches)}) - Hinzufügen neuer Landmarken basierend auf den aktuellen Keypoints.")

           

            frame_rgb_drawn = cv2.drawKeypoints(self.frame_rgb, ransac_draw_keypoints, None, color=(0,255,0), flags=0)
            cv2.imshow("second RGB Image", frame_rgb_drawn)
            cv2.waitKey(1)
                

    def listener_depth_callback(self,msg):
        self.frame_depth=self.bridge.imgmsg_to_cv2(msg,'passthrough')
        self.frame_depth_stamp = msg.header.stamp

         

    def _stamp_to_sec(self, stamp) -> float:
        return stamp.sec + stamp.nanosec * 1e-9


    def ransac(self, tolerance: float, iteration: int, n_samples: int, matches: List[cv2.DMatch], landmarks_odom_pos: List[Coordinate], valid_kp: List[cv2.KeyPoint], valid_kp_depth: np.ndarray) -> Tuple[Coordinate, float, List[int], List[int], List[int], List[int], int]:
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
            self.get_logger().warn(f"Nicht genug Punkte für RANSAC: {len(P)} < {n_samples}")
            return Coordinate(0.0, 0.0, 0.0), 0.0, [], [], [], [], 0

        P_array = np.array(P)
        Q_array = np.array(Q)


        # NEU: Diese Listen speichern wir NUR für das allerbeste Modell
        best_P_inlier = []
        best_Q_inlier = []
        best_valid_kp_index = []
        best_draw_Q_inlier = []
        best_landmark_index = []
        mean_last_e = 10000.0

        iteration = int(self.ransac_iteration)

        for iter in range(iteration):
            P_samples = []
            Q_samples = []

            samples = random.sample(range(len(P)), n_samples)
            P_samples = [P[i] for i in samples]
            Q_samples = [Q[i] for i in samples]

            R, t, theta  = kabsch(np.array(P_samples), np.array(Q_samples), self.max_rotation_angle_deg)
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
        R, t, theta  = kabsch(np.array(best_P_inlier), np.array(best_Q_inlier), self.max_rotation_angle_deg)
        if R is None:
            self.get_logger().warn("Kabsch returned None after RANSAC, likely due to large rotation.")
            return Coordinate(0.0, 0.0, 0.0), 0.0, [], [], [], [], 0
            
        best_t = Coordinate(float(t[0]), float(t[1]), 0.0)
        best_theta = theta

        valid_set = set(best_valid_kp_index)
        not_matched_kp = [i for i in range(len(valid_kp)) if i not in valid_set]

        self.get_logger().info(f"RANSAC abgeschlossen. Beste Lösung hatte {best_inlier_count} Inlier.")
        
        # Gebe nun exakt die synchronisierten "best_"-Listen zurück
        return best_t, best_theta, best_landmark_index, best_draw_Q_inlier, best_valid_kp_index, not_matched_kp, best_inlier_count

    def publish_pixels(self, frame_depth: NDArray, frame_rgb: NDArray, publisher, time: Time, frame_id: str):
        calculated_point_coordinates = []

        divisor = 15
        for pix_u in range(frame_depth.shape[1]//divisor):
            for pix_v in range(frame_depth.shape[0]//divisor):
                depth_value = frame_depth[pix_v*divisor, pix_u*divisor]
                if depth_value > MIN_DEPTH and depth_value < MAX_DEPTH:
                    pos = pixel_to_kinect(PixelCoordinate(pix_u*divisor, pix_v*divisor, depth_value))
                    b, g, r = frame_rgb[pix_v*divisor, pix_u*divisor]
                    rgb = (int(r) << 16) | (int(g) << 8) | int(b)

                    calculated_point_coordinates.append((pos.x, pos.y, pos.z, rgb))


        self.publish_pointcloud(calculated_point_coordinates, publisher=self.publisher_3d, time=time, frame_id=frame_id )
    
    def publish_pointcloud(self, points_with_rgb, publisher, time: Time, frame_id: str):
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
            points=points_with_rgb
        )

        

        publisher.publish(msg)
        self.get_logger().info(f"PointCloud mit {len(points_with_rgb)} Punkten gesendet!")

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

    
def normalize_angle(angle: float) -> float:
    if angle > pi:
        angle -= 2*pi
    elif angle < -pi:
        angle += 2*pi

    return angle    
    
def main():
    rclpy.init()
    node=VisualOdom()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
