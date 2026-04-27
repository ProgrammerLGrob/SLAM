#!/usr/bin/env python3

KINECT_FRAME_ID = "kinect_depth"
BASE_LINK_FRAME_ID = "base_link"
VISUAL_ODOM_FRAME_ID = "odom_visual"


POINTCLOUD_FRAME_ID = "points_3d"
KEYPOINT_POINTCLOUD_FRAME_ID = "keypoint_3d"

MIN_DEPTH = 400
MAX_DEPTH = 5000

DESCRIPTOR_TOLERANCE = 50
RANSAC_EVALUATION_TOLERANCE = 45 #in mm
RANSAC_ITERATION = 1000
RANSAC_SAMPLE_SIZE = 3
RGB_DEPTH_SYNC_TOLERANCE_SEC = 0.05

WAITING_FRAMES = 2

F = 526.61
CU = 318.525
CV = 241.181

import tf2_ros 
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation
from numpy.typing import NDArray
from rclpy.time import Time

from math import pi
import random
from typing import List, Tuple

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from sensor_msgs_py import point_cloud2

from cv_bridge import CvBridge
import cv2
import numpy as np
from visual_odom.landmark import kabsch

class VisualOdom(Node):
    def __init__(self):
        super().__init__('visual_odom')
        
        # Load parameters from config file with explicit type casting
        self.min_depth: int = int(self.declare_parameter('min_depth', MIN_DEPTH).value)  # type: ignore
        self.max_depth: int = int(self.declare_parameter('max_depth', MAX_DEPTH).value)  # type: ignore
        self.descriptor_tolerance: float = float(self.declare_parameter('descriptor_tolerance', DESCRIPTOR_TOLERANCE).value)  # type: ignore
        self.ransac_evaluation_tolerance: float = float(self.declare_parameter('ransac.evaluation_tolerance', RANSAC_EVALUATION_TOLERANCE).value)  # type: ignore
        self.ransac_iteration: int = int(self.declare_parameter('ransac.iterations', RANSAC_ITERATION).value)  # type: ignore
        self.ransac_sample_size: int = int(self.declare_parameter('ransac.sample_size', RANSAC_SAMPLE_SIZE).value)  # type: ignore
        self.rgb_depth_sync_tolerance_sec: float = float(self.declare_parameter('rgb_depth_sync_tolerance_sec', RGB_DEPTH_SYNC_TOLERANCE_SEC).value)  # type: ignore
        self.waiting_frames: int = int(self.declare_parameter('waiting_frames', WAITING_FRAMES).value)  # type: ignore

        self.bridge = CvBridge()
        # Initiate ORB detector
        self.orb = cv2.ORB_create()

        self.subscription_rgb = self.create_subscription(Image,'/serf01/nav_rgbd_1/rgb/image_raw', self.listener_rgb_callback, 10)
        self.subscription_depth = self.create_subscription(Image,'/serf01/nav_rgbd_1/depth/image_raw', self.listener_depth_callback, 10)

        self.publisher_keypoints_3d = self.create_publisher(PointCloud2, KEYPOINT_POINTCLOUD_FRAME_ID, 10)
        self.publisher_3d = self.create_publisher(PointCloud2, POINTCLOUD_FRAME_ID, 10)


        self.tf_broadcaster = TransformBroadcaster(self)

        self.landmarks = []
        self.frame_rgb_first = None
        self.frame_rgb_sec = None
        self.frame_depth_first = None
        self.frame_depth_sec = None
        self.frame_depth = None
        self.frame_depth_stamp = None
        self.frame_rgb = None
        self.counter = 0
        self.theta = 0
        self.P = np.array([0, 0]) #Position in odom frame

        

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
        h, w = self.frame_depth.shape[:2]
        calculated_keypoint_coordinates = []

        for p in self.kp:
            u, v = p.pt
            u = int(round(u))
            v = int(round(v))

            if not (0 <= u < w and 0 <= v < h):
                continue

            depth_value = self.frame_depth[v, u]

            if depth_value > self.min_depth and depth_value < self.max_depth:
                valid_kp.append(p) 
                rgb = (int(0) << 16) | (int(255) << 8) | int(0)

                x, y, z = self.calculate_coordinate(u, v, depth_value)/1000.0
                calculated_keypoint_coordinates.append((x, y, z, rgb))

        if len(valid_kp) < self.ransac_sample_size: # type: ignore
            self.get_logger().warn("Zu wenige valide Keypoints fuer robuste Schaetzung.")
            return

        if self.frame_rgb_first is None:
            self.frame_rgb_first = self.frame_rgb
            self.frame_depth_first = self.frame_depth
            self.kp_first,self.des_first = self.orb.compute(self.frame_rgb, valid_kp)
            self.get_logger().info(f"Start")
            return

        self.counter += 1

        if self.frame_rgb_sec is None and self.counter > self.waiting_frames: # type: ignore
            self.counter = 0
            self.frame_rgb_sec = self.frame_rgb
            self.frame_depth_sec = self.frame_depth
            self.kp_sec,self.des_sec = self.orb.compute(self.frame_rgb, valid_kp)
        else:
            return

        
        #create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # Match descriptors.
        matches = bf.match(self.des_first,self.des_sec)

        if len(matches) < self.ransac_sample_size: # type: ignore
            self.get_logger().warn(f"Zu wenige Matches fuer RANSAC: {len(matches)}")
            return

        ransac_result = self.ransac(self.ransac_evaluation_tolerance, self.ransac_iteration, self.ransac_sample_size, matches, self.kp_first, self.kp_sec, self.frame_depth_first, self.frame_depth_sec)

        self.theta += normalize_angle(ransac_result[1])
        self.theta = normalize_angle(self.theta)

        odom_to_base_footprint, self.P = calculate_tf(ransac_result[0], self.theta, rgb_stamp, self.P, VISUAL_ODOM_FRAME_ID, BASE_LINK_FRAME_ID)

        self.tf_broadcaster.sendTransform(odom_to_base_footprint)

        self.get_logger().info(f"RANSAC Result: Rotation um z Achse in ° {ransac_result[1]*180/pi}")
        self.get_logger().info(f"RANSAC Result: Translation  in mm {ransac_result[0]}")
        self.get_logger().info(f"RANSAC Result: Theta in ° {self.theta*180/pi}")
        
        #Important for iteration
        self.frame_rgb_first = self.frame_rgb_sec
        self.frame_depth_first = self.frame_depth_sec       
        self.kp_first,self.des_first = self.kp_sec,self.des_sec

        kp = ransac_result[2] # type: ignore

        self.publish_pixels(self.frame_depth, self.frame_rgb, self.publisher_3d, rgb_stamp, KINECT_FRAME_ID)
        self.publish_pointcloud(calculated_keypoint_coordinates, self.publisher_keypoints_3d, rgb_stamp, KINECT_FRAME_ID)
        
        self.frame_rgb_sec = cv2.drawKeypoints(self.frame_rgb_sec, kp, None, color=(0,255,0), flags=0)
        cv2.imshow("second RGB Image", self.frame_rgb_sec)
        cv2.waitKey(1)

        self.frame_rgb_sec = None
        self.frame_depth_sec = None
            

    def listener_depth_callback(self,msg):
        self.frame_depth=self.bridge.imgmsg_to_cv2(msg,'passthrough')
        self.frame_depth_stamp = msg.header.stamp

         

    def _stamp_to_sec(self, stamp) -> float:
        return stamp.sec + stamp.nanosec * 1e-9


    def ransac(self, tolerance: float, iteration: int, n_samples: int, matches: List[cv2.DMatch], kp_first: List[cv2.KeyPoint], kp_sec: List[cv2.KeyPoint], frame_depth_first, frame_depth_sec) -> Tuple[NDArray, float, List[cv2.KeyPoint]]:
        P = []
        Q = []

        best_inlier_count = 0
        inlier_count = 0
        best_t = np.zeros((2,1))
        best_theta = 0

        h_first, w_first = frame_depth_first.shape[:2]
        h_sec, w_sec = frame_depth_sec.shape[:2]

        for m in range(len(matches)):            
            #Matrix mit Koordinaten im kinect frame
            keypoint_first = kp_first[matches[m].queryIdx].pt
            keypoint_sec = kp_sec[matches[m].trainIdx].pt
            
            u_first, v_first = keypoint_first
            u_first = round(u_first)
            v_first = round(v_first)

            if not (0 <= u_first < w_first and 0 <= v_first < h_first):
                continue

            z_first = frame_depth_first[v_first, u_first]

            u_sec, v_sec = keypoint_sec
            u_sec = round(u_sec)
            v_sec = round(v_sec)

            if not (0 <= u_sec < w_sec and 0 <= v_sec < h_sec):
                continue

            z_sec = frame_depth_sec[v_sec, u_sec]

            if z_first <= self.min_depth or z_first >= self.max_depth or z_sec <= self.min_depth or z_sec >= self.max_depth:
                continue

            coor = self.calculate_coordinate(u_first, v_first, z_first)
            P.append([coor[0], coor[2]])
            coor = self.calculate_coordinate(u_sec, v_sec, z_sec)
            Q.append([coor[0], coor[2]])


        if len(P) < n_samples:
            self.get_logger().warn(f"Nicht genug Punkte für RANSAC: {len(P)} < {n_samples}")
            return best_t, best_theta, []

        P_array = np.array(P)
        Q_array = np.array(Q)

        P_inlier = []
        Q_inlier = []

        draw_Q_inlier: List[cv2.KeyPoint] = []

        iteration = len(matches)/100 * self.ransac_iteration # type: ignore #
        iteration = int(iteration)

        for iter in range(iteration):
            P_samples = []
            Q_samples = []

            inlier_count = 0 
            samples = random.sample(range(len(P)), n_samples) #Take new random samples for next iteration
            P_samples = [P[i] for i in samples]
            Q_samples = [Q[i] for i in samples]


            R, t, theta  = kabsch(np.array(P_samples), np.array(Q_samples))
   
            e = np.linalg.norm(P_array - ((R @ Q_array.T).T + t.T), axis=1) #calculate error for all points
            inlier_count = np.sum(e < tolerance) #count inliers with error smaller than tolerance

            if inlier_count > best_inlier_count:
                P_inlier = []
                Q_inlier = []
                best_inlier_count = inlier_count
                for i in range(len(e)):
                    if e[i] < tolerance:
                        draw_Q_inlier.append(kp_sec[matches[i].trainIdx])
                        P_inlier.append(P_array[i])
                        Q_inlier.append(Q_array[i])

        R, t, theta  = kabsch(np.array(P_inlier), np.array(Q_inlier))
        best_t = t
        best_theta = theta

        self.get_logger().info(f"RANSAC abgeschlossen. Beste Lösung hatte {best_inlier_count} Inlier von {len(matches)} Punkten.")
        
        return best_t, best_theta, draw_Q_inlier
    
    def publish_pixels(self, frame_depth: NDArray, frame_rgb: NDArray, publisher, time: Time, frame_id: str):
        calculated_point_coordinates = []

        for pix_u in range(frame_depth.shape[1]):
            for pix_v in range(frame_depth.shape[0]):
                depth_value = frame_depth[pix_v, pix_u]
                if depth_value > MIN_DEPTH and depth_value < MAX_DEPTH:
                    x,y,z =self.calculate_coordinate(pix_u, pix_v, depth_value)/1000.0
                    b, g, r = frame_rgb[pix_v, pix_u]
                    rgb = (int(r) << 16) | (int(g) << 8) | int(b)

                    calculated_point_coordinates.append((x, y, z, rgb))


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

    
    def calculate_coordinate(self, u: int, v: int, z: float) -> NDArray:
        y = z * (v - CV) / F
        x = z * (u - CU) / F

        p3d_vector = np.array([x, y, z])
        return p3d_vector
    
def calculate_tf(translation, theta: float, timestamp, P_old: NDArray, parent_frame_id: str, child_frame_id: str) -> Tuple[TransformStamped, NDArray]:
    
    translation = np.array(translation) /1000.0 #Convert from mm to m
    t = TransformStamped()
    t.header.stamp = timestamp
    t.header.frame_id = parent_frame_id
    t.child_frame_id = child_frame_id

    R = np.array([[ np.cos(theta), np.sin(theta)],
                    [ -np.sin(theta),  np.cos(theta)]])

    delta_P_o = R @ translation 

    P_new = P_old + delta_P_o
    
    t.transform.translation.x = P_new[0]
    t.transform.translation.y = P_new[1]
    t.transform.translation.z = 0.0

    euler = Rotation.from_euler('z', float(theta))
    quat = euler.as_quat(canonical=True)
    # Cast quaternion components to Python floats
    t.transform.rotation.x = quat[0]
    t.transform.rotation.y = quat[1]
    t.transform.rotation.z = quat[2]
    t.transform.rotation.w = quat[3]

    return t, P_new

    
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