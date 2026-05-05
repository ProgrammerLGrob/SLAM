#!/usr/bin/env python3

from visual_odom.visual_odom_map import VisualOdomMap
from visual_odom.constants import *
import tf2_ros 
from tf2_ros import Buffer, TransformBroadcaster, TransformListener
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
from visual_odom.landmark import kabsch, Coordinate
from visual_odom.tf_methods import *
from nav_msgs.msg import Odometry

class VisualOdom(Node):
    def __init__(self):
        super().__init__('visual_odom')
        
        # Load parameters from config file with explicit type casting
        self.descriptor_tolerance: float = float(self.declare_parameter('descriptor_tolerance', DESCRIPTOR_TOLERANCE).value)  # type: ignore
        self.ransac_evaluation_tolerance: float = float(self.declare_parameter('ransac.evaluation_tolerance', RANSAC_EVALUATION_TOLERANCE).value)  # type: ignore
        self.ransac_iteration: int = int(self.declare_parameter('ransac.iterations', RANSAC_ITERATION).value)  # type: ignore
        self.ransac_sample_size: int = int(self.declare_parameter('ransac.sample_size', RANSAC_SAMPLE_SIZE).value)  # type: ignore
        self.rgb_depth_sync_tolerance_sec: float = float(self.declare_parameter('rgb_depth_sync_tolerance_sec', RGB_DEPTH_SYNC_TOLERANCE_SEC).value)  # type: ignore
        self.waiting_frames: int = int(self.declare_parameter('waiting_frames', WAITING_FRAMES).value)  # type: ignore

        self.topic_visual_odometry_msg: str = self.declare_parameter('topics.visual_odometry_msg', VISUAL_ODOM_MSG_TOPIC).value  # type: ignore

        self.bridge = CvBridge()
        # Initiate ORB detector
        self.orb = cv2.ORB_create()

        self.subscription_rgb = self.create_subscription(Image,'/serf01/nav_rgbd_1/rgb/image_raw', self.listener_rgb_callback, 10)
        self.subscription_depth = self.create_subscription(Image,'/serf01/nav_rgbd_1/depth/image_raw', self.listener_depth_callback, 10)

        self.publisher_keypoints_3d = self.create_publisher(PointCloud2, KEYPOINT_POINTCLOUD_FRAME_ID, 10)
        self.publisher_3d = self.create_publisher(PointCloud2, POINTCLOUD_FRAME_ID, 10)
        self.publisher_visual_odometry_msg = self.create_publisher(Odometry, self.topic_visual_odometry_msg, 10)


        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.landmarks = []

        self.visual_map = VisualOdomMap()
        
        self.frame_rgb_first = None
        self.frame_rgb_sec = None
        self.frame_depth_first = None
        self.frame_depth_sec = None
        self.frame_depth = None
        self.frame_depth_stamp = None
        self.frame_rgb = None
        self.counter = 0
        self.theta = 0
        self.pos_base = np.array([0.0, 0.0, 0.0]) #Position in odom frame

        self.get_logger().info(f"Start")

        


    def listener_depth_callback(self,msg):
        self.frame_depth=self.bridge.imgmsg_to_cv2(msg,'passthrough')
        self.frame_depth_stamp = msg.header.stamp
        odom_to_base_tf, self.pos_base = calculate_tf([0.0, 0.0], self.theta, self.frame_depth_stamp, self.pos_base, VISUAL_ODOM_FRAME_ID, BASE_LINK_FRAME_ID)
        self.tf_broadcaster.sendTransform(odom_to_base_tf)
        

    def listener_rgb_callback(self,msg):
        # Convert ROS Image message to OpenCV format
        self.frame_rgb=self.bridge.imgmsg_to_cv2(msg,'bgr8')
        # Detect keypoints in the RGB image using ORB
        self.kp = self.orb.detect(self.frame_rgb, None)


        # Ensure that depth data is available and synchronized with the RGB frame before proceeding
        if self.frame_depth is None or self.frame_depth_stamp is None:
            return  

        rgb_stamp = msg.header.stamp
        if abs(self._stamp_to_sec(rgb_stamp) - self._stamp_to_sec(self.frame_depth_stamp)) > self.rgb_depth_sync_tolerance_sec:
            self.get_logger().debug("RGB/Depth not synchronized - Frame skipped.")
            return
        

        # Evaluation of the detected keypoints: Only keypoints with valid depth values are kept for further processing
        valid_kp = []
        valid_kp_depth = []
        h, w = self.frame_depth.shape[:2]
        calculated_keypoint_coordinates = []

        for p in self.kp:
            u, v = p.pt
            u = int(round(u))
            v = int(round(v))

            if not (0 <= u < w and 0 <= v < h):
                continue

            depth_value = self.frame_depth[v, u]

            if depth_value > MIN_DEPTH and depth_value < MAX_DEPTH:
                valid_kp.append(p) 
                valid_kp_depth.append(depth_value)

                rgb = (int(0) << 16) | (int(255) << 8) | int(0)

                x, y, z = self.calculate_coordinate(u, v, depth_value)/1000.0
                calculated_keypoint_coordinates.append((x, y, z, rgb))

        if len(valid_kp) < self.ransac_sample_size: # type: ignore
            self.get_logger().warn(f"Too few valid keypoints for RANSAC: {len(valid_kp)} < {self.ransac_sample_size}")
            return
        
        self.valid_kp, self.des = self.orb.compute(self.frame_rgb, valid_kp)

        self.counter += 1

        if self.counter > self.waiting_frames: # type: ignore
            self.counter = 0
        else:
            return

        
        if(len(self.visual_map) == 0):
            self.get_logger().info(f"Visual Map is empty")
            self.visual_map.add_landmarks_from_kps(self.valid_kp, self.des, self.frame_rgb, self.frame_depth, rgb_stamp, self.tf_buffer)
            return
            
        else:
            #create BFMatcher object
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            self.pos_kinect_depth = base_link_to_kinect_depth(self.tf_buffer, self.pos_base, rgb_stamp)
            visible_landmarks = self.visual_map.get_visible_landmarks(self.pos_base, self.theta)

            # Match descriptors.
            matches = bf.match(visible_landmarks.get_descriptors(), self.des)

            if len(matches) < self.ransac_sample_size: # type: ignore
                self.get_logger().warn(f"Too few matches for RANSAC: {len(matches)} < {self.ransac_sample_size}")
                return
        
            ransac_result = self.ransac(self.ransac_evaluation_tolerance, self.ransac_iteration, self.ransac_sample_size, matches, visible_landmarks.get_kps(), self.valid_kp, visible_landmarks.get_depth(), valid_kp_depth, rgb_stamp)
            self.theta += normalize_angle(ransac_result[1])
            self.theta = normalize_angle(self.theta)


            odom_to_base_tf, self.pos_base = calculate_tf(ransac_result[0], self.theta, rgb_stamp, self.pos_base, VISUAL_ODOM_FRAME_ID, BASE_LINK_FRAME_ID)

            self.tf_broadcaster.sendTransform(odom_to_base_tf)
            
            #Publish odometry message for visualization in RViz
            self.publish_odometry_msg(self.publisher_visual_odometry_msg, self.pos_base, self.theta, rgb_stamp, VISUAL_ODOM_FRAME_ID, BASE_LINK_FRAME_ID)

            #Logging for RANSAC result
            self.get_logger().info(f"RANSAC Result: Rotation um z Achse in ° {ransac_result[1]*180/pi}")
            self.get_logger().info(f"RANSAC Result: Translation  in mm {ransac_result[0]}")
            self.get_logger().info(f"RANSAC Result: Theta in ° {self.theta*180/pi}")

            #Add kps as landmarks to map
            inlier_index = ransac_result[2]
            self.visual_map.add_landmarks_from_kps(self.valid_kp[inlier_index], self.des[inlier_index], self.frame_rgb, self.frame_depth, rgb_stamp, self.tf_buffer)

            self.publish_pixels(self.frame_depth, self.frame_rgb, self.publisher_3d, rgb_stamp, KINECT_FRAME_ID)
            self.visual_map.publish_pointcloud_map(self.publisher_keypoints_3d, rgb_stamp)
        
            self.frame_rgb_drawn = cv2.drawKeypoints(self.frame_rgb, self.valid_kp[inlier_index], None, color=(0,255,0), flags=0)
            cv2.imshow("second RGB Image", self.frame_rgb_drawn)
            cv2.waitKey(1)

    def ransac(self, tolerance: float, iteration: int, n_samples: int, matches: List[cv2.DMatch], landmarks: VisualOdomMap, valid_kps: List[cv2.KeyPoint], valid_kp_depth: List[float], timestamp: Time) -> Tuple[NDArray, float, List[cv2.KeyPoint]]:
        P = []
        Q = []

        best_inlier_count = 0
        inlier_count = 0
        best_t = np.zeros((2,1))
        best_theta = 0

        for m in range(len(matches)):            
            coor = odom_to_kinect_depth(landmarks[matches[m].queryIdx].odom_coordinates, timestamp)
            P.append([coor[0], coor[2]])
            
            keypoint_frame = valid_kps[matches[m].trainIdx].pt

            u, v = keypoint_frame
            u = round(u)
            v = round(v)

            z = valid_kp_depth[matches[m].trainIdx]

            coor = self.calculate_coordinate(u, v, z)
            Q.append([coor[0], coor[2]])


        if len(P) < n_samples:
            self.get_logger().warn(f"Nicht genug Punkte für RANSAC: {len(P)} < {n_samples}")
            return best_t, best_theta, []

        P_array = np.array(P)
        Q_array = np.array(Q)

        P_inlier = []
        Q_inlier = []

        

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
                index_kp_Q_inlier: List[int] = []
                best_inlier_count = inlier_count
                for i in range(len(e)):
                    if e[i] < tolerance:
                        index_kp_Q_inlier.append(matches[i].trainIdx)
                        P_inlier.append(P_array[i])
                        Q_inlier.append(Q_array[i])

        R, t, theta  = kabsch(np.array(P_inlier), np.array(Q_inlier))
        best_t = t
        best_theta = theta

        self.get_logger().info(f"RANSAC: Best inlier count: {best_inlier_count} / {len(P)}")
        
        return best_t, best_theta, index_kp_Q_inlier
    
    def publish_pixels(self, frame_depth: NDArray, frame_rgb: NDArray, publisher, time: Time, frame_id: str):
        calculated_point_coordinates = []

        for pix_u in range(frame_depth.shape[1]//2):
            for pix_v in range(frame_depth.shape[0]//2):
                depth_value = frame_depth[pix_v*2, pix_u*2]
                if depth_value > MIN_DEPTH and depth_value < MAX_DEPTH:
                    x,y,z =self.calculate_coordinate(pix_u*2, pix_v*2, depth_value)/1000.0
                    b, g, r = frame_rgb[pix_v*2, pix_u*2]
                    rgb = (int(r) << 16) | (int(g) << 8) | int(b)

                    calculated_point_coordinates.append((x, y, z, rgb))


        self.publish_pointcloud(calculated_point_coordinates, publisher=self.publisher_3d, time=time, frame_id=frame_id )
    
    def calculate_coordinate(self, u: int, v: int, z: float) -> NDArray:
        y = z * (v - CV) / F
        x = z * (u - CU) / F

        p3d_vector = np.array([x, y, z])
        return p3d_vector
    
    def publish_odometry_msg(self, publisher, pos, theta: float, timestamp, parent_frame_id: str, child_frame_id: str):
        msg = Odometry()

        # Header
        msg.header.stamp = timestamp
        msg.header.frame_id = parent_frame_id
        msg.child_frame_id = child_frame_id

        # Pose
        msg.pose.pose.position.x = pos[0]
        msg.pose.pose.position.y = pos[1]
        msg.pose.pose.position.z = pos[2]

        euler = Rotation.from_euler('z', float(theta))
        quat = euler.as_quat(canonical=True)
        msg.pose.pose.orientation.x = quat[0]
        msg.pose.pose.orientation.y = quat[1]
        msg.pose.pose.orientation.z = quat[2]
        msg.pose.pose.orientation.w = quat[3]

        publisher.publish(msg)

    def _stamp_to_sec(self, stamp) -> float:
        return stamp.sec + stamp.nanosec * 1e-9

    
def calculate_tf(trans, theta: float, timestamp, P_old: NDArray, parent_frame_id: str, child_frame_id: str) -> Tuple[TransformStamped, NDArray]:
    
    translation = np.zeros(3)
    translation[0] = trans[1]
    translation[1] = -trans[0]
    translation[2] = 0.0

    t = TransformStamped()
    t.header.stamp = timestamp
    t.header.frame_id = parent_frame_id
    t.child_frame_id = child_frame_id

    R = np.array([[ np.cos(theta), -np.sin(theta), 0.0],
                    [ np.sin(theta),  np.cos(theta), 0.0],
                    [ 0.0, 0.0, 1.0]])

    delta_P_o = (R @ translation) #not sure which one is correct
    #delta_P_o = (R @ translation.T ).T

    P_new = P_old + delta_P_o/1000.0 
    
    t.transform.translation.x = P_new[0]
    t.transform.translation.y = P_new[1]
    t.transform.translation.z = P_new[2]

    euler = Rotation.from_euler('z', float(theta))
    quat = euler.as_quat(canonical=True)
    # Cast quaternion components to Python floats
    t.transform.rotation.x = quat[0]
    t.transform.rotation.y = quat[1]
    t.transform.rotation.z = quat[2]
    t.transform.rotation.w = quat[3]

    return t, P_new

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
        self.get_logger().info(f"PointCloud with {len(points_with_rgb)} points sent!")
 
    
def main():
    rclpy.init()
    node=VisualOdom()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()