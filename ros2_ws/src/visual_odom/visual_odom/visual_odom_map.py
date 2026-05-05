from __future__ import annotations

from visual_odom import constants

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional
from .landmark import Landmark

from sensor_msgs.msg import Image, PointCloud2, PointField

import tf2_ros 
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation
from numpy.typing import NDArray
from rclpy.time import Time

from math import pi, atan2, cos, sin
import random
from typing import List, Tuple

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from sensor_msgs_py import point_cloud2

from cv_bridge import CvBridge
import cv2
import numpy as np
from visual_odom.landmark import *
from visual_odom.landmark import Landmark
from nav_msgs.msg import Odometry

from visual_odom.constants import *
from visual_odom.tf_methods import *



DEPTH_ERROR = 10 # in mm


class VisualOdomMap(list):
    def __init__(self,
        landmarks: Optional[Iterable[Landmark]] = None,
    ) -> None:
        super().__init__()

        if landmarks is not None:
            self.extend(landmarks)

    def get_landmark(self, index: int) -> Landmark:
        """
        Get a landmark by index.
        """
        return self[index]
    
    def add_landmark(self, landmark: Landmark) -> None:
        """
        Add a landmark to the map.
        """
        self.append(landmark)
    
    def get_all_landmarks(self) -> list[Landmark]:
        """
        Get all landmarks in the map.
        """
        return list(self)
    
    def get_size(self) -> int:
        """
        Get the number of landmarks in the map.
        """
        return len(self)
    

    def add_landmarks_from_kps(self, kps: Iterable[cv2.KeyPoint], des: np.ndarray,frame_rgb: np.ndarray, frame_depth: np.ndarray, timestamp: Time, tf_buffer: tf2_ros.Buffer) -> None:
        for i, p in enumerate(kps):
            u, v = p.pt
            u = int(round(u))
            v = int(round(v))

            depth_value = frame_depth[v, u]
            b, g, r = frame_rgb[v, u]
            rgb = (int(r) << 16) | (int(g) << 8) | int(b)
            # Use only the i-th descriptor for this specific landmark
            landmark = Landmark(u=u, v=v, z=depth_value, kp=p, des=des[i], color=rgb)
            odom_coor = kinect_depth_to_odom(tf_buffer, landmark.get_kinect_coordinates())

            landmark.set_odom_coordinates(odom_coor)
            self.add_landmark(landmark)

    def get_visible_landmarks(self, camera_pos_odom, theta) -> VisualOdomMap:
        visible_landmarks = VisualOdomMap()

        if isinstance(camera_pos_odom, Coordinate):
            point = np.array([camera_pos_odom.x, camera_pos_odom.y, camera_pos_odom.z], dtype=float)
        else:
            point = np.array(camera_pos_odom, dtype=float)

        for l in self:
            land_pos = l.get_odom_coordinates()
            
            # Both camera and landmark are in odom_visual frame - direct comparison
            l_pos = np.array([land_pos.x, land_pos.y, land_pos.z])
            delta = l_pos - point

            distance = np.linalg.norm(delta)

            delta_u_angle = abs(atan2(CU, F)) # max angle in u direction
            delta_v_angle = abs(atan2(CV, F)) # max angle in v direction
        
            delta_azimuth = abs(atan2(delta[1], delta[0])-theta)
            delta_azimuth = normalize_angle(delta_azimuth)

            delta_altitude = abs(atan2(delta[2], np.linalg.norm(delta[:2])))
            delta_altitude = normalize_angle(delta_altitude)

            max_range = MAX_DEPTH/(cos(delta_azimuth)* cos(delta_altitude))
            min_range = MIN_DEPTH/(cos(delta_azimuth)* cos(delta_altitude))

            if delta_azimuth < delta_u_angle and delta_altitude < delta_v_angle and distance < max_range and distance > min_range:
                visible_landmarks.append(l)

        return visible_landmarks
        
    
    
    def get_descriptors(self) -> np.ndarray:
        return np.array([l.get_descriptor() for l in self])
    
    
    def get_kps(self) -> list[cv2.KeyPoint]:
        kps = []
        for l in self:
            kps.append(l.get_kp())
        return kps
    
    def get_depth(self) -> list[int]:
        return [l.get_depth() for l in self]
    
    def get_u_coordinates(self) -> list[int]:
        """Get all u pixel coordinates from all landmarks."""
        return [l.get_u() for l in self]
    
    def get_v_coordinates(self) -> list[int]:
        """Get all v pixel coordinates from all landmarks."""
        return [l.get_v() for l in self]
    
    def get_ages(self) -> list[int]:
        """Get all ages from all landmarks."""
        return [l.get_age() for l in self]
    
    def get_colors(self) -> list[int]:
        """Get all colors from all landmarks."""
        return [l.get_color() for l in self]
    
    def get_odom_coordinates(self) -> list:
        """Get all odom coordinates from all landmarks."""
        return [l.get_odom_coordinates() for l in self]
    
    def get_kinect_coordinates(self) -> list:
        """Get all kinect coordinates from all landmarks."""
        return [l.get_kinect_coordinates() for l in self]

    def publish_pointcloud_map(self, publisher, time: Time):
        from std_msgs.msg import Header
        h = Header()
        h.stamp = time
        h.frame_id = VISUAL_ODOM_FRAME_ID

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
        ]

        points_with_rgb = []
        for l in self:
            points_with_rgb.append(( l.get_odom_coordinates().x, l.get_odom_coordinates().y, l.get_odom_coordinates().z, l.get_color()))

        msg = point_cloud2.create_cloud(
            header=h,
            fields=fields,
            points=points_with_rgb
        )

        publisher.publish(msg)
        rclpy.logging.get_logger(__name__).info(f"PointCloud with {len(points_with_rgb)} points sent!")