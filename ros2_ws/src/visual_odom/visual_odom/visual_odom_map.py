#!/usr/bin/env python3
from __future__ import annotations
from typing import Iterable, Optional
from .landmark import Landmark

from rclpy.time import Time

from math import atan2, cos

import rclpy
from sensor_msgs.msg import PointField
from sensor_msgs_py import point_cloud2

import cv2
import numpy as np
from visual_odom.landmark import *
from visual_odom.constants import *
from visual_odom.tf_methods import *


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
    

    def add_landmarks_from_kps(self, kps: Iterable[cv2.KeyPoint], des: np.ndarray,frame_rgb: np.ndarray, valid_kp_depth: np.ndarray, theta: float, pos_baselink: np.ndarray) -> None:
        """
        Add new landmarks to the map from keypoints, descriptors, and depth information.
        """
        for i, p in enumerate(kps):

            u, v = int(p.pt[0]), int(p.pt[1])

            depth_value = int(valid_kp_depth[i])
            
            b, g, r = frame_rgb[v, u]
            rgb = (int(r) << 16) | (int(g) << 8) | int(b)
            # Use only the i-th descriptor for this specific landmark
            landmark = Landmark(u=u, v=v, z=depth_value, kp=p, des=des[i], color=rgb)

            odom_coor = kinect_depth_to_odom(landmark.get_kinect_coordinates(), theta, pos_baselink)
            if odom_coor is None:
                rclpy.logging.get_logger(__name__).warning("TF failed, skipping landmark")
                continue
            landmark.set_odom_coordinates(odom_coor)
            self.add_landmark(landmark)
            

    def get_visible_landmarks(self, pos: Coordinate, theta: float) -> VisualOdomMap:
        """
        Get all landmarks that are currently visible from the given position and orientation.
        """
        visible_landmarks = VisualOdomMap()

        for l in self:
            is_visible = l.is_visible(pos, theta)

            if is_visible:
                visible_landmarks.append(l)

        if len(visible_landmarks) < 5:
            rclpy.logging.get_logger(__name__).info(
                f"Only a few visible landmarks: {len(visible_landmarks)}"
            )

        return visible_landmarks
    
    
    def get_descriptors(self) -> np.ndarray:
        """
        Get the descriptors of all landmarks in the map as a numpy array.
        """
        return np.array([l.get_descriptor() for l in self])
    
    
    def get_kps(self) -> list[cv2.KeyPoint]:
        """
        Get the keypoints of all landmarks in the map as a list.
        """
        kps = []
        for l in self:
            kps.append(l.get_kp())
        return kps
    
    def get_depth(self) -> list[int]:
        """Get all depth values from all landmarks."""
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
    
    def get_odom_coordinates(self) -> list[Coordinate]:
        """Get all odom coordinates from all landmarks."""
        return [l.get_odom_coordinates() for l in self]
    
    def get_kinect_coordinates(self) -> list[Coordinate]:
        """Get all kinect coordinates from all landmarks."""
        return [l.get_kinect_coordinates() for l in self]

    def publish_pointcloud_map(self, publisher, time: Time):
        """
        Publish the current map as a PointCloud2 message for visualization in RViz.
        """
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

    def age_and_cleanup_old_landmarks(self, visible_landmarks, landmark_index):
        """
        Increase the age of visible landmarks which are not in the current set of visible landmarks, and remove those which are too old and not matched in @MAX_LANDMARK_AGE frames.
        """
        for i, l in enumerate(visible_landmarks):
            if i in landmark_index:
                l.reset_age()
            else:
                l.increase_age()

        for l in visible_landmarks:
            if l.get_age() > MAX_LANDMARK_AGE:
                self.remove(l)