#!/usr/bin/env python3

"""!
@file visual_odom_map.py
@package visual_odom.visual_odom_map
@brief Implements a persistent spatial landmark collection container tracking mapped environment features.
"""

from __future__ import annotations
from typing import Iterable, Optional, List, Tuple
from numpy.typing import NDArray

from rclpy.time import Time
import rclpy
from sensor_msgs.msg import PointField
from sensor_msgs_py import point_cloud2

import cv2
import numpy as np
from visual_odom.landmark import Landmark
from visual_odom.constants import (
    Coordinate, PixelCoordinate, VISUAL_ODOM_FRAME_ID
)
from visual_odom.tf_methods import kinect_depth_to_odom, pixel_to_kinect
import visual_odom.constants as constants


class VisualOdomMap(list[Landmark]):
    """!
    @brief Holds and manages persistent Landmark records mapped inside the particle trajectories.
    """

    def __init__(self, landmarks: Optional[Iterable[Landmark]] = None) -> None:
        """!
        @brief Initializes a VisualOdomMap collection instance.

        @param landmarks Optional collection of pre-existing landmark records.
        """
        super().__init__()
        self.descriptors = {}

        if landmarks is not None:
            self.extend(landmarks)

    def get_landmark(self, index: int) -> Landmark:
        """!
        @brief Retrieves a tracked landmark record by its indexing identifier.

        @param index Target list position.
        @return Tracked Landmark instance.
        """
        return self[index]
    
    def add_landmark(self, landmark: Landmark) -> None:
        """!
        @brief Appends a newly created landmark record to the collection.

        @param landmark Landmark target instance.
        """
        self.append(landmark)
    
    def get_all_landmarks(self) -> list[Landmark]:
        """!
        @brief Gets the raw list structure containing all active landmark instances.

        @return Collection list.
        """
        return list(self)
    
    def get_size(self) -> int:
        """!
        @brief Gets the total landmark population size currently inside the map.

        @return Landmark population size.
        """
        return len(self)

    def add_landmarks_from_kps(self, P_init: NDArray, kps: Iterable[cv2.KeyPoint], des: np.ndarray, frame_rgb: np.ndarray, kp_depth: np.ndarray, theta: float, pos_baselink: Coordinate) -> None:
        """!
        @brief Instantiates new landmarks from visual feature parameters and adds them to the map.

        Filters out spatial duplicates using binary ORB descriptor hash structures.

        @param P_init Initial covariance matrix assigned to newly registered features.
        @param kps Detected OpenCV KeyPoint structures.
        @param des Binary descriptors corresponding to kps.
        @param frame_rgb Color reference frame image.
        @param kp_depth Raw sensor depth value matches.
        @param theta Current robot heading yaw orientation.
        @param pos_baselink Current robot translation position.
        """
        for i, p in enumerate(kps):
            u, v = int(p.pt[0]), int(p.pt[1])
            depth_value = int(kp_depth[i])
            
            b, g, r = frame_rgb[v, u]
            rgb = (int(r) << 16) | (int(g) << 8) | int(b)
            
            # Extract the specific row descriptor matching the keypoint index
            landmark = Landmark(P_init=P_init, pixel_coor=PixelCoordinate(u, v, depth_value), kp=p, des=des[i], color=rgb)
            odom_coor = kinect_depth_to_odom(landmark.get_kinect_coordinates(), theta, pos_baselink)
            
            if odom_coor is None:
                rclpy.logging.get_logger(__name__).warning("TF projection calculation failed, skipping landmark")
                continue
                
            # Prevent duplicate landmark structures using descriptor byte signatures
            if landmark.get_descriptor().tobytes() in self.descriptors:
                continue
            self.descriptors[landmark.get_descriptor().tobytes()] = True

            landmark.set_odom_coordinates(odom_coor)
            self.add_landmark(landmark)

    def get_visible_landmarks(self, camera_pos: Coordinate, theta: float) -> VisualOdomMap:
        """!
        @brief Queries the map for all landmarks falling inside the camera's visual field-of-view frustum.

        @param camera_pos Position of the camera in the global odom frame.
        @param theta Current robot heading yaw orientation.
        @return New map collection containing only visible landmark candidates.
        """
        visible_landmarks = VisualOdomMap()
        for l in self:
            is_visible = l.is_visible(camera_pos, theta)
            if is_visible:
                visible_landmarks.append(l)
        return visible_landmarks
    
    def get_descriptors(self) -> np.ndarray:
        """!
        @brief Stack all internal visual binary descriptors as a single NumPy array for batch processing.

        @return 2D descriptor matrix matching active landmarks.
        """
        return np.array([l.get_descriptor() for l in self])
    
    def get_kps(self) -> list[cv2.KeyPoint]:
        """!
        @brief Gets a list of all active OpenCV KeyPoints tracked by the map.

        @return KeyPoints list.
        """
        kps = []
        for l in self:
            kps.append(l.get_kp())
        return kps
    
    def get_depth(self) -> list[int]:
        """!
        @brief Gets raw depth values from all active landmarks in the map.

        @return Depth values list.
        """
        return [l.get_depth() for l in self]
    
    def get_u_coordinates(self) -> list[int]:
        """!
        @brief Gets horizontal pixel indices from all active landmarks in the map.

        @return Horizontal columns list.
        """
        return [l.get_u() for l in self]
    
    def get_v_coordinates(self) -> list[int]:
        """!
        @brief Gets vertical pixel indices from all active landmarks in the map.

        @return Vertical rows list.
        """
        return [l.get_v() for l in self]
    
    def get_colors(self) -> list[int]:
        """!
        @brief Gets packed RGB rendering properties from all active landmarks in the map.

        @return Packed color integers list.
        """
        return [l.get_color() for l in self]
    
    def get_odom_coordinates(self) -> list[Coordinate]:
        """!
        @brief Gets estimated odom-frame coordinates from all active landmarks in the map.

        @return Coordinates list.
        """
        return [l.get_odom_coordinates() for l in self]
    
    def get_kinect_coordinates(self) -> list[Coordinate]:
        """!
        @brief Gets local camera-frame coordinates from all active landmarks in the map.

        @return Coordinates list.
        """
        return [l.get_kinect_coordinates() for l in self]

    def publish_pointcloud_map(self, publisher: rclpy.publisher.Publisher, time: Time) -> None:
        """!
        @brief Formats and publishes active landmarks as a standard PointCloud2 sensor message for RViz.

        @param publisher Target PointCloud2 publisher channel.
        @param time Target timeline timestamp.
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
            points_with_rgb.append((
                l.get_odom_coordinates().x,
                l.get_odom_coordinates().y,
                l.get_odom_coordinates().z,
                l.get_color()
            ))

        msg = point_cloud2.create_cloud(
            header=h,
            fields=fields,
            points=points_with_rgb
        )
        publisher.publish(msg)

    def cleanup_old_landmarks(self, visible_landmarks: VisualOdomMap, landmark_index: List[int]) -> None:
        """!
        @brief Manages landmark trust properties and purges unobserved/unstable features from memory.

        Increments trust metrics for actively observed features and applies decay factors to missed ones.

        @param visible_landmarks Subset collection containing currently visible landmarks.
        @param landmark_index Vector of matched inlier indices.
        """
        matched_landmark_indices = set(landmark_index)
        for i, l in enumerate(visible_landmarks):
            if i in matched_landmark_indices:
                l.increase_trust()
            else:
                l.decrease_trust()

        for l in visible_landmarks:
            if l.get_trust() < constants.parameters.min_landmark_trust:
                if l in self:
                    self.remove(l)

    def landmark_kalman_iteration(self, pos_baselink: Coordinate, theta: float, landmark_indices: List[int], kp_pos: List[PixelCoordinate]) -> None:
        """!
        @brief Runs a single EKF iteration step across all actively matched landmarks.

        @param pos_baselink Current position of the robot's base_link.
        @param theta Current robot heading yaw orientation.
        @param landmark_indices Map indices corresponding to matched landmarks.
        @param kp_pos Observed pixel coordinate measurements.
        """
        for idx, kp in zip(landmark_indices, kp_pos):
            self[idx].landmark_kalman_iteration(pos_baselink, theta, kp)

    def calculate_log_weight(self, pos_baselink: Coordinate, theta: float, landmark_indices: List[int], kp_pos: List[PixelCoordinate]) -> float:
        """!
        @brief Sums the log-likelihood of all matched landmarks to update particle log weights.

        @param pos_baselink Current position of the robot's base_link.
        @param theta Current robot heading yaw orientation.
        @param landmark_indices Map indices corresponding to matched landmarks.
        @param kp_pos Observed pixel coordinate measurements.
        @return The accumulated particle log-weight float value.
        """
        log_weight = 0.0
        for idx, kp in zip(landmark_indices, kp_pos):
            z_pos_odom = kinect_depth_to_odom(pixel_to_kinect(kp), theta, pos_baselink)
            likelihood = self[idx].calculate_likelihood(z_pos_odom, theta)
            
            if likelihood > 0.0:
                log_weight += np.log(likelihood)
            else:
                log_weight += np.log(1e-10)  # Use safe fallback baseline boundary values
        return log_weight