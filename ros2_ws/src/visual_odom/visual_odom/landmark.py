#!/usr/bin/env python3

"""!
@file landmark.py
@package visual_odom.landmark
@brief Defines the Landmark data class and the Kabsch optimal 2D rigid alignment algorithm.
"""

from math import atan2, pi, cos, sin
import numpy as np
import cv2
from numpy.typing import NDArray
from typing import Iterable, Optional, List, Tuple


from visual_odom.constants import (
    Coordinate, PixelCoordinate, MIN_DEPTH, CAMERA_ANGLE_HOR_RAD,
    CAMERA_ANGLE_VER_RAD, INCREASE_TRUST_VALUE, DECREASE_TRUST_FACTOR,
    MIN_DET_VALUE
)
import visual_odom.constants as constants
from visual_odom.tf_methods import pixel_to_kinect
from visual_odom.ekf_landmark import ExtendedKalmanFilterLandmark


class Landmark:
    """!
    @brief Landmark class for storing visual features and performing spatial estimation.
    """
    
    def __init__(self, P_init: NDArray, pixel_coor: PixelCoordinate, kp: cv2.KeyPoint, des: np.ndarray, trust: float = 30.0, color: int = 0, odom_coordinates: Coordinate = Coordinate(0.0, 0.0, 0.0)) -> None:
        """!
        @brief Initializes a Landmark tracking instance.

        @param P_init Initial EKF error covariance matrix.
        @param pixel_coor 2D pixel coordinate containing columns, rows, and raw depth values.
        @param kp OpenCV KeyPoint object associated with this visual feature.
        @param des Binary descriptor array of the ORB keypoint.
        @param trust Initial confidence tracking metric assigned to this feature.
        @param color RGB packing integer value used to render point cloud structures.
        @param odom_coordinates Initial transformed Coordinate frame estimation.
        """
        self.pixel_coor = pixel_coor
        self.kp = kp
        self.des = des
        self.trust = trust
        self.color = color
        self.kinect_coordinates = pixel_to_kinect(self.pixel_coor)
        self.odom_coordinates = odom_coordinates
        self.P = P_init
        self.landmark_ekf = ExtendedKalmanFilterLandmark(self.odom_coordinates, self.P, pix_coor=self.pixel_coor)
    
    def get_descriptor(self) -> np.ndarray:
        """!
        @brief Gets the binary feature descriptor array.

        @return 1D feature descriptor.
        """
        return self.des
    
    def get_kp(self) -> cv2.KeyPoint:
        """!
        @brief Gets the corresponding OpenCV KeyPoint.

        @return KeyPoint metadata.
        """
        return self.kp
    
    def get_depth(self) -> float:
        """!
        @brief Gets the raw depth tracking value.

        @return Sensor depth value in millimeters.
        """
        return self.pixel_coor.z
    
    def get_u(self) -> float:
        """!
        @brief Gets the horizontal pixel coordinate.

        @return Column pixel index.
        """
        return self.pixel_coor.u
    
    def get_v(self) -> float:
        """!
        @brief Gets the vertical pixel coordinate.

        @return Row pixel index.
        """
        return self.pixel_coor.v
    
    def get_trust(self) -> float:
        """!
        @brief Gets the current visual trust tracking parameter.

        @return Trust metric.
        """
        return self.trust

    def get_color(self) -> int:
        """!
        @brief Gets the color property used for visual representation.

        @return Packed color integer.
        """
        return self.color
    
    def get_odom_coordinates(self) -> Coordinate:
        """!
        @brief Gets the estimated global odom-frame coordinates.

        @return Coordinate object in meters.
        """
        return self.odom_coordinates

    def set_odom_coordinates(self, odom_coordinates: Coordinate) -> None:
        """!
        @brief Sets the global odom-frame coordinates and updates the internal EKF.

        @param odom_coordinates New Coordinate location.
        """
        self.odom_coordinates = odom_coordinates
        self.landmark_ekf.x = odom_coordinates

    def get_kinect_coordinates(self) -> Coordinate:
        """!
        @brief Gets the local Kinect frame coordinates.

        @return Coordinate object in meters.
        """
        return self.kinect_coordinates
    
    def get_P(self) -> NDArray:
        """!
        @brief Gets the estimated covariance matrix.

        @return 3x3 error covariance matrix.
        """
        return self.P
    
    def is_visible(self, pos_camera_odom: Coordinate, theta_robot: float) -> bool:
        """!
        @brief Checks if the landmark falls within the horizontal and vertical field of view of the sensor.

        @param pos_camera_odom Position of the camera in the global odom frame.
        @param theta_robot Robot heading yaw orientation in radians.
        @return True if the landmark is within depth, azimuth, and altitude boundaries, otherwise False.
        """
        delta_odom = self.odom_coordinates - pos_camera_odom

        c = cos(theta_robot)
        s = sin(theta_robot)
        
        delta_base_link = Coordinate(0.0, 0.0, 0.0)
        delta_base_link.x = c * delta_odom.x + s * delta_odom.y
        delta_base_link.y = -s * delta_odom.x + c * delta_odom.y
        delta_base_link.z = delta_odom.z  

        if delta_base_link.x < MIN_DEPTH / 1000.0 or delta_base_link.x > constants.parameters.max_depth / 1000.0:
            return False
        
        azimuth = atan2(delta_base_link.y, delta_base_link.x)
        max_azimuth = CAMERA_ANGLE_HOR_RAD / 2.0
        if abs(azimuth) > max_azimuth:
            return False

        altitude = atan2(delta_base_link.z, delta_base_link.x)
        max_alt = CAMERA_ANGLE_VER_RAD / 2.0
        if abs(altitude) > max_alt:
            return False

        return True

    def reset_trust(self) -> None:
        """!
        @brief Resets the visual tracking trust property to its initial baseline value of 30.0.
        """
        self.trust = 30.0

    def increase_trust(self) -> None:
        """!
        @brief Increments the trust value after the landmark is successfully re-observed.
        """
        self.trust += INCREASE_TRUST_VALUE

    def decrease_trust(self) -> None:
        """!
        @brief Applies an exponential decay to the trust value when the landmark is missed.
        """
        self.trust *= DECREASE_TRUST_FACTOR

    def landmark_kalman_iteration(self, pos_baselink: Coordinate, theta: float, pixel_coor: PixelCoordinate) -> None:
        """!
        @brief Runs a prediction-update EKF cycle to refine estimated global odom coordinates.

        @param pos_baselink Current position of the robot's base_link in the odom frame.
        @param theta Current yaw angle heading of the robot.
        @param pixel_coor Updated raw sensor pixel observation.
        """
        self.odom_coordinates, self.P = self.landmark_ekf.landmark_kalman_iteration(pos_baselink, theta, pixel_coor)

    def calculate_likelihood(self, z_pos_odom: Coordinate, theta: float) -> float:
        """!
        @brief Calculates the measurement likelihood of the landmark given its current state.

        Must be called after running the EKF tracking iteration.

        @param z_pos_odom Measured coordinate in the odom frame.
        @param theta Current robot heading yaw orientation.
        @return Gaussian probability density likelihood value.
        """
        R_noise = self.landmark_ekf.get_R()
        c = cos(theta)
        s = sin(theta)

        R_ob = np.array([
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0]
        ])
        s_matrix = self.P + R_ob @ R_noise @ R_ob.T

        det_s = np.linalg.det(s_matrix)
        if det_s <= MIN_DET_VALUE:
            det_s = MIN_DET_VALUE

        error = z_pos_odom - self.odom_coordinates
        error_vector = np.array([error.x, error.y, error.z])
        
        likelihood = (1.0 / np.sqrt((2 * np.pi) ** 3 * det_s)) * np.exp(-0.5 * error_vector.T @ np.linalg.inv(s_matrix) @ error_vector)
        return likelihood


def kabsch(P_i: np.ndarray, Q_i: np.ndarray, max_rotation_angle_deg: float = 15.0) -> Tuple[np.ndarray, np.ndarray, float]:
    """!
    @brief Computes the optimal 2D rotation and translation to align point set Q onto set P.

    Implements a 2D closed-form Kabsch algorithm to find least-squares alignment parameters.

    @param P_i 2D point coordinate array in the reference coordinate frame.
    @param Q_i 2D point coordinate array in the target coordinate frame.
    @param max_rotation_angle_deg Plausibility constraint capping estimated rotation.
    @return A tuple containing:
            - R: Optimal 2x2 rotation matrix, or None if degenerate.
            - t: Optimal 2D translation offset vector, or None if degenerate.
            - theta: Rotational heading orientation change in radians, or None if degenerate.
    """
    import rclpy
    
    if P_i.shape[0] > 1 and Q_i.shape[0] > 1:
        m_P = np.mean(P_i, axis=0)
        m_Q = np.mean(Q_i, axis=0)

        P_i_centered = P_i - m_P
        Q_i_centered = Q_i - m_Q 
        first_sum = np.sum(Q_i_centered[:, 0] * P_i_centered[:, 1] - Q_i_centered[:, 1] * P_i_centered[:, 0])
        sec_sum = np.sum(Q_i_centered[:, 0] * P_i_centered[:, 0] + Q_i_centered[:, 1] * P_i_centered[:, 1])
    else:
        rclpy.logging.get_logger("Kabsch").info("Only one point pair, cannot compute rotation, skipping Kabsch")
        return None, None, None
    
    theta = atan2(first_sum, sec_sum)
    
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    
    t = m_P - R @ m_Q
        
    return R, t, theta