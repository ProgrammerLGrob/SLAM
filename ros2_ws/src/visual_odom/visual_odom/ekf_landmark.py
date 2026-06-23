#!/usr/bin/env python3

"""!
@file ekf_landmark.py
@package visual_odom.ekf_landmark
@brief Extended Kalman Filter implementation for tracking individual landmark positions.
"""

from math import sin, cos
import numpy as np
from typing import Tuple
from numpy.typing import NDArray

from visual_odom.constants import (
    Coordinate, PixelCoordinate, ERROR_MIN_DEPTH, ERROR_QUADRATIC_DEPTH,
    MIN_DEPTH, SIGMA_PIXEL, F, CU, CV, ROT_BK
)
from visual_odom.tf_methods import kinect_depth_to_baselink, pixel_to_kinect


class ExtendedKalmanFilterLandmark:
    """!
    @brief Extended Kalman Filter for tracking an individual 3D landmark in the global odom frame.
    """

    def __init__(self, x: Coordinate, P: NDArray, pix_coor: PixelCoordinate, R: NDArray = None) -> None:
        """!
        @brief Initializes the landmark EKF state and covariance matrices.

        @param x Initial 3D coordinate state estimate of the landmark.
        @param P Initial estimation error covariance matrix.
        @param pix_coor Initial pixel measurement coordinate used to initialize noise.
        @param R Optional pre-calculated measurement noise covariance matrix.
        """
        self.x = x
        self.P = P
        if R is None:
            self.set_R(pix_coor)  # Initialize R based on the depth of the landmark
        else:
            self.R = R
        self.Q = self.calculate_Q_matrix()

    def landmark_kalman_iteration(self, pos_baselink: Coordinate, theta: float, pixel_coor: PixelCoordinate) -> Tuple[Coordinate, NDArray]:
        """!
        @brief Performs a full prediction and update cycle for the landmark position.

        @param pos_baselink The current position of the robot's base_link in the odom frame.
        @param theta The current yaw angle heading of the robot.
        @param pixel_coor The newly observed pixel coordinate of this landmark.
        @return A tuple containing the updated Coordinate state and the updated covariance matrix.
        """
        x_tt1, P_tt1 = self.prediction(self.x)
        updated_x, updated_P = self.update(x_tt1, P_tt1, pos_baselink, theta, pixel_coor)

        self.x = updated_x
        self.P = updated_P
        return self.x, self.P

    def state_func(self, x: Coordinate) -> Coordinate:
        """!
        @brief State transition model representing stationary landmarks.

        @param x The current state estimate of the landmark.
        @return The predicted landmark state (identity transformation).
        """
        return x

    def calculate_jacobian_F(self) -> NDArray:
        """!
        @brief Calculates the state transition Jacobian matrix.

        @return A 3x3 identity Jacobian matrix since the landmarks are static.
        """
        F = np.array([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]])
        return F

    def calculate_Q_matrix(self) -> NDArray:
        """!
        @brief Calculates the stationary process noise covariance matrix.

        @return A 3x3 diagonal process noise covariance matrix.
        """
        Q = np.eye(3) * 1e-6
        return Q

    def meas_func(self, x_tt1: Coordinate, pos_robot: Coordinate, theta_robot: float) -> NDArray:
        """!
        @brief Measurement projection model mapping odom-frame coordinates into base_link space.

        @param x_tt1 Predicted landmark coordinate in the global odom frame.
        @param pos_robot Current robot position coordinates in the global odom frame.
        @param theta_robot Current robot yaw heading orientation in radians.
        @return Projected 3D coordinate vector in the robot's local base_link frame.
        """
        c = cos(theta_robot)
        s = sin(theta_robot)
        R_theta = np.array([[c, -s, 0.0],
                            [s, c, 0.0],
                            [0.0, 0.0, 1.0]])
        h_x = R_theta.T @ (np.array([x_tt1.x, x_tt1.y, x_tt1.z]) - np.array([pos_robot.x, pos_robot.y, pos_robot.z]))
        return h_x

    def calculate_jacobian_H(self, theta_robot: float) -> NDArray:
        """!
        @brief Calculates the measurement model Jacobian matrix.

        @param theta_robot Current robot yaw heading orientation in radians.
        @return 3x3 projection Jacobian matrix representing rotation into local base_link.
        """
        c = cos(theta_robot)
        s = sin(theta_robot)
        R_theta = np.array([[c, -s, 0.0],
                            [s, c, 0.0],
                            [0.0, 0.0, 1.0]])
        H = R_theta.T
        return H

    def sigma_R_approximation(self, kp: PixelCoordinate) -> NDArray:
        """!
        @brief Approximates the raw pixel-space and depth-space measurement variances.

        @param kp The observed PixelCoordinate containing pixel indices and raw depth.
        @return 3x3 diagonal measurement covariance matrix in pixel/depth space.
        """
        z_m = kp.z / 1000.0
        s_z = ERROR_MIN_DEPTH + ERROR_QUADRATIC_DEPTH * (z_m - MIN_DEPTH / 1000.0) ** 2

        R_sigma = np.array([[SIGMA_PIXEL**2, 0.0, 0.0],
                            [0.0, SIGMA_PIXEL**2, 0.0],
                            [0.0, 0.0, s_z**2]])
        return R_sigma

    def calculate_R_sigma_matrix(self, kp: PixelCoordinate) -> NDArray:
        """!
        @brief Calculates the measurement noise covariance matrix R transformed into the base_link frame.

        Applies the pinhole projection Jacobian followed by the camera-to-base_link rotation matrix.

        @param kp The observed PixelCoordinate containing current feature matching data.
        @return 3x3 measurement covariance matrix in the local robot base_link frame.
        """
        z_m = kp.z / 1000.0

        R_sigma_pixel = self.sigma_R_approximation(kp)
        J_kinect = np.array([[z_m / F, 0.0, (kp.u - CU) / F],
                             [0.0, z_m / F, (kp.v - CV) / F],
                             [0.0, 0.0, 1.0]])
        
        R_sigma_kinect = J_kinect @ R_sigma_pixel @ J_kinect.T  # Transform pixel variance to Kinect coordinates
        J_baselink = ROT_BK  # Rotation matrix from Kinect to Base Link frame
        R_sigma_base_link = J_baselink @ R_sigma_kinect @ J_baselink.T

        return R_sigma_base_link

    def prediction(self, x: Coordinate) -> Tuple[Coordinate, NDArray]:
        """!
        @brief Runs the EKF prediction step.

        Propagates state and covariance forwards in time with process noise.

        @param x Current coordinate state estimate.
        @return A tuple of (predicted state, predicted covariance).
        """
        x_tt1 = self.state_func(x)
        self.set_Q()
        self.set_jacobian_F()

        P_tt1 = self.JF @ self.P @ self.JF.T + self.Q
        return x_tt1, P_tt1

    def predictMeasurement(self, x_tt1: Coordinate, pos_robot: Coordinate, theta_robot: float) -> NDArray:
        """!
        @brief Projects the predicted state into the expected measurement vector space.

        @param x_tt1 Predicted landmark coordinate in the global odom frame.
        @param pos_robot Current robot base link position coordinates.
        @param theta_robot Current robot yaw heading orientation.
        @return Projected local base_link coordinate array.
        """
        pmeas = self.meas_func(x_tt1, pos_robot, theta_robot)
        return pmeas

    def computeKalmanGain(self, P_tt1: NDArray) -> NDArray:
        """!
        @brief Computes the optimal Kalman gain matrix.

        @param P_tt1 Predicted state covariance matrix.
        @return Calculated 3x3 Kalman gain matrix.
        """
        PHT = P_tt1 @ self.JH.T
        HPHT = self.JH @ PHT
        HPHTpRi = np.linalg.inv(HPHT + self.R)
        K = PHT @ HPHTpRi
        return K

    def update(self, x_tt1: Coordinate, P_tt1: NDArray, pos_robot: Coordinate, theta_robot: float, kp: PixelCoordinate) -> Tuple[Coordinate, NDArray]:
        """!
        @brief Performs the measurement update step of the EKF using Joseph-form covariance updates.

        @param x_tt1 Predicted landmark coordinate in the global odom frame.
        @param P_tt1 Predicted state covariance matrix.
        @param pos_robot Current global robot position coordinates.
        @param theta_robot Current robot yaw heading orientation in radians.
        @param kp PixelCoordinate corresponding to the observed landmark feature.
        @return A tuple of (updated state, updated covariance).
        """
        z = kinect_depth_to_baselink(pixel_to_kinect(kp))

        self.set_jacobian_H(theta_robot)
        self.set_R(kp)
        K = self.computeKalmanGain(P_tt1)

        z_tt1 = self.predictMeasurement(x_tt1, pos_robot, theta_robot)
        delta_z = np.array([z.x - z_tt1[0], z.y - z_tt1[1], z.z - z_tt1[2]])
            
        delta = K @ delta_z
        x = x_tt1 + Coordinate(delta[0], delta[1], delta[2])
        
        # Joseph-form covariance update to ensure numerical stability and positive semi-definiteness
        I_KH = np.eye(3) - K @ self.JH
        P = I_KH @ P_tt1 @ I_KH.T + K @ self.R @ K.T
        return x, P

    def set_jacobian_F(self) -> None:
        """!
        @brief Updates the internal state transition Jacobian.
        """
        self.JF = self.calculate_jacobian_F()

    def set_jacobian_H(self, theta_robot: float) -> None:
        """!
        @brief Updates the internal measurement model Jacobian.

        @param theta_robot Current robot yaw heading orientation in radians.
        """
        self.JH = self.calculate_jacobian_H(theta_robot)

    def set_R(self, kp: PixelCoordinate) -> None:
        """!
        @brief Calculates and stores the measurement noise covariance matrix.

        @param kp Observed pixel coordinate from the raw sensor stream.
        """
        self.R = self.calculate_R_sigma_matrix(kp)

    def set_Q(self) -> None:
        """!
        @brief Stores the process noise covariance matrix.
        """
        self.Q = self.calculate_Q_matrix()

    def get_R(self) -> NDArray:
        """!
        @brief Gets the active measurement noise covariance matrix in the base_link frame.

        @return 3x3 covariance matrix.
        """
        return self.R