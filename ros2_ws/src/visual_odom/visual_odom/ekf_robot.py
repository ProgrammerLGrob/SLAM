#!/usr/bin/env python3

"""!
@file ekf_robot.py
@package visual_odom.ekf_robot
@brief Extended Kalman Filter for estimating robot pose using wheel odometry and RANSAC visual updates.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple

from visual_odom.constants import (
    State, SIGMA_X_ODOM_WHEEL_Q, SIGMA_Y_ODOM_WHEEL_Q, SIGMA_THETA_ODOM_WHEEL_Q,
    NOISE_INCREMENT_RANSAC_FAILURE, MAX_ADDITIONAL_NOISE_RANSAC_FAILURE
)
import visual_odom.constants as constants


class ExtendedKalmanFilterRobot:
    """!
    @brief Extended Kalman Filter tracking the 2D planar robot base pose (x, y, theta).
    """

    def __init__(self, x: State, P: NDArray, Q: NDArray = None):
        """!
        @brief Initializes the robot EKF tracker.

        @param x Initial 2D state estimate of the robot platform.
        @param P Initial estimation error covariance matrix.
        @param Q Optional pre-calculated process noise covariance matrix.
        """
        self.x = x
        self.P = P
        if Q is not None:
            self.Q = Q
        else:
            self.set_Q()

        self.F = np.array([[1.0, 0.0, 0.0], 
                           [0.0, 1.0, 0.0], 
                           [0.0, 0.0, 1.0]])
        
        self.H = np.array([[1.0, 0.0, 0.0], 
                           [0.0, 1.0, 0.0], 
                           [0.0, 0.0, 1.0]])
        
        self.additional_noise = 0.0

    def prediction(self, delta_wheel_odom: State) -> None:
        """!
        @brief Predicts the robot pose forward based on relative wheel odometry increments.

        @param delta_wheel_odom Relative displacement step calculated from wheel encoders.
        """
        x_tt1 = self.x + delta_wheel_odom
        P_tt1 = self.F @ self.P @ self.F.T + self.Q
        
        self.x = x_tt1
        self.P = P_tt1

    def computeKalmanGain(self, P_tt1: NDArray) -> NDArray:
        """!
        @brief Computes the optimal Kalman gain matrix.

        @param P_tt1 Predicted robot state error covariance matrix.
        @return Calculated 3x3 Kalman gain matrix.
        """
        PHT = P_tt1 @ self.H.T
        HPHT = self.H @ PHT
        HPHTpRi = np.linalg.inv(HPHT + self.R)
        K = PHT @ HPHTpRi
        return K

    def update(self, z: State) -> None:
        """!
        @brief Fuses a RANSAC visual odometry pose calculation into the tracking state.

        @param z Evaluated pose estimate from the RANSAC visual alignment pipeline.
        @param inlier_ratio Percentage of robust inliers tracked during visual alignment.
        """
        self.set_R()
        K = self.computeKalmanGain(self.P)

        z_tt1 = self.x
        delta_z = np.array([z.x - z_tt1.x, z.y - z_tt1.y, z.theta - z_tt1.theta])
        
        delta = K @ delta_z
        x_t1t1 = self.x + State(delta[0], delta[1], delta[2])
        
        # Joseph-form covariance update to ensure numerical stability and positive semi-definiteness
        I_KH = np.eye(3) - K @ self.H
        P_t1t1 = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        self.x = x_t1t1
        self.P = P_t1t1

    def set_R(self) -> None:
        """!
        @brief Configures the visual odometry measurement noise covariance matrix R.

        The measurement noise scales dynamically based on recent RANSAC failures.

        @param inlier_ratio Ratio of inliers used during standard visual processing loops.
        """
        sigma =  constants.parameters.ransac_evaluation_tolerance + self.additional_noise
        self.R = np.array([[sigma**2, 0.0, 0.0], 
                           [0.0, sigma**2, 0.0], 
                           [0.0, 0.0, sigma**2]])

    def set_Q(self) -> None:
        """!
        @brief Configures the process noise covariance matrix Q based on configured standard deviations.
        """
        sigma_x = SIGMA_X_ODOM_WHEEL_Q
        sigma_y = SIGMA_Y_ODOM_WHEEL_Q
        sigma_theta = SIGMA_THETA_ODOM_WHEEL_Q
        self.Q = np.array([[sigma_x**2, 0.0, 0.0], 
                           [0.0, sigma_y**2, 0.0], 
                           [0.0, 0.0, sigma_theta**2]])

    def get_state(self) -> State:
        """!
        @brief Gets the current robot base tracking pose.

        @return Estimated State object containing positions and heading.
        """
        return self.x

    def get_covariance_p(self) -> NDArray:
        """!
        @brief Gets the current pose estimation error covariance matrix.

        @return 3x3 error covariance matrix.
        """
        return self.P

    def add_noise_to_R(self) -> None:
        """!
        @brief Increments the tracking update error offset following a RANSAC visual execution failure.

        Ensures that repeated visual tracking errors gracefully inflate uncertainty parameters.
        """
        self.additional_noise = min(
            self.additional_noise + NOISE_INCREMENT_RANSAC_FAILURE,
            MAX_ADDITIONAL_NOISE_RANSAC_FAILURE
        )