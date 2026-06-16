#!/usr/bin/env python3
from math import atan2, pi
import numpy as np
import cv2

from visual_odom.constants import *
from visual_odom.tf_methods import *
from visual_odom.ekf_landmark import *
import visual_odom.constants as constants

class Landmark:
    """Landmark class for storing features detected in images."""
    
    def __init__(self, P_init: NDArray, pixel_coor: PixelCoordinate, kp: cv2.KeyPoint, des: np.ndarray, trust: float = 30, color: int = 0, odom_coordinates: Coordinate = Coordinate(0.0, 0.0, 0.0)) -> None:
        """
        Initialize a Landmark.
        """
        self.pixel_coor = pixel_coor
        self.kp = kp
        self.des = des
        self.trust = trust
        self.color = color
        self.kinect_coordinates  = pixel_to_kinect(self.pixel_coor)
        self.odom_coordinates = odom_coordinates
        self.P = P_init
        self.landmark_ekf = ExtendedKalmanFilterLandmark(self.odom_coordinates, self.P, pix_coor=self.pixel_coor)
    
    def get_descriptor(self) -> np.ndarray:
        """
        Get the descriptor of the landmark.
        """
        return self.des
    
    def get_kp(self) -> cv2.KeyPoint:
        """
        Get the keypoint of the landmark.
        """
        return self.kp
    
    def get_depth(self) -> int:
        """
        Get the depth value of the landmark.
        """
        return self.pixel_coor.z
    
    def get_u(self) -> int:
        """
        Get the u pixel coordinate.
        """
        return self.pixel_coor.u
    
    def get_v(self) -> int:
        """
        Get the v pixel coordinate.
        """
        return self.pixel_coor.v
    
    def get_trust(self) -> float:
        """
        Get the trust value of the landmark.
        """
        return self.trust

    def get_color(self) -> int:
        """
        Get the color value of the landmark.
        """
        return self.color
    
    def get_odom_coordinates(self) -> Coordinate:
        """
        Get the odom coordinates of the landmark.
        """
        return self.odom_coordinates

    def set_odom_coordinates(self, odom_coordinates: Coordinate) -> None:
        """
        Set the odom coordinates of the landmark.
        """
        self.odom_coordinates = odom_coordinates
        self.landmark_ekf.x = odom_coordinates

    def get_kinect_coordinates(self) -> Coordinate:
        """
        Get the kinect coordinates of the landmark.
        """
        return self.kinect_coordinates
    
    def get_P(self) -> NDArray:
        """
        Get the covariance matrix P of the landmark.
        """
        return self.P
    
    def is_visible(self, pos_camera_odom: Coordinate, theta_robot: float) -> bool:

        delta_odom = self.odom_coordinates - pos_camera_odom

        c = cos(theta_robot)
        s = sin(theta_robot)
        
        delta_base_link = Coordinate(0.0, 0.0, 0.0)
        delta_base_link.x =  c * delta_odom.x + s * delta_odom.y
        delta_base_link.y = -s * delta_odom.x + c * delta_odom.y
        delta_base_link.z =  delta_odom.z  

        if delta_base_link.x < MIN_DEPTH/1000 or delta_base_link.x > constants.parameters.max_depth/1000:
            return False
        
        azimuth = atan2(delta_base_link.y, delta_base_link.x)
        
        max_azimuth = CAMERA_ANGLE_HOR_RAD/ 2.0
        if abs(azimuth) > max_azimuth:
            return False

        altitude = atan2(delta_base_link.z, delta_base_link.x)
        
        max_alt = CAMERA_ANGLE_VER_RAD/ 2.0
        if abs(altitude) > max_alt:
            return False

        return True
    

    def reset_trust(self) -> None:
        """
        Reset the trust value of the landmark to its initial value.
        """
        self.trust = 30.0

    def increase_trust(self) -> None:
        """
        Increase the trust value of the landmark by INCREASE_TRUST_VALUE.
        """
        self.trust += INCREASE_TRUST_VALUE  # Increment trust by a fraction of the minimum trust threshold

    def decrease_trust(self) -> None:
        self.trust *= DECREASE_TRUST_FACTOR  # Exponentieller Abfall, z.B. 0.8 oder 0.9 pro Frame ohne Sichtung



    def landmark_kalman_iteration(self, pos_baselink: Coordinate, theta: float, pixel_coor: PixelCoordinate) -> None:
        """
        Perform a Kalman iteration for the landmark's EKF.
        """
        self.odom_coordinates, self.P = self.landmark_ekf.landmark_kalman_iteration(pos_baselink, theta, pixel_coor)

    def calculate_likelihood(self, z_pos_odom: Coordinate, theta: float) -> float:
        """
        Calculate the likelihood of the landmark. Must be after the kalman iteration
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
    



def kabsch(P_i: np.ndarray, Q_i: np.ndarray, max_rotation_angle_deg: float = 15.0):
    """
    @param P_i: 2D points in the first frame
    @param Q_i: 2D points in the second frame
    """
     
    if P_i.shape[0] > 1 and Q_i.shape[0] > 1:
        m_P = np.mean(P_i,axis=0)
        m_Q = np.mean(Q_i,axis=0)


        P_i_centered = P_i - m_P
        Q_i_centered = Q_i - m_Q 
        first_sum = np.sum(Q_i_centered[:,0]*P_i_centered[:,1] - Q_i_centered[:,1]*P_i_centered[:,0])
        sec_sum = np.sum(Q_i_centered[:,0]*P_i_centered[:,0] + Q_i_centered[:,1]*P_i_centered[:,1])
    else:
        rclpy.logging.get_logger("Kabsch").info(f"Only one point pair, cannot compute rotation, skipping Kabsch")
        return None, None, None
    
    theta = atan2(first_sum, sec_sum)
    
    #if abs(theta) > max_rotation_angle_deg * pi / 180.0:
     #  return None, None, None
    
    R = np.array([[ np.cos(theta), -np.sin(theta)],
                  [ np.sin(theta),  np.cos(theta)]])
    
    t = m_P - R @ m_Q
        
    return R, t, theta

