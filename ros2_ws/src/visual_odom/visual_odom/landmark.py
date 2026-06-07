#!/usr/bin/env python3
from math import atan2
import numpy as np
import cv2

from visual_odom.constants import *
from visual_odom.tf_methods import *
from visual_odom.ekf_landmark import *

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
        self.landmark_ekf = ExtendedKalmanFilterLandmark(self.odom_coordinates, self.P)
    
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
        
       
        local_x =  c * delta_odom.x + s * delta_odom.y
        local_y = -s * delta_odom.x + c * delta_odom.y
        local_z =  delta_odom.z  

        if local_x < MIN_DEPTH/1000 or local_x > MAX_DEPTH/1000:
            return False
        
        azimuth = atan2(local_y, local_x)
        
        max_azimuth = CAMERA_ANGLE_HOR_RAD/ 2.0
        if abs(azimuth) > max_azimuth:
            return False

        altitude = atan2(local_z, local_x)
        
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
        R_noice = self.landmark_ekf.get_R()
        c = cos(theta)
        s = sin(theta)

        R_ob = np.array([
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0]
        ])
        s_matrix = self.P + R_ob*R_noice*R_ob.T

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
    m_P = np.mean(P_i,axis=0)
    m_Q = np.mean(Q_i,axis=0)


    P_i_centered = P_i - m_P
    Q_i_centered = Q_i - m_Q   

    if P_i_centered.shape[0] == 0 or Q_i_centered.shape[0] == 0:
        return np.eye(2), np.zeros((2,1)), 0.0
    elif P_i_centered.shape[0] == 1 or Q_i_centered.shape[0] == 1:
        first_sum = Q_i_centered[0,0]*P_i_centered[0,1] - Q_i_centered[0,1]*P_i_centered[0,0]
        sec_sum = Q_i_centered[0,0]*P_i_centered[0,0] + Q_i_centered[0,1]*P_i_centered[0,1]
    else:
        first_sum = np.sum(Q_i_centered[:,0]*P_i_centered[:,1] - Q_i_centered[:,1]*P_i_centered[:,0])
        sec_sum = np.sum(Q_i_centered[:,0]*P_i_centered[:,0] + Q_i_centered[:,1]*P_i_centered[:,1])

    theta = atan2(first_sum, sec_sum)
    
    if abs(theta) > np.deg2rad(max_rotation_angle_deg):
        return None, None, None
    
    R = np.array([[ np.cos(theta), -np.sin(theta)],
                  [ np.sin(theta),  np.cos(theta)]])
    
    t = m_P - R @ m_Q

    """
    # Pseudo-Code nach dem RANSAC/Kabsch-Schritt im Node:
    MAX_ALLOWED_SPEED_PER_FRAME = 0.08 # 5 cm pro Frame max bei Vorwärtsfahrt
    MAX_ALLOWED_ROTATION_PER_FRAME = 0.08 # ca. 4,5 Grad pro Frame max

    if abs(sqrt(t[0]**2 + t[1]**2)) > MAX_ALLOWED_SPEED_PER_FRAME or abs(theta) > MAX_ALLOWED_ROTATION_PER_FRAME:
        t = np.array([0.0, 0.0])
        theta = 0.0
        R = np.array([[ np.cos(theta), -np.sin(theta)],
                  [ np.sin(theta),  np.cos(theta)]])
    """

        
    return R, t, theta

