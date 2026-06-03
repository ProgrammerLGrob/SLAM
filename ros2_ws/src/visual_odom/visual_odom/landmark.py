#!/usr/bin/env python3
from math import atan2
import numpy as np
import cv2

from visual_odom.constants import *
from visual_odom.tf_methods import *
from visual_odom.ekf_landmark import *

class Landmark:
    """Landmark class for storing features detected in images."""
    
    def __init__(self, P_init: NDArray, pixel_coor: PixelCoordinate, kp: cv2.KeyPoint, des: np.ndarray, age: int = 0, color: int = 0, odom_coordinates: Coordinate = Coordinate(0.0, 0.0, 0.0)) -> None:
        """
        Initialize a Landmark.
        """
        self.pixel_coor = pixel_coor
        self.kp = kp
        self.des = des
        self.age = age
        self.color = color
        self.kinect_coordinates  = pixel_to_kinect(self.pixel_coor)
        self.odom_coordinates = odom_coordinates
        self.P = P_init
        self.ekf = ExtendedKalmanFilterLandmark(self.odom_coordinates, self.P)

    
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
    
    def get_age(self) -> int:
        """
        Get the age of the landmark.
        """
        return self.age
    
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
        self.ekf.x = odom_coordinates

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
    
    def is_visible(self, pos: Coordinate, theta: float) -> bool:
        delta = self.odom_coordinates - pos
        horizontal_dist = np.linalg.norm([delta.x, delta.y])
        distance = np.linalg.norm([delta.x, delta.y, delta.z])

        azimuth = normalize_angle(atan2(delta.y, delta.x) - theta)
        
        if abs(azimuth) > MAX_AZIMUTH:
            return False
        
        altitude = atan2(delta.z, horizontal_dist)

        # Hard FOV check
        if abs(altitude) > MAX_ALTITUDE:
            return False
        
        n = abs(cos(azimuth)*cos(altitude))
        if n == 0:
            return False

        if not (MIN_DEPTH/(n*1000) < distance < MAX_DEPTH/(n*1000)):
            return False

        return True
    

    def reset_age(self) -> None:
        """
        Reset the age of the landmark to 0.
        """
        self.age = 0

    def increase_age(self) -> None:
        """
        Increase the age of the landmark by 1.
        """
        self.age += 1

    def kalman_iteration(self, pos_baselink: Coordinate, theta: float, pixel_coor: PixelCoordinate) -> None:
        """
        Perform a Kalman iteration for the landmark's EKF.
        """
        self.odom_coordinates, self.P  = self.ekf.kalman_iteration(pos_baselink, theta, pixel_coor)

    


"""
@param P_i: 2D points in the first frame
@param Q_i: 2D points in the second frame
"""
def kabsch(P_i: np.ndarray, Q_i: np.ndarray, max_rotation_angle_deg: float = 15.0):

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

    return R, t, theta