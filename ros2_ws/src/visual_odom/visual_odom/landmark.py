#!/usr/bin/env python3
from math import atan2
import numpy as np
import cv2

from visual_odom.constants import *
from visual_odom.tf_methods import *

class Landmark:
    """Landmark class for storing features detected in images."""
    
    def __init__(self, u: int, v: int, z: int, kp: cv2.KeyPoint, des: np.ndarray, age: int = 0, color: int = 0, odom_coordinates: Coordinate = Coordinate(0.0, 0.0, 0.0)) -> None:
        """
        Initialize a Landmark.
        """
        self.u = u
        self.v = v
        self.z = z
        self.kp = kp
        self.des = des
        self.age = age
        self.color = color
        self.kinect_coordinates  = pixel_to_kinect(u, v, z)
        self.odom_coordinates = odom_coordinates

    
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
        return self.z
    
    def get_u(self) -> int:
        """
        Get the u pixel coordinate.
        """
        return self.u
    
    def get_v(self) -> int:
        """
        Get the v pixel coordinate.
        """
        return self.v
    
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

    def get_kinect_coordinates(self) -> Coordinate:
        """
        Get the kinect coordinates of the landmark.
        """
        return self.kinect_coordinates


"""
@param P_i: 2D points in the first frame
@param Q_i: 2D points in the second frame
"""
def kabsch(P_i: np.ndarray, Q_i: np.ndarray):

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
    
    
    R = np.array([[ np.cos(theta), -np.sin(theta)],
                  [ np.sin(theta),  np.cos(theta)]])
    
    t = m_P - R @ m_Q

    return R, t, theta

