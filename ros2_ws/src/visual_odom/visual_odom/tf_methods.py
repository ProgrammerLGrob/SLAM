
from math import sin,cos

import rclpy
from visual_odom.constants import *
from scipy.spatial.transform import Rotation
from numpy.typing import NDArray
from rclpy.time import Time

import numpy as np



def kinect_depth_to_odom(kinect_point: Coordinate, theta: float, pos_baselink: np.ndarray) -> Coordinate:
    """
    Transform a point from the Kinect frame into the odom frame using the LATEST available TF.
    Ignores timestamp and uses the most recent transformation to avoid extrapolation errors.
    """
    c = cos(theta)
    s = sin(theta)
    R = [[c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0]]
    pos = pos_baselink + R@np.array([kinect_point.z+CAMERA_POS_IN_BASELINK.x, -kinect_point.x+CAMERA_POS_IN_BASELINK.y, -kinect_point.y+CAMERA_POS_IN_BASELINK.z])

    return pos

def kinect_depth_to_baselink(kinect_point: Coordinate) -> Coordinate:
    """
    Transform a point from the Kinect frame into the baselink frame using the LATEST available TF.
    Ignores timestamp and uses the most recent transformation to avoid extrapolation errors.
    """
    
    pos = np.array([kinect_point.z+CAMERA_POS_IN_BASELINK.x, -kinect_point.x+CAMERA_POS_IN_BASELINK.y, -kinect_point.y+CAMERA_POS_IN_BASELINK.z])
    return pos