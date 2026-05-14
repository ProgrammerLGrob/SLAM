#!/usr/bin/env python3
from math import sin,cos
from visual_odom.constants import *

import numpy as np


def kinect_depth_to_odom(kinect_point: Coordinate, theta: float, pos_baselink: Coordinate) -> Coordinate:
    """
    Transform a point from the Kinect frame into the odom frame using the LATEST available TF.
    Ignores timestamp and uses the most recent transformation to avoid extrapolation errors.
    """
    c = cos(theta)
    s = sin(theta)
    R = np.array([[c, -s, 0.0],
                  [s, c, 0.0],
                  [0.0, 0.0, 1.0]])

    pos_kb = kinect_depth_to_baselink(kinect_point)

    pos_odom = R@np.array([pos_kb.x, pos_kb.y, pos_kb.z])

    pos = pos_baselink + Coordinate(float(pos_odom[0]), float(pos_odom[1]), float(pos_odom[2]))
    return pos

def kinect_depth_to_baselink(kinect_point: Coordinate) -> Coordinate:
    """
    Transform a point from the Kinect frame into the baselink frame using the LATEST available TF.
    Ignores timestamp and uses the most recent transformation to avoid extrapolation errors.
    """

    pos_baselink = ROT_KB@np.array([kinect_point.x, kinect_point.y, kinect_point.z])
    pos = CAMERA_POS_IN_BASELINK + Coordinate(float(pos_baselink[0]), float(pos_baselink[1]), float(pos_baselink[2]))
    
    return pos

def odom_to_baselink(odom_point: Coordinate, theta: float, pos_baselink: Coordinate) -> Coordinate:
    """
    Transform a point from the odom frame into the baselink frame using the LATEST available TF.
    Ignores timestamp and uses the most recent transformation to avoid extrapolation errors.
    """
    c = cos(theta)
    s = sin(theta)
    R = np.array([[c, -s, 0.0],
                  [s, c, 0.0],
                  [0.0, 0.0, 1.0]])

    pos = R.T@(np.array([odom_point.x, odom_point.y, odom_point.z]) - np.array([pos_baselink.x, pos_baselink.y, pos_baselink.z]))

    pos = Coordinate(float(pos[0]), float(pos[1]), float(pos[2]))
    return pos




def pixel_to_kinect(u: int, v: int, z: float) -> Coordinate:
    """
    Calculate 3D coordinates from pixel and depth values.
    """
    y = z * (v - CV) / F
    x = z * (u - CU) / F
    return Coordinate(x, y, z)/1000.0 # Convert from mm to m