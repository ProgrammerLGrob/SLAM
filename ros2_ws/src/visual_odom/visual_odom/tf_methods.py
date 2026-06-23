#!/usr/bin/env python3

"""!
@file tf_methods.py
@package visual_odom.tf_methods
@brief Stateless coordinate frame transformation utilities.
"""

from math import sin, cos
from visual_odom.constants import (
    Coordinate, PixelCoordinate, CV, F, CU, ROT_BK, CAMERA_POS_IN_BASELINK
)
import numpy as np
from numpy.typing import NDArray


def kinect_depth_to_odom(kinect_point: Coordinate, theta: float, pos_baselink: Coordinate) -> Coordinate:
    """!
    @brief Transform a point from the Kinect frame into the global odom frame using the LATEST available robot state.
    
    Ignores historical timestamps and uses the most recent transformation matrices to avoid extrapolation errors.

    @param kinect_point Coordinate object in the kinect_depth frame.
    @param theta Current yaw heading orientation angle of the robot base in radians.
    @param pos_baselink Global tracking position coordinate of the robot base link center.
    @return Coordinate object mapped completely into the global odom map frame.
    """
    c = cos(theta)
    s = sin(theta)
    R = np.array([[c, -s, 0.0],
                  [s, c, 0.0],
                  [0.0, 0.0, 1.0]])

    pos_kb = kinect_depth_to_baselink(kinect_point)
    pos_odom = R @ np.array([pos_kb.x, pos_kb.y, pos_kb.z])

    pos = pos_baselink + Coordinate(float(pos_odom[0]), float(pos_odom[1]), float(pos_odom[2]))
    return pos


def kinect_depth_to_baselink(kinect_point: Coordinate) -> Coordinate:
    """!
    @brief Transform a point from the Kinect optical frame into the robot base_link frame using static calibrations.

    Applies the static calibration matrix and the mounting offset translations relative to the base center.

    @param kinect_point Coordinate object in the kinect_depth optical frame.
    @return Coordinate object transformed into the local robot base_link frame.
    """
    pos = ROT_BK @ np.array([kinect_point.x, kinect_point.y, kinect_point.z])
    pos_baselink = CAMERA_POS_IN_BASELINK + Coordinate(float(pos[0]), float(pos[1]), float(pos[2]))
    
    return pos_baselink


def odom_to_baselink(odom_point: Coordinate, theta: float, pos_baselink: Coordinate) -> Coordinate:
    """!
    @brief Transform a point from the global odom frame back into the local base_link frame.

    Applies the inverse heading rotation and structural translations relative to the current robot base pose.

    @param odom_point Coordinate object in the global odom map frame.
    @param theta Current heading orientation angle of the robot base in radians.
    @param pos_baselink Global tracking position coordinate of the robot base link center.
    @return Coordinate object referenced in the local robot base_link frame.
    """
    c = cos(theta)
    s = sin(theta)
    R = np.array([[c, -s, 0.0],
                  [s, c, 0.0],
                  [0.0, 0.0, 1.0]])

    pos = R.T @ (np.array([odom_point.x, odom_point.y, odom_point.z]) - np.array([pos_baselink.x, pos_baselink.y, pos_baselink.z]))
    pos_coordinate = Coordinate(float(pos[0]), float(pos[1]), float(pos[2]))
    return pos_coordinate


def pixel_to_kinect(kp: PixelCoordinate) -> Coordinate:
    """!
    @brief Calculate 3D coordinates from pixel indices and its registered depth values.

    Applies the pinhole model and scales the output values from millimeters to meters.

    @param kp PixelCoordinate object containing pixel indices (u, v) and depth z (in millimeters).
    @return Coordinate object in the Kinect optical frame (meters).
    """
    y = kp.z * (kp.v - CV) / F
    x = kp.z * (kp.u - CU) / F
    return Coordinate(x, y, kp.z) / 1000.0  # Convert from mm to m


def pixel_to_kinect_(u: int, v: int, z: float) -> NDArray:
    """!
    @brief Calculate 3D coordinates from raw pixel indices and depth values directly to a NumPy representation.

    Applies the pinhole projection equations and returns a standardized numpy vector array scaled in meters.

    @param u Horizontal pixel column index in the image plane.
    @param v Vertical pixel row index in the image plane.
    @param z Raw depth value from the sensor (in millimeters).
    @return 3D NumPy array vector [x, y, z] in the Kinect frame (meters).
    """
    z_m = z / 1000.0  # Convert from mm to m
    y = z_m * (v - CV) / F
    x = z_m * (u - CU) / F
    return np.array([x, y, z_m])