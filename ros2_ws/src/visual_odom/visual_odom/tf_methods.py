
import rclpy
from visual_odom.constants import *
from scipy.spatial.transform import Rotation
from numpy.typing import NDArray
from rclpy.time import Time

import numpy as np
from visual_odom.landmark import Coordinate



def kinect_depth_to_odom(tf_buffer, kinect_point: Coordinate) -> Coordinate:
    """
    Transform a point from the Kinect frame into the odom frame using the LATEST available TF.
    Ignores timestamp and uses the most recent transformation to avoid extrapolation errors.
    """
    # Kinect coordinates are stored in millimeters; TF uses meters.
    point_vector = np.array([
        kinect_point.x,
        kinect_point.y,
        kinect_point.z,
    ], dtype=float) / 1000.0

    try:
        transform = tf_buffer.lookup_transform(
            VISUAL_ODOM_FRAME_ID,
            KINECT_FRAME_ID,
            Time(),  # Use latest available transform
        )
    except Exception as e:
        rclpy.logging.get_logger("tf_methods").error(f"Error in base_link_to_kinect_depth: {e}")
        return None

    translation = transform.transform.translation
    rotation = transform.transform.rotation
    rotation_matrix = Rotation.from_quat([
        rotation.x,
        rotation.y,
        rotation.z,
        rotation.w,
    ])

    transformed_point = rotation_matrix.apply(point_vector) + np.array([
        translation.x,
        translation.y,
        translation.z,
    ])

    return Coordinate(
        float(transformed_point[0]),
        float(transformed_point[1]),
        float(transformed_point[2]),
    )

def base_link_to_kinect_depth(tf_buffer, point: Coordinate | NDArray) -> Coordinate:
    """
    Transform a point from the base_link frame into the Kinect frame using the latest available TF.
    Ignores timestamp and uses the most recent transformation to avoid extrapolation errors.
    """
    if isinstance(point, Coordinate):
        point_vector = np.array([point.x, point.y, point.z], dtype=float)
    else:
        point_vector = np.array(point, dtype=float)

    try:
        transform = tf_buffer.lookup_transform(
            KINECT_FRAME_ID,
            BASE_LINK_FRAME_ID,
            Time(),  # Use latest available transform
        )
    except Exception as e:
        rclpy.logging.get_logger("tf_methods").error(f"Error in base_link_to_kinect_depth: {e}")
        return None

    translation = transform.transform.translation
    rotation = transform.transform.rotation
    rotation_matrix = Rotation.from_quat([
        rotation.x,
        rotation.y,
        rotation.z,
        rotation.w,
    ])

    transformed_point = rotation_matrix.apply(point_vector) + np.array([
        translation.x,
        translation.y,
        translation.z,
    ])

    # Return Kinect coordinates in millimeters.
    transformed_point *= 1000.0

    return Coordinate(
        float(transformed_point[0]),
        float(transformed_point[1]),
        float(transformed_point[2]),
    )

def odom_to_kinect_depth(tf_buffer, point: Coordinate | NDArray) -> Coordinate:
    """
    Transform a point from the odom frame into the Kinect frame using the latest available TF.
    Ignores timestamp and uses the most recent transformation to avoid extrapolation errors.
    """
    if isinstance(point, Coordinate):
        point_vector = np.array([point.x, point.y, point.z], dtype=float)
    else:
        point_vector = np.array(point, dtype=float)

    try:
        transform = tf_buffer.lookup_transform(
            KINECT_FRAME_ID,
            VISUAL_ODOM_FRAME_ID,
            Time(),  # Use latest available transform
        )
    except Exception as e:
        rclpy.logging.get_logger("tf_methods").error(f"Error in odom_to_kinect_depth: {e}")
        return None

    translation = transform.transform.translation
    rotation = transform.transform.rotation
    rotation_matrix = Rotation.from_quat([
        rotation.x,
        rotation.y,
        rotation.z,
        rotation.w,
    ])

    transformed_point = rotation_matrix.apply(point_vector) + np.array([
        translation.x,
        translation.y,
        translation.z,
    ])

    # Return Kinect coordinates in millimeters.
    transformed_point *= 1000.0

    return Coordinate(
        float(transformed_point[0]),
        float(transformed_point[1]),
        float(transformed_point[2]),
    )

def kinect_depth_to_base_link(tf_buffer, kinect_point: Coordinate | NDArray) -> Coordinate:
    """
    Transform a point from the Kinect frame into the base_link frame using the latest available TF.
    Ignores timestamp and uses the most recent transformation to avoid extrapolation errors.
    
    Input Kinect coordinates are expected in millimeters.
    Returns base_link coordinates in meters.
    """
    # Convert Kinect coordinates from millimeters to meters for TF
    if isinstance(kinect_point, Coordinate):
        point_vector = np.array([
            kinect_point.x,
            kinect_point.y,
            kinect_point.z,
        ], dtype=float) / 1000.0
    else:
        point_vector = np.array(kinect_point, dtype=float) / 1000.0

    try:
        transform = tf_buffer.lookup_transform(
            BASE_LINK_FRAME_ID,
            KINECT_FRAME_ID,
            Time(),  # Use latest available transform
        )
    except Exception as e:
        rclpy.logging.get_logger("tf_methods").error(f"Error in kinect_depth_to_base_link: {e}")
        return None

    translation = transform.transform.translation
    rotation = transform.transform.rotation
    rotation_matrix = Rotation.from_quat([
        rotation.x,
        rotation.y,
        rotation.z,
        rotation.w,
    ])

    transformed_point = rotation_matrix.apply(point_vector) + np.array([
        translation.x,
        translation.y,
        translation.z,
    ])

    # Return base_link coordinates in meters.
    return Coordinate(
        float(transformed_point[0]),
        float(transformed_point[1]),
        float(transformed_point[2]),
    )