#!/usr/bin/env python3

"""!
@package visual_odom.constants
@brief Global configuration, camera model, ROS topic definitions and helper datatypes used by the visual odometry system.

This module centralizes all runtime constants, camera calibration parameters,
ROS topic names, TF frame identifiers and helper dataclasses required by the
visual odometry pipeline.
"""

import numpy as np
from math import pi, atan2
from dataclasses import dataclass


# ==============================================================================
# Dataclasses
# ==============================================================================

@dataclass(slots=True)
class Coordinate:
    """!
    @brief 3D spatial coordinate container.

    Typically used for positions in the @c base_link, @c kinect_depth, or @c odom
    coordinate frames.
    """
    x: float  ##< X-coordinate component in meters.
    y: float  ##< Y-coordinate component in meters.
    z: float  ##< Z-coordinate component in meters.

    def __add__(self, other: "Coordinate") -> "Coordinate":
        """!
        @brief Add two Coordinate objects component-wise.

        @param other Coordinate object to add.
        @return New Coordinate containing the component-wise sum.
        """
        if not isinstance(other, Coordinate):
            return NotImplemented
        return Coordinate(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Coordinate") -> "Coordinate":
        """!
        @brief Subtract two Coordinate objects component-wise.

        @param other Coordinate object to subtract.
        @return New Coordinate containing the component-wise difference.
        """
        if not isinstance(other, Coordinate):
            return NotImplemented
        return Coordinate(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, factor) -> "Coordinate":
        """!
        @brief Multiply a Coordinate object by a scalar value.

        @param factor Scalar multiplication factor.
        @return New Coordinate with scaled components.
        """
        if isinstance(factor, (int, float)):
            return Coordinate(self.x * factor, self.y * factor, self.z * factor)
        return NotImplemented

    def __truediv__(self, factor: float) -> "Coordinate":
        """!
        @brief Divide a Coordinate object by a scalar value.

        @param factor Scalar divisor.
        @return New Coordinate with divided components.
        """
        return Coordinate(self.x / factor, self.y / factor, self.z / factor)


@dataclass(slots=True)
class PixelCoordinate:
    """!
    @brief Container for a 2D image pixel and its depth value.

    Stores a pixel coordinate in the image plane together with the corresponding
    registered raw depth value.
    """
    u: float  ##< Horizontal pixel column index in the image plane.
    v: float  ##< Vertical pixel row index in the image plane.
    z: float  ##< Raw depth value from the sensor corresponding to this pixel.


@dataclass(slots=True)
class State:
    """!
    @brief 2D robot pose representation.

    Represents planar robot translation and heading.
    """
    x: float      ##< Global X-position of the robot base in meters.
    y: float      ##< Global Y-position of the robot base in meters.
    theta: float  ##< Heading/yaw orientation of the robot in radians.

    def __add__(self, other: "State") -> "State":
        """!
        @brief Add two State objects component-wise.

        @param other State object to add.
        @return New State containing the component-wise sum.
        """
        if not isinstance(other, State):
            return NotImplemented
        return State(self.x + other.x, self.y + other.y, self.theta + other.theta)

    def __sub__(self, other: "State") -> "State":
        """!
        @brief Subtract two State objects component-wise.

        @param other State object to subtract.
        @return New State containing the component-wise difference.
        """
        if not isinstance(other, State):
            return NotImplemented
        return State(self.x - other.x, self.y - other.y, self.theta - other.theta)

    def __mul__(self, factor) -> "State":
        """!
        @brief Multiply a State object by a scalar value.

        @param factor Scalar multiplication factor.
        @return New State with scaled values.
        """
        if isinstance(factor, (int, float)):
            return State(self.x * factor, self.y * factor, self.theta * factor)
        return NotImplemented

    def __truediv__(self, factor: float) -> "State":
        """!
        @brief Divide a State object by a scalar value.

        @param factor Scalar divisor.
        @return New State with divided values.
        """
        return State(self.x / factor, self.y / factor, self.theta / factor)


@dataclass(slots=True)
class Parameters:
    """!
    @brief Runtime configuration and threshold parameter package for the SLAM node.

    Stores all relevant runtime parameters loaded from ROS 2 parameter declarations.
    """
    max_depth: int                     ##< Maximum reliable sensor range threshold.
    ransac_evaluation_tolerance: float ##< Inlier distance threshold for RANSAC outlier rejection in meters.
    ransac_iteration: int              ##< Maximum number of optimization loops for the RANSAC routine.
    ransac_sample_size: int            ##< Minimum data subset size used to compute a transform hypothesis.
    rgb_depth_sync_tolerance_sec: float##< Maximum allowed time delta between synchronized RGB and depth frames.
    pixel_tolerance: int               ##< Pixel window distance threshold for visual matching checks.
    matches_for_new_landmarks: int     ##< Required match counter threshold before declaring a stable new map landmark.
    min_matches_for_ransac: int        ##< Minimum required feature matches to attempt a RANSAC visual odometry calculation.
    max_rotation_angle_deg: float      ##< Safety constraint capping the maximum plausible robot rotation per frame step.
    min_landmark_trust: float          ##< Minimum trust value before an unobserved landmark is pruned from map memory.
    ransac_min_inlier_ratio: float     ##< Minimum acceptable percentage of inliers needed to trust a RANSAC solution.
    n_robot_samples: int               ##< Number of particles/samples generated within the localization filter.
    topic_visual_odometry_msg: str     ##< ROS 2 topic name where the calculated odometry path is published.


# ==============================================================================
# Helper Functions
# ==============================================================================

def normalize_angle(angle: float) -> float:
    """!
    @brief Normalize an angle to the range [-pi, pi].

    Normalizes a given angle into the standard bounding range of [-pi, pi].

    @param angle Input angle in radians.
    @return Normalized angle in radians.
    """
    return (angle + pi) % (2 * pi) - pi


# ==============================================================================
# ROS Topics
# ==============================================================================

## @brief Primary ROS 2 publisher output topic for navigation odometry.
VISUAL_ODOM_MSG_TOPIC = "/serf01/odometry/project_slam"

## @brief ROS 2 topic for raw RGB image input.
RGB_IMAGE_TOPIC = "/serf01/nav_rgbd_1/rgb/image_raw"

## @brief ROS 2 topic for raw depth image input.
DEPTH_IMAGE_TOPIC = "/serf01/nav_rgbd_1/depth/image_raw"

## @brief ROS 2 topic for wheel odometry input.
WHEEL_ODOMETRY_TOPIC = "/serf01/odometry/wheel"

## @brief ROS 2 topic for filtered odometry.
FILTERED_ODOMETRY_TOPIC = "/serf01/odometry/filtered"

## @brief ROS 2 topic for IMU-based odometry input.
IMU_ODOMETRY_TOPIC = "/serf01/odometry/imu"

## @brief ROS 2 topic for static TF transforms.
TF_STATIC_TOPIC = "/tf_static"

## @brief ROS 2 topic for publishing or visualizing the camera vision cone.
VISION_CONE_TOPIC = "/serf01/camera/vision_cone"

## @brief ROS 2 topic for publishing images with visualized keypoints.
KP_IMAGE_TOPIC = "/serf01/camera/keypoint_image"


# ==============================================================================
# TF Frame IDs
# ==============================================================================

## @brief TF frame identifier for the optical center of the RGB-D camera.
KINECT_FRAME_ID = "kinect_depth"

## @brief TF frame identifier for the physical center of the robot platform.
BASE_LINK_FRAME_ID = "base_link"

## @brief TF frame identifier for the fixed global odometry origin.
VISUAL_ODOM_FRAME_ID = "odom"

## @brief Frame/topic identifier for the structural environment point cloud.
POINTCLOUD_FRAME_ID = "points_3d"

## @brief Frame/topic identifier for tracked ORB descriptor 3D keypoints.
KEYPOINT_POINTCLOUD_FRAME_ID = "keypoint_3d"


# ==============================================================================
# Camera Model
# ==============================================================================

## @brief Horizontal and vertical focal length parameter of the Kinect optical model.
F = 526.61

## @brief Horizontal pixel coordinate of the camera principal point.
CU = 318.525

## @brief Vertical pixel coordinate of the camera principal point.
CV = 241.181

## @brief Horizontal image plane resolution in pixels.
MAX_U = 640

## @brief Vertical image plane resolution in pixels.
MAX_V = 480

## @brief Derived horizontal angle limit matching the sensor field of view.
MAX_AZIMUTH = abs(atan2(CU, F))

## @brief Derived vertical angle limit matching the sensor field of view.
MAX_ALTITUDE = abs(atan2(CV, F))

## @brief Fixed physical camera offset relative to base_link.
CAMERA_POS_IN_BASELINK = Coordinate(0.1, 0.0, 0.75)

## @brief Vertical field of view angle of the RGB-D camera in radians.
CAMERA_ANGLE_VER_RAD = 43 * pi / 180

## @brief Horizontal field of view angle of the RGB-D camera in radians.
CAMERA_ANGLE_HOR_RAD = 57 * pi / 180

## @brief Horizontal discretization resolution of the camera vision cone.
RESOLUTION_CONE_H = 40

## @brief Vertical discretization resolution of the camera vision cone.
RESOLUTION_CONE_V = 30

## @brief Rotation matrix from Kinect optical frame to ROS base_link frame.
#
#  Maps Kinect optical frame conventions to the standard ROS base_link convention.
#  The optical Z-axis is mapped to the base_link X-axis, and the optical X-axis is
#  mapped to the negative base_link Y-axis.
ROT_BK = np.array([[0.0, 0.0, 1.0],
                   [-1.0, 0.0, 0.0],
                   [0.0, -1.0, 0.0]], dtype=np.float64)


# ==============================================================================
# Depth Filter
# ==============================================================================

## @brief Minimum reliable depth value in millimeters.
MIN_DEPTH = 400

## @brief Maximum reliable depth value in millimeters.
MAX_DEPTH = 7000


# ==============================================================================
# RANSAC
# ==============================================================================

## @brief Inlier distance threshold for evaluating RANSAC hypotheses in meters.
RANSAC_EVALUATION_TOLERANCE = 0.045

## @brief Maximum number of RANSAC iterations.
RANSAC_ITERATION = 100

## @brief Number of matched point pairs used for one RANSAC hypothesis.
RANSAC_SAMPLE_SIZE = 3

## @brief Minimum inlier ratio required to accept a RANSAC solution.
RANSAC_MIN_INLIER_RATIO = 0.3

## @brief Minimum number of feature matches required to start RANSAC.
MIN_MATCHES_FOR_RANSAC = 15

## @brief Maximum plausible robot rotation per frame step in degrees.
MAX_ROTATION_ANGLE_DEG = 15.0

## @brief Maximum allowed temporal difference between RGB and depth frames in seconds.
RGB_DEPTH_SYNC_TOLERANCE_SEC = 0.05

## @brief Pixel-space search radius used for visual match validation.
PIXEL_TOLERANCE = 8


# ==============================================================================
# Particle Filter
# ==============================================================================

## @brief Default number of particles/samples used for localization.
N_ROBOT_SAMPLES = 1000

## @brief Noise increment added after a failed RANSAC estimation.
NOISE_INCREMENT_RANSAC_FAILURE = 0.002

## @brief Maximum allowed additional noise caused by repeated RANSAC failures.
MAX_ADDITIONAL_NOISE_RANSAC_FAILURE = 0.22


# ==============================================================================
# Landmark / Map Management
# ==============================================================================

## @brief Minimum tracking count required to stabilize a new landmark.
MATCHES_FOR_NEW_LANDMARKS = 50

## @brief Initial landmark count threshold used for map bootstrapping.
INITIAL_LANDMARK_COUNT = 260

## @brief Minimum trust value before a landmark is considered unreliable.
MIN_LANDMARK_TRUST = 20.0

## @brief Trust value added to a landmark after successful observation.
INCREASE_TRUST_VALUE = 10.0

## @brief Multiplicative trust decay factor for unobserved landmarks.
DECREASE_TRUST_FACTOR = 0.95


# ==============================================================================
# Noise / EKF
# ==============================================================================

## @brief Wheel odometry process noise standard deviation in x-direction.
SIGMA_X_ODOM_WHEEL_Q = 0.02

## @brief Wheel odometry process noise standard deviation in y-direction.
SIGMA_Y_ODOM_WHEEL_Q = 0.02

## @brief Wheel odometry process noise standard deviation for heading angle.
SIGMA_THETA_ODOM_WHEEL_Q = 0.01

## @brief Pixel projection noise coefficient for image-plane coordinates.
SIGMA_PIXEL = 0.8 / 3

## @brief Base depth measurement noise offset at closest sensor range in meters.
ERROR_MIN_DEPTH = 0.001477

## @brief Quadratic depth error factor modeling RGB-D precision degradation.
ERROR_QUADRATIC_DEPTH = 0.002294

## @brief Small epsilon cutoff used to avoid determinant singularities.
MIN_DET_VALUE = 1e-12


# ==============================================================================
# Internal / Runtime State
# ==============================================================================

## @brief Global runtime parameter instance populated at node startup.
parameters: Parameters = None