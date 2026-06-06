#!/usr/bin/env python3
import numpy as np
from math import pi, atan2
from dataclasses import dataclass

@dataclass
class Coordinate:
    """
    A 3D spatial coordinate container.

    Typically used for positions in the 'base_link', 'kinect_depth', or 'odom' frames.

    Attributes:
        x (float): X-coordinate component in meters.
        y (float): Y-coordinate component in meters.
        z (float): Z-coordinate component in meters.
    """
    x: float
    """X-coordinate component in meters."""
    y: float
    """Y-coordinate component in meters."""
    z: float
    """Z-coordinate component in meters."""

    def __add__(self, other: "Coordinate") -> "Coordinate":
        if not isinstance(other, Coordinate):
            return NotImplemented
        return Coordinate(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Coordinate") -> "Coordinate":
        if not isinstance(other, Coordinate):
            return NotImplemented
        return Coordinate(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, factor) -> "Coordinate":
        if isinstance(factor, (int, float)):
            return Coordinate(self.x * factor, self.y * factor, self.z * factor)
        return NotImplemented

    def __truediv__(self, factor: float) -> "Coordinate":
        return Coordinate(self.x / factor, self.y / factor, self.z / factor)
    

@dataclass
class PixelCoordinate:
    """
    Container for a 2D image pixel combined with its registered depth value.

    Attributes:
        u (float): Horizontal pixel column index in the image plane.
        v (float): Vertical pixel row index in the image plane.
        z (float): Raw depth value from the sensor corresponding to this pixel.
    """
    u: float
    """Horizontal pixel column index in the image plane."""
    v: float
    """Vertical pixel row index in the image plane."""
    z: float
    """Raw depth value from the sensor corresponding to this pixel."""

@dataclass 
class State:
    """
    2D Robot pose representation containing planar translation and heading.

    Attributes:
        x (float): Global X-position of the robot base in meters.
        y (float): Global Y-position of the robot base in meters.
        theta (float): Heading/yaw orientation of the robot in radians [-pi, pi].
    """
    x: float
    """Global X-position of the robot base in meters."""
    y: float
    """Global Y-position of the robot base in meters."""
    theta: float
    """Heading/yaw orientation of the robot in radians [-pi, pi]."""

    def __add__(self, other: "State") -> "State":
        if not isinstance(other, State):
            return NotImplemented
        return State(self.x + other.x, self.y + other.y, self.theta + other.theta)

    def __sub__(self, other: "State") -> "State":
        if not isinstance(other, State):
            return NotImplemented
        return State(self.x - other.x, self.y - other.y, self.theta - other.theta)

    def __mul__(self, factor) -> "State":
        if isinstance(factor, (int, float)):
            return State(self.x * factor, self.y * factor, self.theta * factor)
        return NotImplemented

    def __truediv__(self, factor: float) -> "State":
        return State(self.x / factor, self.y / factor, self.theta / factor)
    

@dataclass
class Parameters:
    """
    Runtime configuration and threshold parameter package for the SLAM node.

    Attributes:
        min_depth (int): Minimum reliable sensor range threshold.
        max_depth (int): Maximum reliable sensor range threshold.
        ransac_evaluation_tolerance (float): Inlier distance threshold for RANSAC outlier rejection in meters.
        ransac_iteration (int): Maximum number of optimization loops for the RANSAC routine.
        ransac_sample_size (int): Minimum data subset size used to compute a transform hypothesis.
        rgb_depth_sync_tolerance_sec (float): Maximum allowed time delta between synchronized RGB and Depth frames.
        pixel_tolerance (int): Pixel window distance threshold for visual matching checks.
        matches_for_new_landmarks (int): Required match counter threshold before declaring a stable new map landmark.
        min_matches_for_ransac (int): Minimum required feature matches to attempt a RANSAC visual odometry calculation.
        max_rotation_angle_deg (float): Safety constraint capping the maximum plausible robot rotation per frame step.
        max_landmark_age (float): Lifespan limit before an unobserved landmark is pruned from the map memory.
        ransac_min_inlier_ratio (float): Minimum acceptable percentage of inliers needed to trust a RANSAC solution.
        n_robot_samples (int): Number of particles/samples generated within the localization filter.
        topic_visual_odometry_msg (str): ROS 2 topic name where the calculated odometry path is published.
    """
    min_depth: int
    """Minimum reliable sensor range threshold."""
    max_depth: int
    """Maximum reliable sensor range threshold."""
    ransac_evaluation_tolerance: float
    """Inlier distance threshold for RANSAC outlier rejection in meters."""
    ransac_iteration: int
    """Maximum number of optimization loops for the RANSAC routine."""
    ransac_sample_size: int
    """Minimum data subset size used to compute a transform hypothesis."""
    rgb_depth_sync_tolerance_sec: float
    """Maximum allowed time delta between synchronized RGB and Depth frames."""
    pixel_tolerance: int
    """Pixel window distance threshold for visual matching checks."""
    matches_for_new_landmarks: int
    """Required match counter threshold before declaring a stable new map landmark."""
    min_matches_for_ransac: int
    """Minimum required feature matches to attempt a RANSAC visual odometry calculation."""
    max_rotation_angle_deg: float
    """Safety constraint capping the maximum plausible robot rotation per frame step."""
    max_landmark_age: float
    """Lifespan limit before an unobserved landmark is pruned from the map memory."""
    ransac_min_inlier_ratio: float
    """Minimum acceptable percentage of inliers needed to trust a RANSAC solution."""
    n_robot_samples: int
    """Number of particles/samples generated within the localization filter."""
    topic_visual_odometry_msg: str
    """ROS 2 topic name where the calculated odometry path is published."""
    

def normalize_angle(angle: float) -> float:
    """
    Normalizes a given angle into the standard bounding range of [-pi, pi].

    Args:
        angle (float): Input angle in radians.

    Returns:
        float: Normalized angle in radians.
    """
    return (angle + pi) % (2 * pi) - pi    


KINECT_FRAME_ID = "kinect_depth"
"""TF frame identifier for the optical center of the RGB-D camera."""

BASE_LINK_FRAME_ID = "base_link"
"""TF frame identifier for the physical center/drehpunkt of the robot platform."""

VISUAL_ODOM_FRAME_ID = "odom"
"""TF frame identifier for the fixed global odometry origin."""


POINTCLOUD_FRAME_ID = "points_3d"
"""Topic/Frame ID mapping the structural environment point cloud."""

KEYPOINT_POINTCLOUD_FRAME_ID = "keypoint_3d"
"""Topic/Frame ID isolating the tracked ORB descriptor 3D keypoints."""

VISUAL_ODOM_MSG_TOPIC = "/serf01/odometry/project_slam"
"""Primary ROS 2 publisher output path for navigation odometry."""

N_ROBOT_SAMPLES = 1000
"""Default sample size/particle count allocated for the localization state space."""

RANSAC_EVALUATION_TOLERANCE = 0.045
"""Distance margin (4.5 centimeters) defining whether a point validates a RANSAC hypothesis."""

RANSAC_ITERATION = 300
"""Loop iteration budget assigned to find the optimal geometric transform."""

RANSAC_SAMPLE_SIZE = 3
"""Absolute count of matched 3D point pairs used to calculate the 6DOF alignment."""

MIN_MATCHES_FOR_RANSAC = 15
"""Hard lower bound of features required to initiate visual tracking computations."""

RGB_DEPTH_SYNC_TOLERANCE_SEC = 0.05
"""Maximum temporal alignment error allowed between image pairs (50ms)."""

MAX_ROTATION_ANGLE_DEG = 15.0
"""Safety constraint capping the maximum plausible robot rotation per frame step."""

MAX_LANDMARK_AGE = 15.0
"""Forgetting-factor threshold; old landmarks unseen for 15 frames face map deletion."""

RANSAC_MIN_INLIER_RATIO = 0.3
"""Filter constraint requiring at least 30% of matched pairs to fit the final consensus."""

PIXEL_TOLERANCE = 8
"""Spatial search bounding radius in image space for match validations."""

MATCHES_FOR_NEW_LANDMARKS = 50
"""Minimal tracking history count needed to stabilize mapping features."""

MIN_DET_VALUE = 1e-12
"""Small epsilon cutoff used to shield matrix calculations from determinant singularities."""

F = 526.61
"""Horizontal and vertical focal length parameter of the Kinect optical model."""

CU = 318.525
"""Horizontal pixel coordinate of the camera principal point (optical center)."""

CV = 241.181
"""Vertical pixel coordinate of the camera principal point (optical center)."""


MIN_DEPTH = 400
"""Hard hardware sensor limit ignoring range readings closer than 40 centimeters."""

MAX_DEPTH = 5000
"""Hard hardware sensor limit dropping range readings farther than 5.0 meters."""

MAX_U = 640
"""Image plane horizontal pixel boundary layout dimension."""

MAX_V = 480
"""Image plane vertical pixel boundary layout dimension."""

MAX_AZIMUTH = abs(atan2(CU, F))
"""Derived horizontal angle limit bound matching the sensor's physical Field of View."""

MAX_ALTITUDE = abs(atan2(CV, F))
"""Derived vertical angle limit bound matching the sensor's physical Field of View."""

CAMERA_POS_IN_BASELINK = Coordinate(0.1, 0.0, 0.75)
"""Fixed physical offset of the camera frame relative to base_link (10cm forward, 75cm high)."""

ROT_BK = np.array([[0.0, 0.0, 1.0],
                   [-1.0, 0.0, 0.0],
                   [0.0, -1.0, 0.0]])
"""
Coordinate system transform matrix mapping the Kinect optical frame conventions to standard ROS base_link.

Swaps axes such that Z-Optical points along X-BaseLink (forward) and X-Optical maps to -Y-BaseLink (right).
"""

# measurement noise/error
SIGMA_PIXEL = 0.8/3
"""Pixel projection variance noise model coefficient along the image plane coordinates."""

ERROR_MIN_DEPTH = 0.001477
"""Base baseline measurement noise offset constant at closest sensor range in meters."""

ERROR_QUADRATIC_DEPTH = 0.002294
"""Quadratic sensor regression factor modeling how RGB-D precision degrades across distance."""

# SIGMA POINT APPROXIMATION
# --- ERROR X ---
ERR_FUNC_X_CONST = -6.529873e+00
"""Intercept baseline term for the fitted state-propagation error expansion model in X."""

ERR_FUNC_X_D_1 = 1.787
"""First-order depth dependency scale component for tracking noise in X."""

ERR_FUNC_X_T_1 = -9.668554e+00
"""First-order heading dependency scale component for tracking noise in X."""

ERR_FUNC_X_D_2 = -6.820445e-02
"""Second-order quadratic depth scaling component for tracking noise in X."""

ERR_FUNC_X_D1_T1 = 1.888666e+00
"""Coupled depth-heading linear cross-term modifier for tracking noise in X."""

ERR_FUNC_X_T_2 = -3.876097e+00
"""Second-order quadratic heading scaling component for tracking noise in X."""

ERR_FUNC_X_D_3 = -2.427950e-03
"""Third-order cubic depth scaling coefficient for tracking noise in X."""

ERR_FUNC_X_D2_T1 = -4.202577e-02
"""Coupled quadratic-depth linear-heading modifier for tracking noise in X."""

ERR_FUNC_X_D1_T2 = 5.908962e-01
"""Coupled linear-depth quadratic-heading modifier for tracking noise in X."""

ERR_FUNC_X_T_3 = -2.417847e-01
"""Third-order cubic heading scaling coefficient for tracking noise in X."""


# --- ERROR Y ---
ERR_FUNC_Y_CONST = 6.435185e+00
"""Intercept baseline term for the fitted state-propagation error expansion model in Y."""

ERR_FUNC_Y_D_1 = -1.819108e+00
"""First-order depth dependency scale component for tracking noise in Y."""

ERR_FUNC_Y_T_1 = 8.910108e+00
"""First-order heading dependency scale component for tracking noise in Y."""

ERR_FUNC_Y_D_2 = 1.127993e-01
"""Second-order quadratic depth scaling component for tracking noise in Y."""

ERR_FUNC_Y_D1_T1 = -1.965693e+00
"""Coupled depth-heading linear cross-term modifier for tracking noise in Y."""

ERR_FUNC_Y_T_2 = 3.570893e+00
"""Second-order quadratic heading scaling component for tracking noise in Y."""

ERR_FUNC_Y_D_3 = 3.033334e-03
"""Third-order cubic depth scaling coefficient for tracking noise in Y."""

ERR_FUNC_Y_D2_T1 = 8.519163e-02
"""Coupled quadratic-depth linear-heading modifier for tracking noise in Y."""

ERR_FUNC_Y_D1_T2 = -4.980875e-01
"""Coupled linear-depth quadratic-heading modifier for tracking noise in Y."""

ERR_FUNC_Y_T_3 = 3.085193e-01
"""Third-order cubic heading scaling coefficient for tracking noise in Y."""


# --- ERROR THETA ---
ERR_FUNC_THETA_CONST = 7.607340e+00
"""Intercept baseline term for the fitted state-propagation error expansion model in Heading."""

ERR_FUNC_THETA_D_1 = -1.974354e+00
"""First-order depth dependency scale component for tracking noise in Heading."""

ERR_FUNC_THETA_T_1 = 8.981931e+00
"""First-order heading dependency scale component for tracking noise in Heading."""

ERR_FUNC_THETA_D_2 = 1.057925e-01
"""Second-order quadratic depth scaling component for tracking noise in Heading."""

ERR_FUNC_THETA_D1_T1 = -1.908562e+00
"""Coupled depth-heading linear cross-term modifier for tracking noise in Heading."""

ERR_FUNC_THETA_T_2 = 2.821687e+00
"""Second-order quadratic heading scaling component for tracking noise in Heading."""

ERR_FUNC_THETA_D_3 = 3.266059e-03
"""Third-order cubic depth scaling coefficient for tracking noise in Heading."""

ERR_FUNC_THETA_D2_T1 = 7.723625e-02
"""Coupled quadratic-depth linear-heading modifier for tracking noise in Heading."""

ERR_FUNC_THETA_D1_T2 = -4.240273e-01
"""Coupled linear-depth quadratic-heading modifier for tracking noise in Heading."""

ERR_FUNC_THETA_T_3 = 6.209562e-02
"""Third-order cubic heading scaling coefficient for tracking noise in Heading."""