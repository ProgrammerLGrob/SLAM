#!/usr/bin/env python3
import numpy as np
from math import pi, atan2


from dataclasses import dataclass

@dataclass
class Coordinate:
    x: float
    y: float
    z: float

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
    u: float
    v: float
    z: float

@dataclass 
class State:
    x: float
    y: float
    theta: float

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
    min_depth: int 
    max_depth: int
    ransac_evaluation_tolerance: float
    ransac_iteration: int
    ransac_sample_size: int
    rgb_depth_sync_tolerance_sec: float
    pixel_tolerance: int
    matches_for_new_landmarks: int
    min_matches_for_ransac: int
    max_rotation_angle_deg: float
    max_landmark_age: float
    ransac_min_inlier_ratio: float
    n_robot_samples: int
    topic_visual_odometry_msg: str
    



def normalize_angle(angle: float) -> float:
    while abs(angle) > pi:
        if angle > pi:
            angle -= 2*pi
        elif angle < -pi:
            angle += 2*pi

    return angle    


KINECT_FRAME_ID = "kinect_depth"
BASE_LINK_FRAME_ID = "base_link"
VISUAL_ODOM_FRAME_ID = "odom"


POINTCLOUD_FRAME_ID = "points_3d"
KEYPOINT_POINTCLOUD_FRAME_ID = "keypoint_3d"

VISUAL_ODOM_MSG_TOPIC = "/serf01/odometry/project_slam"

N_ROBOT_SAMPLES = 1000

RANSAC_EVALUATION_TOLERANCE = 0.045 #in m
RANSAC_ITERATION = 300
RANSAC_SAMPLE_SIZE = 3
MIN_MATCHES_FOR_RANSAC = 15
RGB_DEPTH_SYNC_TOLERANCE_SEC = 0.05
MAX_ROTATION_ANGLE_DEG = 15.0

MAX_LANDMARK_AGE = 15.0 

RANSAC_MIN_INLIER_RATIO = 0.3

PIXEL_TOLERANCE = 8
MATCHES_FOR_NEW_LANDMARKS = 50

F = 526.61
CU = 318.525
CV = 241.181


MIN_DEPTH = 400
MAX_DEPTH = 5000

MAX_U = 640
MAX_V = 480

MAX_AZIMUTH = abs(atan2(CU, F))
MAX_ALTITUDE = abs(atan2(CV, F))
CAMERA_POS_IN_BASELINK = Coordinate(0.1, 0.0, 0.75)

ROT_KB = np.array([[0.0, 0.0, 1.0],
                     [-1.0, 0.0, 0.0],
                     [0.0, -1.0, 0.0]])

#measruement noice/error
SIGMA_PIXEL = 0.8/3 # in u, v direction, in pixels; 0,086°
ERROR_MIN_DEPTH = 0.001477 # konstanter Offset (Rauschen bei minimalem Abstand)
ERROR_QUADRATIC_DEPTH = 0.002294 # quadratischer Koeffizient (fitted)

#SIGMA POINT APPROXIMATION
# --- ERROR X ---
ERR_FUNC_X_CONST = -6.529873e+00 
ERR_FUNC_X_D_1 =   1.787
ERR_FUNC_X_T_1 =  -9.668554e+00
ERR_FUNC_X_D_2 =  -6.820445e-02
ERR_FUNC_X_D1_T1 = 1.888666e+00
ERR_FUNC_X_T_2 =  -3.876097e+00
ERR_FUNC_X_D_3 =  -2.427950e-03
ERR_FUNC_X_D2_T1 = -4.202577e-02
ERR_FUNC_X_D1_T2 =  5.908962e-01
ERR_FUNC_X_T_3 =  -2.417847e-01


# --- ERROR Y ---
ERR_FUNC_Y_CONST =  6.435185e+00 
ERR_FUNC_Y_D_1 =   -1.819108e+00
ERR_FUNC_Y_T_1 =    8.910108e+00
ERR_FUNC_Y_D_2 =    1.127993e-01
ERR_FUNC_Y_D1_T1 = -1.965693e+00
ERR_FUNC_Y_T_2 =    3.570893e+00
ERR_FUNC_Y_D_3 =    3.033334e-03
ERR_FUNC_Y_D2_T1 =  8.519163e-02
ERR_FUNC_Y_D1_T2 = -4.980875e-01
ERR_FUNC_Y_T_3 =    3.085193e-01


# --- ERROR THETA ---
ERR_FUNC_THETA_CONST =  7.607340e+00 
ERR_FUNC_THETA_D_1 =   -1.974354e+00
ERR_FUNC_THETA_T_1 =    8.981931e+00
ERR_FUNC_THETA_D_2 =    1.057925e-01
ERR_FUNC_THETA_D1_T1 = -1.908562e+00
ERR_FUNC_THETA_T_2 =    2.821687e+00
ERR_FUNC_THETA_D_3 =    3.266059e-03
ERR_FUNC_THETA_D2_T1 =  7.723625e-02
ERR_FUNC_THETA_D1_T2 = -4.240273e-01
ERR_FUNC_THETA_T_3 =    6.209562e-02

 