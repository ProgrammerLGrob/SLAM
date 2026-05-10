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

def normalize_angle(angle: float) -> float:
    while abs(angle) > pi:
        if angle > pi:
            angle -= 2*pi
        elif angle < -pi:
            angle += 2*pi

    return angle    


KINECT_FRAME_ID = "kinect_depth"
BASE_LINK_FRAME_ID = "base_link"
VISUAL_ODOM_FRAME_ID = "odom_visual"


POINTCLOUD_FRAME_ID = "points_3d"
KEYPOINT_POINTCLOUD_FRAME_ID = "keypoint_3d"

VISUAL_ODOM_MSG_TOPIC = "/serf01/odometry/project_slam"

RANSAC_EVALUATION_TOLERANCE = 0.045 #in m
RANSAC_ITERATION = 300
RANSAC_SAMPLE_SIZE = 3
RGB_DEPTH_SYNC_TOLERANCE_SEC = 0.05

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

