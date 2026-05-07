from math import pi, atan2

KINECT_FRAME_ID = "kinect_depth"
BASE_LINK_FRAME_ID = "base_link"
VISUAL_ODOM_FRAME_ID = "odom_visual"


POINTCLOUD_FRAME_ID = "points_3d"
KEYPOINT_POINTCLOUD_FRAME_ID = "keypoint_3d"

VISUAL_ODOM_MSG_TOPIC = "/serf01/odometry/project_slam"

DESCRIPTOR_TOLERANCE = 50
RANSAC_EVALUATION_TOLERANCE = 0.045 #in m
RANSAC_ITERATION = 1000
RANSAC_SAMPLE_SIZE = 3
RGB_DEPTH_SYNC_TOLERANCE_SEC = 0.05

WAITING_FRAMES = 1

F = 526.61
CU = 318.525
CV = 241.181


MIN_DEPTH = 400
MAX_DEPTH = 5000

MAX_U = 640
MAX_V = 480

MAX_AZIMUTH = abs(atan2(CU, F))
MAX_ALTITUDE = abs(atan2(CV, F))

def normalize_angle(angle: float) -> float:
    while abs(angle) > pi:
        if angle > pi:
            angle -= 2*pi
        elif angle < -pi:
            angle += 2*pi

    return angle    