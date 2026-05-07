from math import atan2
import numpy as np
import cv2



from visual_odom.constants import *
from scipy.spatial.transform import Rotation
from numpy.typing import NDArray
from rclpy.time import Time
import numpy as np



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

    def __mul__(self, factor: float) -> "Coordinate":
        return Coordinate(self.x * factor, self.y * factor, self.z * factor)

    def __rmul__(self, factor: float) -> "Coordinate":
        return self.__mul__(factor)

    def __truediv__(self, factor: float) -> "Coordinate":
        return Coordinate(self.x / factor, self.y / factor, self.z / factor)


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
        coor = self.calculate_coordinate(u, v, z) # in mm
        self.kinect_coordinates = Coordinate(coor[0], coor[1], coor[2]) # in mm
        self.odom_coordinates = odom_coordinates

    def calculate_coordinate(self, u: int, v: int, z: float) -> np.ndarray:
        """
        Calculate 3D coordinates in kinect frame from pixel and depth values.
        """
        x = z * (u - CU) / F
        y = z * (v - CV) / F
        z = z
        return np.array([x, y, z])
    
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

