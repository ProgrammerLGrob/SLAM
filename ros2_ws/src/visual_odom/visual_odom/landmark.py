from math import atan2
import numpy as np

F = 526.61
CU = 318.525
CV = 241.181



class Landmark:
    """Landmark class for storing features detected in images."""
    
    def __init__(self, u: int, v: int, z: int, des: np.ndarray, age: int = 0, color: int = 0) -> None:
        """
        Initialize a Landmark.
        """
        self.u = u
        self.v = v
        self.z = z
        self.des = des
        self.age = age
        self.color = color
        self.world_coordinates = np.zeros((3, 1), dtype=int)
        self.kinect_coordinates = self.calculate_coordinate(u, v, z)
    
    def calculate_coordinate(self, u: int, v: int, z: float) -> np.ndarray:
        """
        Calculate 3D coordinates from pixel and depth values.
        """
        y = z * (v - CV) / F
        x = z * (u - CU) / F
        return np.array([x, y, z])/1000.0

    


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
    if (abs(theta) > 20*np.pi/180):
        theta = 0
    
    
    R = np.array([[ np.cos(theta), -np.sin(theta)],
                  [ np.sin(theta),  np.cos(theta)]])
    
    t = m_P - R @ m_Q

    return R, t, theta