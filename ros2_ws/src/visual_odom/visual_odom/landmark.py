from dataclasses import dataclass, field
from math import atan2
import numpy as np


@dataclass
class landmark:
    #pixel coordinates
    u: int
    v: int

    #depth value in mm
    z: int
    des: np.ndarray 

    #x,y,z in world coordinates
    coordinates: np.ndarray = field(default_factory=lambda: np.zeros((3, 1), dtype=int))


"""
@param P_i: 2D points in the first frame
@param Q_i: 2D points in the second frame
"""
def kabash(P_i: np.ndarray, Q_i: np.ndarray):

    
    print("P_i: ", P_i)
    print("Q_i: ", Q_i)

    m_P = np.mean(P_i,axis=0)
    m_Q = np.mean(Q_i,axis=0)

    print("m_P: ", m_P)
    print("m_Q: ", m_Q)

    P_i_centered = P_i - m_P
    Q_i_centered = Q_i - m_Q   

    print("P_i_centered: ", P_i_centered)
    print("Q_i_centered: ", Q_i_centered)
    
    first_sum = np.sum(Q_i_centered[:,0]*P_i_centered[:,1] - Q_i_centered[:,1]*P_i_centered[:,0])
    sec_sum = np.sum(Q_i_centered[:,0]*P_i_centered[:,0] + Q_i_centered[:,1]*P_i_centered[:,1])
    theta = atan2(first_sum, sec_sum)
    
    #Hier steckt noch ein Fehler drin da die rotation um z in world hier nicht passend ist 
    R = np.array([[ np.cos(theta), -np.sin(theta)],
                  [ np.sin(theta),  np.cos(theta)]])
    
    t = m_P - R @ m_Q

    return R, t, theta

