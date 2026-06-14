from math import *
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

from typing import List, Tuple
from numpy.typing import NDArray

from visual_odom.constants import *
from visual_odom.visual_odom_map import *

class ExtendedKalmanFilterRobot:
	def __init__(self, x: State, P: NDArray, Q: NDArray):
		self.x = x
		self.P = P
		#self.Q = Q -> Hier auskommentiert, weil im predict eh immer setQ drin stand!
		self.Q = self.set_Q()
		self.F = np.array([[1.0, 0.0, 0.0], 
					 		[0.0, 1.0, 0.0], 
							[0.0, 0.0, 1.0]])
		
		self.H = np.array([[1.0, 0.0, 0.0], 
					 		[0.0, 1.0, 0.0], 
							[0.0, 0.0, 1.0]])
		self.last_ransac_pose  = x
		self.last_wheel_odom = State(0.0, 0.0, 0.0)
	
	def kalman_iteration(self, state_wheel_odom: State, delta_ransac: State | None, inlier_ratio: float) -> Tuple[State, NDArray]:
		x_tt1, P_tt1 = self.prediction(self.x, state_wheel_odom)

		if inlier_ratio > 0.05 and delta_ransac is not None:
			x_t1t1, P_t1t1 = self.update(x_tt1, P_tt1, delta_ransac, inlier_ratio)
			return x_t1t1, P_t1t1
		else:
			return x_tt1, P_tt1

	def prediction(self, state_wheel_odom: State): #keine Rückgabe, weil P und x im EKF gepsiehcert werden; x Rückgabe im Update mit oder ohne RANSAC Update
		
		delta_pos = state_wheel_odom - self.last_wheel_odom
		self.last_wheel_odom = state_wheel_odom

		delta_pos = np.array([delta_pos.x, delta_pos.y])
		x_tt1 = self.x + State(delta_pos[0], delta_pos[1], delta_pos.theta)
		
		P_tt1 = self.F@self.P@self.F.T + self.Q
		
		self.x = x_tt1
		self.P = P_tt1

	
	def computeKalmanGain(self, P_tt1: NDArray) -> NDArray:
		PHT = P_tt1@self.H.T         # PH^\top
		HPHT = self.H@PHT                     # HPH^\top
		HPHTpRi = np.linalg.inv(HPHT + self.R)             # (HPH^\top + R)^{-1}
		K = PHT@HPHTpRi                     # K = PH^\top(HPH^\top + R)^{-1}
		return K

	# Update self.x and self.P, return tuple (x_{t|t}, P_{t_t})
	def update(self, x_tt1: State, P_tt1: NDArray, delta_ransac: State, inlier_ratio: float) -> Tuple[State, NDArray]:
		z = self.last_ransac_pose  + delta_ransac

		self.set_R(inlier_ratio)
		K = self.computeKalmanGain(P_tt1)

		z_tt1 = x_tt1
		
		delta_z = np.array([z.x - z_tt1.x, z.y - z_tt1.y, z.theta - z_tt1.theta])
		
		delta = K@delta_z
		
		x_t1t1 = x_tt1 + State(delta[0], delta[1], delta[2])
		I_KH = np.eye(3) - K @ self.H
		P_t1t1 = I_KH @ P_tt1 @ I_KH.T + K @ self.R @ K.T

		self.last_ransac_pose  = x_t1t1

		self.x = x_t1t1
		self.P = P_t1t1
		return x_t1t1, P_t1t1
	
	def set_R(self, inlier_ratio: float):
		sigma = constants.parameters.ransac_evaluation_tolerance
		self.R = np.array([[sigma**2, 0.0, 0.0], 
						   [0.0, sigma**2, 0.0], 
						   [0.0, 0.0, sigma**2]])
		
	def set_Q(self) -> NDArray:
		sigma = 0.0025
		self.Q = np.array([[sigma**2, 0.0, 0.0], 
						   [0.0, sigma**2, 0.0], 
						   [0.0, 0.0, sigma**2]])
