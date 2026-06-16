from math import *
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

from typing import List, Tuple
from numpy.typing import NDArray

from visual_odom.constants import *
from visual_odom.visual_odom_map import *

class ExtendedKalmanFilterRobot:
	def __init__(self, x: State, P: NDArray, Q: NDArray = None):
		self.x = x
		self.P = P
		if Q is not None:
			self.Q = Q
		else:
			self.set_Q()

		self.F = np.array([[1.0, 0.0, 0.0], 
					 		[0.0, 1.0, 0.0], 
							[0.0, 0.0, 1.0]])
		
		self.H = np.array([[1.0, 0.0, 0.0], 
					 		[0.0, 1.0, 0.0], 
							[0.0, 0.0, 1.0]])
		self.last_ransac_pose  = x
		self.last_u = State(0.0, 0.0, 0.0)
		self.additional_noise = 0.0

	def prediction(self, delta_wheel_odom: State) -> None:
		x_tt1 = self.x + delta_wheel_odom

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
	def update(self, z: State, inlier_ratio: float) -> None:
		self.set_R(inlier_ratio)
		K = self.computeKalmanGain(self.P)

		z_tt1 = self.x
		
		delta_z = np.array([z.x - z_tt1.x, z.y - z_tt1.y, z.theta - z_tt1.theta])
		
		delta = K@delta_z
		
		x_t1t1 = self.x + State(delta[0], delta[1], delta[2])
		I_KH = np.eye(3) - K @ self.H
		P_t1t1 = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

		self.x = x_t1t1
		self.P = P_t1t1
	
	def set_R(self, inlier_ratio: float):
		sigma = constants.parameters.ransac_evaluation_tolerance + self.additional_noise
		self.R = np.array([[sigma**2, 0.0, 0.0], 
						   [0.0, sigma**2, 0.0], 
						   [0.0, 0.0, sigma**2]])
		

		
	def set_Q(self) -> None:
		sigma_x = SIGMA_X_ODOM_WHEEL_Q
		sigma_y = SIGMA_Y_ODOM_WHEEL_Q
		sigma_theta = SIGMA_THETA_ODOM_WHEEL_Q
		self.Q = np.array([[sigma_x**2, 0.0, 0.0], 
						   [0.0, sigma_y**2, 0.0], 
						   [0.0, 0.0, sigma_theta**2]])
		
	def get_state(self):
		return self.x

	def get_covariance_p(self):
		return self.P
	
	def add_noise_to_R(self):
		self.additional_noise = min(self.additional_noise + NOISE_INCREMENT_RANSAC_FAILURE, MAX_ADDITIONAL_NOISE_RANSAC_FAILURE)

