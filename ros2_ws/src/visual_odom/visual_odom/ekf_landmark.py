from math import *
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

from typing import List, Tuple
from numpy.typing import NDArray

from visual_odom.constants import *
from visual_odom.tf_methods import *

class ExtendedKalmanFilterLandmark:
	def __init__(self, x: State, P: NDArray):
		self.x = x
		self.P = P
		self.Q = self.calculate_Q_matrix()

	
	def kalman_iteration(self, pos_baselink: Coordinate, delta_theta: float, pixel_coor: PixelCoordinate) -> Tuple[State, NDArray]:
		x_tt1, P_tt1 = self.prediction(self.x)
		self.x.theta = normalize_angle(self.x.theta)

		updated_x, updated_P = self.update(x_tt1, P_tt1, pos_baselink, delta_theta, pixel_coor)
		self.x.theta = normalize_angle(self.x.theta)

		self.x = updated_x
		self.P = updated_P
		
		return self.x, self.P
	
	def state_func(self, x: Coordinate) -> Coordinate:
		return x
	
	def calculate_jacobian_F(self) -> NDArray:
		F = np.array([[1.0, 0.0, 0],
					  [0.0, 1.0, 0],
					  [0.0, 0.0, 1.0]])
		return F

	def calculate_Q_matrix(self) -> NDArray:
		Q = np.array([[0.0, 0.0, 0.0],
					  [0.0, 0.0, 0.0],
					  [0.0, 0.0, 0.0]])
		return Q
	
	def meas_func(self, x_tt1: Coordinate, pos_robot: Coordinate, theta_robot: float) -> NDArray:
		c = cos(theta_robot)
		s = sin(theta_robot)
		R_theta = np.array([[c, -s, 0.0],
					  [s, c, 0.0],
					  [0.0, 0.0, 1.0]])
		h_x = R_theta.T@(np.array([x_tt1.x, x_tt1.y, x_tt1.z]) - np.array([pos_robot.x, pos_robot.y, pos_robot.z]))
		return h_x
	
	def calculate_jacobian_H(self, theta_robot: float) -> NDArray:
		c = cos(theta_robot)
		s = sin(theta_robot)
		R_theta = np.array([[c, -s, 0.0],
					  [s, c, 0.0],
					  [0.0, 0.0, 1.0]])
		H = R_theta.T
		
		return H
	
	def sigma_R_approximation(self, kp: PixelCoordinate) -> NDArray:
		# Tiefenfehler (d^2 für Kinect structured light)
		s_z = ERROR_MIN_DEPTH + ERROR_QUADRATIC_DEPTH * (kp.z - MIN_DEPTH/1000)**2

		R_sigma = np.array([[SIGMA_PIXEL**2, 0, 0],
							[0, SIGMA_PIXEL**2, 0],
							[0, 0, s_z**2]])
		return R_sigma
	
	def calculate_R_sigma_matrix(self, kp: PixelCoordinate) -> NDArray:
		"""Calculate the measurement noise covariance matrix R in base_link frame based on the depth value."""
		
		R_sigma_pixel = self.sigma_R_approximation(kp)
		J_kinect = np.array([[kp.z/(F*1000), 0, (kp.u-CU)/F],
						  [0, kp.z/(F*1000), (kp.v-CV)/F],
						  [0, 0, 1]])
		
		R_sigma_kinect = J_kinect@R_sigma_pixel@J_kinect.T  # Transformiere die Pixel-Varianz in die Kinect-Koordinaten
		J_baselink = ROT_KB.T  # Rotation von Kinect zu Base Link
		R_sigma_base_link = J_baselink@R_sigma_kinect@J_baselink.T

		return R_sigma_base_link

	def prediction(self, x: Coordinate) -> Tuple[Coordinate, NDArray]:
		x_tt1 = self.state_func(x)
		self.set_Q()
		self.set_jacobian_F()

		P_tt1 = self.JF@self.P@self.JF.T+self.Q

		return x_tt1, P_tt1


	def predictMeasurement(self, x_tt1: Coordinate, pos_robot: Coordinate, theta_robot: float) -> NDArray:
		pmeas = self.meas_func(x_tt1, pos_robot, theta_robot)
		return pmeas
	
	# return matrix K
	def computeKalmanGain(self, P_tt1: NDArray) -> NDArray:
		PHT = P_tt1@self.JH.T         # PH^\top
		HPHT = self.JH@PHT                      # HPH^\top

		HPHTpRi = np.linalg.inv(HPHT + self.R)             # (HPH^\top + R)^{-1}
		K = PHT@HPHTpRi
		return K

	# Update self.x and self.P, return tuple (x_{t|t}, P_{t_t})
	def update(self, x_tt1: State, P_tt1: NDArray, pos_robot: Coordinate, theta_robot: float, kp: PixelCoordinate) -> Tuple[Coordinate, NDArray]:
		z = kinect_depth_to_baselink(pixel_to_kinect(kp))

		self.set_jacobian_H(theta_robot)
		self.set_R(kp)
		K = self.computeKalmanGain(P_tt1)

		z_tt1 = self.predictMeasurement(x_tt1, pos_robot, theta_robot)
		#print("Predicted measurement:", z_tt1)
		#print("Actual measurement:", z)

	
		delta_z = (z-z_tt1) #Compare measurement with hat(z) in base link 
		#rclpy.logging.get_logger(__name__).info("Actual measurement delta: {}".format(delta_z))
		
		delta = K@delta_z
		
		x = x_tt1 + Coordinate(delta[0], delta[1], delta[2])
		P = P_tt1 - K@self.JH@P_tt1
		return x, P
	
	# set Jacobi matrix of the state transition
	def set_jacobian_F(self) -> None:
		self.JF = self.calculate_jacobian_F()
		
	# set Jacobi Matrix of the measurement function
	def set_jacobian_H(self, theta_robot: float) -> None:
		self.JH = self.calculate_jacobian_H(theta_robot)

	# set measurement noise
	def set_R(self, kp: PixelCoordinate) -> None:
		self.R = self.calculate_R_sigma_matrix(kp)
				
	# set model noise -- eg. for EKF
	def set_Q(self) -> None:
		self.Q = self.calculate_Q_matrix()
	
	