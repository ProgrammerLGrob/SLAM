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
		self.Q = Q

	def state_func(self, x: State, delta: State) -> State:
		theta_mid = normalize_angle(x.theta + 0.5 * delta.theta)  # kein In-place!
		c, s = cos(theta_mid), sin(theta_mid)
		R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
		delta_x = R @ np.array([delta.x, delta.y, delta.theta])
		return State(float(x.x + delta_x[0]),
					float(x.y + delta_x[1]),
					float(normalize_angle(theta_mid + delta_x[2] * 0.5)))
	
	def set_jacobi_F_matrix(self, x_tt1: State, delta: State) -> None: 
		c = cos(x_tt1.theta)
		s = sin(x_tt1.theta)
		#F = np.array([[c, -s, -x_tt1.x*s -x_tt1.y*c],
		#			  [s, c, x_tt1.x*c - x_tt1.y*s],
		#			  [0.0, 0.0, 1.0]])
		
		F = np.array([[1.0, 0.0, -delta.x*s -delta.y*c],
					  [0.0, 1.0, delta.x*c - delta.y*s],
					  [0.0, 0.0, 		1.0				]])
		
		self.set_JF(F)	

	def calculate_Q_matrix(self, x: State) -> NDArray:
		c = cos(x.theta)
		s = sin(x.theta)
		R = np.array([[c, -s, 0.0],
					  [s, c, 0.0],
					  [0.0, 0.0, 1.0]])

		sx, sy, st = self.sigma_Q_approximation(x)

		Q = np.array([[sx**2, 0, 0],
					  [0, sy**2, 0],
					  [0, 0, st**2]])
		Q = R@Q@R.transpose()
		return Q

	def meas_func(self, coor: Coordinate, x_tt1: State) -> NDArray:
		
		c = cos(x_tt1.theta)
		s = sin(x_tt1.theta)
		R = np.array([[c, s],
					  [-s, c]])
		h_x = R@(np.array([coor.x-x_tt1.x, coor.y-x_tt1.y]))

		return h_x
	
	def calc_and_set_jacobi_H_matrix(self, coor: Coordinate, x_tt1: State) -> None:
		c = cos(x_tt1.theta)
		s = sin(x_tt1.theta)
		H = np.array([[-c , -s , -s * (coor.x-x_tt1.x) + c*(coor.y-x_tt1.y)],
					  [s, -c , -c * (coor.x-x_tt1.x) - s*(coor.y-x_tt1.y)]])
		
		self.set_JH(H)

	def kalman_iteration(self, delta_p: Coordinate, delta_theta: float, z_dic: dict, visible_landmarks: VisualOdomMap) -> Tuple[State, NDArray]:
		x_tt1, P_tt1 = self.prediction(self.x, delta_p, delta_theta)
		self.x.theta = normalize_angle(self.x.theta)
		self.x = x_tt1
		self.P = P_tt1
		

		return self.x, self.P

	def prediction(self, x: State, delta_p: Coordinate, delta_theta: float) -> Tuple[State, NDArray]:
		x_tt1 = self.state_func(x, State(delta_p.x, delta_p.y, delta_theta))
		self.set_Q(x_tt1)
		self.set_jacobi_F_matrix(x_tt1, delta=State(delta_p.x, delta_p.y, delta_theta))

		P_tt1 = np.matmul(self.JF, np.matmul(self.P, self.JF.transpose()))+self.Q
		#print("Predicted state:", x_tt1)

		return x_tt1, P_tt1

	# return measurement prediction (\hat z_{t|t-1})
	def predictMeasurement(self, coor: Coordinate, x_tt1: State) -> NDArray:
		pmeas = self.meas_func(coor, x_tt1)
		return pmeas
	
	# return matrix K
	def computeKalmanGain(self, P_tt1: NDArray) -> NDArray:
		PHT = np.matmul(P_tt1, self.JH.transpose())         # PH^\top
		HPHT = np.matmul(self.JH, PHT)                      # HPH^\top
		HPHTpRi = np.linalg.inv(HPHT + self.R)             # (HPH^\top + R)^{-1}
		K = np.matmul(PHT, HPHTpRi)
		return K

	# Update self.x and self.P, return tuple (x_{t|t}, P_{t_t})
	def update(self, x_tt1: State, P_tt1: NDArray, l: Landmark, z: NDArray, depth_value: float) -> Tuple[State, NDArray]:
		self.calc_and_set_jacobi_H_matrix(l.get_odom_coordinates(), x_tt1)
		self.set_R(depth_value)
		K = self.computeKalmanGain(P_tt1)

		z_tt1 = self.predictMeasurement(l.get_odom_coordinates(), x_tt1)
		#print("Predicted measurement:", z_tt1)
		#print("Actual measurement:", z)

	
		delta_z = (z-z_tt1) #Compare measurement with hat(z) in base link 
		
		delta =np.matmul(K, delta_z)
		
		x = x_tt1 + State(delta[0], delta[1], delta[2])
		P = P_tt1 - np.matmul(K, np.matmul(self.JH, P_tt1))
		return x, P
	
	# set Jacobi matrix of the state transition
	def set_JF(self, JF: NDArray) -> None:
		self.JF = JF
		
	# set Jacobi Matrix of the measurement function
	def set_JH(self, JH: NDArray) -> None:
		self.JH = JH

	# set measurement noise -- eg. for EKF
	def set_R(self, depth_value: float, x: State) -> None:
		sx, sy = self.sigma_R_approximation(depth_value)
		#sx, sy = 0.01, 0.01
		R = np.array([[sx**2, 0],
					  [0, sy**2]])
		c = cos(x.theta)
		s = sin(x.theta)
		R_rot = np.array([[c, -s],
				  		[s, c]])
		self.R = R_rot@R@R_rot.transpose()	
				

	# set model noise -- eg. for EKF
	def set_Q(self, x: State) -> None:
		self.Q = self.calculate_Q_matrix(x)
	
	def sigma_Q_approximation(self, x: State) -> Tuple[float, float, float]:
		d = np.linalg.norm([x.x, x.y])

		sx = self.error_x(d, x.theta)/3
		sy = self.error_y(d, x.theta)/3
		st = self.error_theta(d, x.theta)/3

		sx = max(abs(sx), 0.005)   # 5mm - typisches Encoder-Mindestrauschen
		sy = max(abs(sy), 0.005)   # 5mm
		st = max(abs(st), 0.001)   # ~0.06° - typisches Gyro-Mindestrauschen

		sx = 0.005
		sy = 0.005
		st = 0.001

		return sx, sy, st
	
	def sigma_R_approximation(self, depth_value: float) -> Tuple[float, float]:
		#a→ konstanter Offset (Rauschen bei minimalem Abstand)
		#b→ quadratischer Koeffizient (fitted)
		a = 0.001477
		b = 0.002294

		 # Tiefenfehler (d^2 für Kinect structured light)
		s_z = a + b * (depth_value - MIN_DEPTH/1000)**2

		# Lateraler Fehler (Bogenlänge, ~0.086° Auflösung)
		s_lat = depth_value * (2 * pi * 0.086 / 360)

		# Gesamtfehler (RSS - root sum of squares)
		s_x = sqrt(s_lat**2 + 0.001**2) 


		return s_z, s_x
	


	def error_x(self, d: float, theta: float) -> float:
		return (
			ERR_FUNC_X_CONST
			+ ERR_FUNC_X_D_1 * d
			+ ERR_FUNC_X_T_1 * theta
			+ ERR_FUNC_X_D_2 * d**2
			+ ERR_FUNC_X_D1_T1 * d * theta
			+ ERR_FUNC_X_T_2 * theta**2
			+ ERR_FUNC_X_D_3 * d**3
			+ ERR_FUNC_X_D2_T1 * d**2 * theta
			+ ERR_FUNC_X_D1_T2 * d * theta**2
			+ ERR_FUNC_X_T_3 * theta**3
		)


	def error_y(self, d: float, theta: float) -> float:
		return (
			ERR_FUNC_Y_CONST
			+ ERR_FUNC_Y_D_1 * d
			+ ERR_FUNC_Y_T_1 * theta
			+ ERR_FUNC_Y_D_2 * d**2
			+ ERR_FUNC_Y_D1_T1 * d * theta
			+ ERR_FUNC_Y_T_2 * theta**2
			+ ERR_FUNC_Y_D_3 * d**3
			+ ERR_FUNC_Y_D2_T1 * d**2 * theta
			+ ERR_FUNC_Y_D1_T2 * d * theta**2
			+ ERR_FUNC_Y_T_3 * theta**3
		)


	def error_theta(self, d: float, theta: float) -> float:
		return (
			ERR_FUNC_THETA_CONST
			+ ERR_FUNC_THETA_D_1 * d
			+ ERR_FUNC_THETA_T_1 * theta
			+ ERR_FUNC_THETA_D_2 * d**2
			+ ERR_FUNC_THETA_D1_T1 * d * theta
			+ ERR_FUNC_THETA_T_2 * theta**2
			+ ERR_FUNC_THETA_D_3 * d**3
			+ ERR_FUNC_THETA_D2_T1 * d**2 * theta
			+ ERR_FUNC_THETA_D1_T2 * d * theta**2
			+ ERR_FUNC_THETA_T_3 * theta**3
		)
