#!/usr/bin/env python3
import math

from tf2_ros import Buffer, TransformBroadcaster, TransformListener
from geometry_msgs.msg import TransformStamped, Point
from scipy.spatial.transform import Rotation
from visualization_msgs.msg import Marker

from numpy.typing import NDArray
from rclpy.time import Time
from scipy.spatial import KDTree


import time

from math import pi, sin, cos, tan, atan2
import random
from typing import List, Tuple

import rclpy
from rclpy.node import Node, Publisher
from sensor_msgs.msg import Image, PointCloud2, PointField
from sensor_msgs_py import point_cloud2

import cv2
from cv_bridge import CvBridge

import numpy as np
from numba import njit

from visual_odom.landmark import *
from visual_odom.visual_odom_map import *
from visual_odom.tf_methods import *
from visual_odom.constants import *
from visual_odom.ekf_robot import *
import visual_odom.constants as constants

from nav_msgs.msg import Odometry


# ──────────────────────────────────────────────
#  Numba-JIT-Kernels  (werden beim ersten Aufruf
#  kompiliert und danach gecacht)
# ──────────────────────────────────────────────

@njit(cache=True)
def _kabsch_numba(P: np.ndarray, Q: np.ndarray):
    """
    Kabsch-Algorithmus für 2-D-Punktwolken.
    Gibt (R, t, theta) zurück oder (None, None, nan) bei Fehler.
    """
    m_P0 = 0.0; m_P1 = 0.0
    m_Q0 = 0.0; m_Q1 = 0.0
    n = P.shape[0]
    for i in range(n):
        m_P0 += P[i, 0]; m_P1 += P[i, 1]
        m_Q0 += Q[i, 0]; m_Q1 += Q[i, 1]
    m_P0 /= n; m_P1 /= n
    m_Q0 /= n; m_Q1 /= n

    first_sum = 0.0; sec_sum = 0.0
    for i in range(n):
        pc0 = P[i, 0] - m_P0; pc1 = P[i, 1] - m_P1
        qc0 = Q[i, 0] - m_Q0; qc1 = Q[i, 1] - m_Q1
        first_sum += qc0 * pc1 - qc1 * pc0
        sec_sum   += qc0 * pc0 + qc1 * pc1

    theta = math.atan2(first_sum, sec_sum)
    c = math.cos(theta); s = math.sin(theta)

    # t = m_P - R @ m_Q
    t0 = m_P0 - (c * m_Q0 - s * m_Q1)
    t1 = m_P1 - (s * m_Q0 + c * m_Q1)

    return c, s, t0, t1, theta


@njit(cache=True)
def _ransac_core(P: np.ndarray, Q: np.ndarray,
                 iteration: int, sample_size: int,
                 tolerance: float, early_break_ratio: float):
    """
    RANSAC-Kern vollständig in Numba.
    Gibt best_mask (bool-Array, Länge n) und best_inlier_count zurück.
    """
    n = P.shape[0]
    best_inlier_count = 0
    mean_last_e = 1e9
    best_mask = np.zeros(n, dtype=np.bool_)

    idx = np.arange(n)

    for _ in range(iteration):
        # Fisher-Yates-Shuffle für sample_size Elemente
        for i in range(sample_size):
            j = i + int(np.random.randint(0, n - i))  # numba unterstützt randint
            tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp

        # Kabsch auf sample_size Punkte
        Ps = P[idx[:sample_size]]
        Qs = Q[idx[:sample_size]]
        c, s, t0, t1, theta = _kabsch_numba(Ps, Qs)

        # Residuen für alle n Punkte
        inlier_count = 0
        mean_e = 0.0
        mask = np.empty(n, dtype=np.bool_)
        for i in range(n):
            rx = P[i, 0] - (c * Q[i, 0] - s * Q[i, 1] + t0)
            ry = P[i, 1] - (s * Q[i, 0] + c * Q[i, 1] + t1)
            e = math.sqrt(rx * rx + ry * ry)
            mean_e += e
            is_in = e < tolerance
            mask[i] = is_in
            if is_in:
                inlier_count += 1
        mean_e /= n

        if inlier_count > best_inlier_count or (
                inlier_count == best_inlier_count and mean_e < mean_last_e):
            best_inlier_count = inlier_count
            mean_last_e = mean_e
            best_mask = mask.copy()

        if (best_inlier_count / n) > early_break_ratio or mean_e < tolerance * 4:
            break

    return best_mask, best_inlier_count


# ──────────────────────────────────────────────
#  Warm-up: JIT-Kompilierung beim Modulimport
#  (läuft einmalig, danach gecacht)
# ──────────────────────────────────────────────
def _warmup_numba():
    dummy_P = np.random.rand(5, 2).astype(np.float64)
    dummy_Q = np.random.rand(5, 2).astype(np.float64)
    _ransac_core(dummy_P, dummy_Q, 2, 3, 0.1, 0.8)

_warmup_numba()


# ──────────────────────────────────────────────
#  Haupt-Klasse
# ──────────────────────────────────────────────

class VisualRobotSample():
    def __init__(self, pos: Coordinate, theta: float, covariance_P: NDArray,
                 matcher: cv2.BFMatcher, odom_publisher: Publisher,
                 map_publisher: Publisher, cone_publisher: Publisher,
                 valid_kp: List[cv2.KeyPoint], valid_des: np.ndarray,
                 valid_kp_depth: np.ndarray, frame_rgb: NDArray):
        self.theta = theta
        self.pos_baselink = pos

        self.bf = matcher

        self.first_iteration = True
        self.ransac_delta_p = None
        self.ransac_inlier_ratio = 0.0
        self.visual_odom_map = VisualOdomMap()
        self.visual_odom_map.add_landmarks_from_kps(
            covariance_P, valid_kp, valid_des, frame_rgb,
            valid_kp_depth, self.theta, self.pos_baselink)

        self.odometry_msg_publisher = odom_publisher
        self.map_publisher = map_publisher
        self.cone_publisher = cone_publisher
        self.covariance_P = covariance_P
        self.noise_Q = np.eye(3) * 1e-6
        self.ekf = ExtendedKalmanFilterRobot(
            State(self.pos_baselink.x, self.pos_baselink.y, self.theta),
            self.covariance_P, self.noise_Q)
        self.log_weight = 0.0
        self.ransac_draw_keypoints = []
        self.last_wheel_odom = None
        self.delta_wheel_odom_ransac = State(0.0, 0.0, 0.0)
        self.pos_visual_odom = State(
            self.pos_baselink.x, self.pos_baselink.y, self.theta)

    # ── Publishing ──────────────────────────────

    def publish_yourself(self, rgb_stamp):
        self.publish_odometry_msg(
            self.odometry_msg_publisher, self.pos_baselink, self.theta,
            self.covariance_P, rgb_stamp, VISUAL_ODOM_FRAME_ID, BASE_LINK_FRAME_ID)
        self.publish_pointcloud_map(self.map_publisher, rgb_stamp)
        self.publish_vision_cone(self.cone_publisher, rgb_stamp)

    def predict_with_wheel_odom(self, state_wheel_odom: State):
        if self.last_wheel_odom is None:
            self.last_wheel_odom = state_wheel_odom
            return
        delta_wheel_odom = state_wheel_odom - self.last_wheel_odom
        self.ekf.prediction(delta_wheel_odom)
        self.last_wheel_odom = state_wheel_odom
        self.delta_wheel_odom_ransac += delta_wheel_odom

    # ── Haupt-Iterations-Methode ────────────────

    def robot_iteration(self, valid_kp: List[cv2.KeyPoint], valid_des: np.ndarray,
                        valid_kp_depth: np.ndarray, frame_rgb, depth_frame, rgb_stamp):
        self.frame_rgb = frame_rgb
        self.frame_depth = depth_frame

        if self.first_iteration and len(self.visual_odom_map) < INITIAL_LANDMARK_COUNT:
            self.visual_odom_map.add_landmarks_from_kps(
                self.covariance_P, valid_kp, valid_des, self.frame_rgb,
                valid_kp_depth, self.theta, self.pos_baselink)
            return
        else:
            if (math.sqrt(self.delta_wheel_odom_ransac.x ** 2 +
                          self.delta_wheel_odom_ransac.y ** 2) < 0.05
                    and abs(self.delta_wheel_odom_ransac.theta) < 0.06):
                return

            self.first_iteration = False
            pos_camera = kinect_depth_to_odom(
                Coordinate(0.0, 0.0, 0.0), self.theta, self.pos_baselink)

            self.visible_landmarks = self.visual_odom_map.get_visible_landmarks(
                pos_camera, self.theta)

            matches = self.bf.match(
                self.visible_landmarks.get_descriptors(), valid_des)

            if len(matches) < constants.parameters.min_matches_for_ransac:
                self.visual_odom_map.add_landmarks_from_kps(
                    self.covariance_P, valid_kp, valid_des, self.frame_rgb,
                    valid_kp_depth, self.theta, self.pos_baselink)
                return

            ransac_result = self.ransac(
                matches, self.visible_landmarks.get_odom_coordinates(),
                valid_kp, valid_kp_depth)

            self.ransac_delta_p         = ransac_result[0]
            self.ransac_delta_theta     = ransac_result[1]
            ransac_landmark_indices     = ransac_result[2]
            self.ransac_draw_keypoints  = ransac_result[3]
            ransac_kp_indices           = ransac_result[4]
            ransac_not_matched_kp_indices = ransac_result[5]
            ransac_inlier_count         = ransac_result[6]
            self.ransac_inlier_ratio = (float(ransac_inlier_count) / len(matches)
                                        if len(matches) > 0 else 0.0)

            if self.ransac_inlier_ratio < constants.parameters.ransac_min_inlier_ratio:
                self.visual_odom_map.add_landmarks_from_kps(
                    self.covariance_P, valid_kp, valid_des, self.frame_rgb,
                    valid_kp_depth, self.theta, self.pos_baselink)
                rclpy.logging.get_logger("RANSAC").info(
                    f"RANSAC result rejected due to low inlier ratio: "
                    f"{float(ransac_inlier_count)/len(matches):.2f} with "
                    f"{ransac_inlier_count} inliers out of {len(matches)} matches.")
                self.ransac_failed()
                return

            c = cos(self.theta)
            s = sin(self.theta + self.ransac_delta_theta)
            R = np.array([[c, -s], [s, c]])
            delta = R @ np.array([self.ransac_delta_p.x, self.ransac_delta_p.y])
            self.ransac_delta_p = Coordinate(delta[0], delta[1], 0.0)

            self.delta_ransac_state = State(
                self.ransac_delta_p.x, self.ransac_delta_p.y, self.ransac_delta_theta)
            self.pos_visual_odom = self.pos_visual_odom + self.delta_ransac_state

            self.ekf.update(self.pos_visual_odom, self.ransac_inlier_ratio)
            self.delta_wheel_odom_ransac = State(0.0, 0.0, 0.0)

            state = self.ekf.get_state()
            self.pos_baselink = Coordinate(state.x, state.y, 0.0)
            self.theta = state.theta
            self.covariance_P = self.ekf.get_covariance_p()

            kp_pos = []
            for i in ransac_kp_indices:
                u, v = valid_kp[i].pt
                u = int(round(u)); v = int(round(v))
                depth_value = self.frame_depth[v, u]
                kp_pos.append(PixelCoordinate(u, v, depth_value))

            self.log_weight = self.visible_landmarks.calculate_log_weight(
                self.pos_baselink, self.theta, ransac_landmark_indices, kp_pos)
            self.visible_landmarks.landmark_kalman_iteration(
                self.pos_baselink, self.theta, ransac_landmark_indices, kp_pos)

            self.visual_odom_map.cleanup_old_landmarks(
                self.visible_landmarks, ransac_landmark_indices)

            if len(matches) < constants.parameters.matches_for_new_landmarks:
                not_matched_kp  = [valid_kp[i]        for i in ransac_not_matched_kp_indices]
                not_matched_des = [valid_des[i]        for i in ransac_not_matched_kp_indices]
                not_matched_kp_depth = [valid_kp_depth[i] for i in ransac_not_matched_kp_indices]
                self.visual_odom_map.add_landmarks_from_kps(
                    self.covariance_P, not_matched_kp, not_matched_des,
                    self.frame_rgb, not_matched_kp_depth, self.theta, self.pos_baselink)

    def ransac_failed(self):
        self.pos_visual_odom = self.pos_visual_odom + self.delta_wheel_odom_ransac
        self.ekf.add_noise_to_R()
        self.ekf.update(self.pos_visual_odom, self.ransac_inlier_ratio)
        state = self.ekf.get_state()
        self.pos_baselink = Coordinate(state.x, state.y, 0.0)
        self.theta = state.theta
        self.delta_wheel_odom_ransac = State(0.0, 0.0, 0.0)

    # ── RANSAC ──────────────────────────────────

    def ransac(self, matches, landmarks_odom_pos, valid_kp, valid_kp_depth):
        t0 = time.perf_counter()
        # --- Vektorisierter Prep-Loop ---
        train_ids = np.array([m.trainIdx for m in matches], dtype=np.intp)
        query_ids = np.array([m.queryIdx for m in matches], dtype=np.intp)
        n = len(matches)

        valid_kp_depth_arr = np.asarray(valid_kp_depth, dtype=np.float64)

        # Q: Kinect → base_link (vektorisiert)
        pts = np.array([valid_kp[i].pt for i in train_ids])       # (n,2)
        z   = valid_kp_depth_arr[train_ids]                        # (n,)
        u   = np.round(pts[:, 0]).astype(np.float64)
        v   = np.round(pts[:, 1]).astype(np.float64)

        x_k = z * (u - CU) / F / 1000.0
        y_k = z * (v - CV) / F / 1000.0
        z_k = z / 1000.0
        kinect_pts = np.stack([x_k, y_k, z_k], axis=1)            # (n,3)

        Q_3d = (ROT_BK @ kinect_pts.T).T                          # (n,3)
        Q_3d[:, 0] += CAMERA_POS_IN_BASELINK.x
        Q_3d[:, 1] += CAMERA_POS_IN_BASELINK.y
        Q_array = np.ascontiguousarray(Q_3d[:, :2])               # (n,2)

        # P: odom → base_link (vektorisiert)
        c_t, s_t = cos(self.theta), sin(self.theta)
        R_odom = np.array([[c_t, -s_t], [s_t, c_t]])

        lm_pts = np.array([[landmarks_odom_pos[i].x,
                             landmarks_odom_pos[i].y] for i in query_ids])
        base = np.array([self.pos_baselink.x, self.pos_baselink.y])
        P_array = np.ascontiguousarray((R_odom.T @ (lm_pts - base).T).T)  # (n,2)

        # Prep (pts, z, Q_array, P_array)
        t1 = time.perf_counter()
    

        # --- Numba-RANSAC-Kern ---
        best_mask, best_inlier_count = _ransac_core(
            P_array, Q_array,
            int(constants.parameters.ransac_iteration),
            int(constants.parameters.ransac_sample_size),
            float(constants.parameters.ransac_evaluation_tolerance),
            0.80
        )
        # RANSAC-Loop
        t2 = time.perf_counter()
       
        if best_inlier_count < constants.parameters.min_matches_for_ransac:
            rclpy.logging.get_logger("RANSAC").info(
                f"RANSAC failed to find a valid transformation with enough inliers. "
                f"Best inlier count: {best_inlier_count} out of {n} matches.")
            return Coordinate(0.0, 0.0, 0.0), 0.0, [], [], [], [], 0

        # Inlier-Listen (Python-Teil, läuft nur einmal nach dem Loop)
        best_P_inlier      = P_array[best_mask]
        best_Q_inlier      = Q_array[best_mask]
        best_valid_kp_index  = [matches[i].trainIdx for i in range(n) if best_mask[i]]
        best_draw_Q_inlier   = [valid_kp[matches[i].trainIdx] for i in range(n) if best_mask[i]]
        best_landmark_index  = [matches[i].queryIdx for i in range(n) if best_mask[i]]

        # Finales Kabsch auf Inliern
        R, delta_t, delta_theta = kabsch(best_P_inlier, best_Q_inlier,
                                         constants.parameters.max_rotation_angle_deg)

        best_t = Coordinate(float(delta_t[0]), float(delta_t[1]), 0.0)

        valid_set    = set(best_valid_kp_index)
        not_matched_kp = [i for i in range(len(valid_kp)) if i not in valid_set]

         # Post (best_*-Listen, not_matched_kp)
        t3 = time.perf_counter()
        print(f"Prep={1000*(t1-t0):.2f}ms Loop={1000*(t2-t1):.2f}ms Post={1000*(t3-t2):.2f}ms")


        return best_t, delta_theta, best_landmark_index, best_draw_Q_inlier, best_valid_kp_index, not_matched_kp, best_inlier_count

    # ── Publishing-Hilfsmethoden ────────────────

    def publish_odometry_msg(self, publisher, pos: Coordinate, theta: float,
                             covariance: NDArray, timestamp,
                             parent_frame_id: str, child_frame_id: str):
        msg = Odometry()
        msg.header.stamp = timestamp
        msg.header.frame_id = parent_frame_id
        msg.child_frame_id = child_frame_id

        msg.pose.pose.position.x = pos.x
        msg.pose.pose.position.y = pos.y
        msg.pose.pose.position.z = pos.z

        euler = Rotation.from_euler('z', float(theta))
        quat  = euler.as_quat(canonical=True)
        msg.pose.pose.orientation.x = quat[0]
        msg.pose.pose.orientation.y = quat[1]
        msg.pose.pose.orientation.z = quat[2]
        msg.pose.pose.orientation.w = quat[3]

        msg.pose.covariance = [
            covariance[0, 0], covariance[0, 1], 0.0, 0.0, 0.0, covariance[0, 2],
            covariance[1, 0], covariance[1, 1], 0.0, 0.0, 0.0, covariance[1, 2],
            0.0,              0.0,              0.0, 0.0, 0.0, 0.0,
            0.0,              0.0,              0.0, 0.0, 0.0, 0.0,
            0.0,              0.0,              0.0, 0.0, 0.0, 0.0,
            covariance[2, 0], covariance[2, 1], 0.0, 0.0, 0.0, covariance[2, 2],
        ]
        publisher.publish(msg)

    def publish_vision_cone(self, cone_publisher: Publisher, rgb_stamp: Time):
        cone_publisher.publish(self.init_camera_cone(rgb_stamp))

    def init_camera_cone(self, rgb_stamp):
        cone_msg = Marker()
        cone_msg.header.frame_id = "kinect_depth"
        cone_msg.header.stamp = rgb_stamp
        cone_msg.ns = "vision_cone"
        cone_msg.id = 0
        cone_msg.type   = Marker.TRIANGLE_LIST
        cone_msg.action = Marker.ADD
        cone_msg.scale.x = cone_msg.scale.y = cone_msg.scale.z = 1.0
        cone_msg.color.r = 0.0
        cone_msg.color.g = 1.0
        cone_msg.color.b = 0.0
        cone_msg.color.a = 0.3

        h = tan(CAMERA_ANGLE_VER_RAD / 2)
        w = tan(CAMERA_ANGLE_HOR_RAD / 2)
        d = constants.parameters.max_depth / 1000

        def pt(x, y, z):
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = float(z)
            return p
        
        p0 = pt(0, 0, 0)
        p1 = pt( d*w,  d*h, d)
        p2 = pt(-d*w,  d*h, d)
        p3 = pt( d*w, -d*h, d)
        p4 = pt(-d*w, -d*h, d)

        cone_msg.points = [p0,p1,p3, p0,p4,p2, p0,p2,p1,
                           p0,p3,p4, p1,p2,p4, p4,p3,p1]
        return cone_msg

    def publish_pointcloud_map(self, publisher: Publisher, time: Time):
        self.visual_odom_map.publish_pointcloud_map(publisher, time)

    def get_drawn_keypoints(self):  return self.ransac_draw_keypoints
    def get_position(self) -> Coordinate: return self.pos_baselink
    def get_theta(self) -> float:   return self.theta
    def get_covariance_P(self) -> NDArray: return self.covariance_P
    def get_log_weight(self) -> float: return self.log_weight

    
