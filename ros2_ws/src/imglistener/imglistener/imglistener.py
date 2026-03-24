#!/usr/bin/env python3
import rclpy
import numpy as np

from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from matplotlib import pyplot as plt


class ImageSubscriber(Node):
	def __init__(self):
		super().__init__('image_subscriber')
		self.bridge = CvBridge()
		self.subscription_rgb = self.create_subscription(Image,'/serf01/nav_rgbd_1/rgb/image_raw', self.listener_callback_rgb, 10)
		self.subscription_depth = self.create_subscription(Image,'/serf01/nav_rgbd_1/depth/image_raw', self.listener_callback_depth, 10)
		self.orb = cv2.ORB_create()

	def listener_callback_rgb(self,msg):
		frame=self.bridge.imgmsg_to_cv2(msg,'bgr8')

		kp = self.orb.detect(frame,None)
		kp, des = self.orb.compute(frame, kp)
		img2 = cv2.drawKeypoints(frame, kp, None, color=(0,255,0), flags=0)
		

		#cv2.imshow("RGB",frame)
		cv2.imshow("Landmarks", img2)
		cv2.waitKey(1)

	def listener_callback_depth(self,msg):
		frame=self.bridge.imgmsg_to_cv2(msg, 'passthrough')
		#cv2.imshow("Depth", frame)
		cv2.waitKey(1)


def main():
	rclpy.init()
	node=ImageSubscriber()
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()

