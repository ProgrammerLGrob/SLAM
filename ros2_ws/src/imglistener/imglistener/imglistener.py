#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.bridge = CvBridge()
        # Initiate ORB detector
        self.orb = cv2.ORB_create()
        self.subscription_rgb = self.create_subscription(Image,'/serf01/nav_rgbd_1/rgb/image_raw', self.listener_rgb_callback, 10)
        self.subscription_depth = self.create_subscription(Image,'/serf01/nav_rgbd_1/depth/image_raw', self.listener_depth_callback, 10)

    def listener_rgb_callback(self,msg):
        frame=self.bridge.imgmsg_to_cv2(msg,'bgr8')
        #cv2.imshow("RGB Image",frame)

        # find the keypoints with ORB
        kp = self.orb.detect(frame,None)
        # compute the descriptors with ORB
        kp, des = self.orb.compute(frame, kp)
        # draw only keypoints location,not size and orientation
        img2 = cv2.drawKeypoints(frame, kp, None, color=(0,255,0), flags=0)
        cv2.imshow("ORB Image",img2)
        cv2.waitKey(1)

    def listener_depth_callback(self,msg):
        frame=self.bridge.imgmsg_to_cv2(msg,'passthrough')
        #cv2.imshow("Depth Image",frame)
        #cv2.waitKey(1)

def main():
    rclpy.init()
    node=ImageSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()