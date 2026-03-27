#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from matplotlib import pyplot as plt

class ImageMatcher(Node):
    def __init__(self):
        super().__init__('image_matcher')
        self.bridge = CvBridge()
        # Initiate ORB detector
        self.orb = cv2.ORB_create()
        self.subscription_rgb = self.create_subscription(Image,'/serf01/nav_rgbd_1/rgb/image_raw', self.listener_rgb_callback, 10)
        self.subscription_rgb2 = self.create_subscription(Image,'/serf01/nav_rgbd_1/rgb/image_raw2', self.listener_rgb_callback2, 10)
        
    def listener_rgb_callback(self,msg):
        frame=self.bridge.imgmsg_to_cv2(msg,'bgr8')
        if not hasattr(self, 'frame2'):
            self.frame2=frame

        kp1, des1 = self.orb.detectAndCompute(frame,None)
        kp2, des2 = self.orb.detectAndCompute(self.frame2,None)

        #create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # Match descriptors.
        matches = bf.match(des1,des2)
        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)
        # Draw first 10 matches.
        img3 = cv2.drawMatches(frame,kp1,self.frame2,kp2,matches[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow("Brute force Matcher Image",img3)
        cv2.waitKey(1)

    def listener_rgb_callback2(self,msg):
        self.frame2=self.bridge.imgmsg_to_cv2(msg,'bgr8')
    

def main():
    rclpy.init()
    node=ImageMatcher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


