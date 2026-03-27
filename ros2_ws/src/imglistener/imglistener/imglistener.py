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
        self.frame_rgb=self.bridge.imgmsg_to_cv2(msg,'bgr8')
        #cv2.imshow("RGB Image",frame)

        # find the keypoints with ORB and compute the descriptors with ORB
        self.kp, des = self.orb.detectAndCompute(self.frame_rgb, None)
        

    def listener_depth_callback(self,msg):
        if not hasattr(self, 'frame_rgb'):
            return
        
        self.frame_depth=self.bridge.imgmsg_to_cv2(msg,'passthrough')

        # draw only keypoints location,not size and orientation
        img2 = cv2.drawKeypoints(self.frame_rgb, self.kp, None, color=(0,255,0), flags=0)
        img3 = img2.copy()

        for p in self.kp:
            x, y = p.pt
            x = round(x)
            y = round(y)
            depth_value = self.frame_depth[y, x]
            img3 = cv2.putText(img3, f"{depth_value/1000:.2f}m", (x+5, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 1, cv2.LINE_AA)

        cv2.imshow("Image with text", img3)
        cv2.waitKey(1)



def main():
    rclpy.init()
    node=ImageSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()