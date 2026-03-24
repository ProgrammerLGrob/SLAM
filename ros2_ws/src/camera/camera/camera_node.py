#!/usr/bin/env python3

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher')
        
        # Publisher für Kamerabilder
        self.publisher_ = self.create_publisher(Image, 'camera/image_raw', 10)
        
        # Timer: 30 FPS (1/30 sekunden)
        self.timer = self.create_timer(1/30.0, self.timer_callback)
        
        # OpenCV Bridge
        self.bridge = CvBridge()
        
        # Kamera öffnen (0 = Standard-Kamera)
        # Versuche mit V4L2 Backend (besser für Linux)
        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        
        if not self.cap.isOpened():
            self.get_logger().warn("V4L2 Backend fehlgeschlagen, versuche Standard-Backend...")
            self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            self.get_logger().error("Kamera konnte nicht geöffnet werden!")
            self.get_logger().error("Überprüfe: ls -l /dev/video* und v4l2-ctl --list-devices")
            return
        
        # Kamera-Parameter setzen
        try:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception as e:
            self.get_logger().warn(f"Fehler beim Einstellen der Kamera-Eigenschaften: {e}")
        
        self.frame_count = 0
        self.get_logger().info("Kamera gestartet. Veröffentliche auf /camera/image_raw")

    def timer_callback(self):
        """Liest Frame von der Kamera und veröffentlicht ihn"""
        if not self.cap.isOpened():
            if self.frame_count % 30 == 0:  # Nur alle 30 Frames loggen
                self.get_logger().error("Kamera ist nicht geöffnet!")
            self.frame_count += 1
            return
        
        ret, frame = self.cap.read()
        
        if ret and frame is not None:
            try:
                # Konvertiere OpenCV-Format zu ROS-Message
                msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                self.publisher_.publish(msg)
                
                if self.frame_count % 30 == 0:  # Nur alle 30 Frames loggen
                    self.get_logger().debug(f"Frame veröffentlicht: {frame.shape}")
                self.frame_count += 1
            except Exception as e:
                self.get_logger().error(f"Fehler beim Konvertieren oder Veröffentlichen: {e}")
        else:
            if self.frame_count % 30 == 0:  # Nur alle 30 Frames loggen
                self.get_logger().warn("Fehler beim Lesen des Frames!")
            self.frame_count += 1

    def destroy_node(self):
        """Cleanup"""
        self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    camera_publisher = CameraPublisher()
    
    try:
        rclpy.spin(camera_publisher)
    except KeyboardInterrupt:
        print("\nProgramm unterbrochen...")
    finally:
        camera_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
