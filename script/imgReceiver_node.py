#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image

import cv2
from cv_bridge import CvBridge

class Imgreceiver(Node):
    def __init__(self):
        super().__init__('img_receiver')
        self.sub = self.create_subscription(Image, '/image_raw', self.Image_callback)
        #self.pub = self.create_publisher(Image, '/image_raw', 10)
    
    def Image_callback(self, data):
        self.get_logger().info('got images!!')
        pass

    def main():
        rclpy.init()
        img_receiver = Imgreceiver()
        try:
            rclpy.spin(img_receiver)
        except KeyboardInterrupt:
            pass
        rclpy.shutdown()
        

