#!/usr/bin/env python3

import cv_bridge
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from ultralytics_ros.msg import YoloResult

import cv2
from ultralytics import YOLO


class DetectNode(Node):
    def __init__(self):
        super().__init__("detect_node")
        #predict parameter
        self.declare_parameter("yolo_model", "yolov8x.pt")
        self.declare_parameter("input_topic", "image_raw")
        self.declare_parameter("result_topic", "yolo_result")
        self.declare_parameter("result_image_topic", "yolo_image")
        self.declare_parameter("conf_thres", 0.25)
        self.declare_parameter("iou_thres", 0.45)
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("max_det", 300)
        self.declare_parameter("device", "0")
        self.declare_parameter("result_conf", True)
        self.declare_parameter("result_line_width", 1)
        self.declare_parameter("result_font_size", 1)
        self.declare_parameter("result_font", "Arial.ttf")
        self.declare_parameter("result_labels", True)
        self.declare_parameter("result_boxes", True)

        yolo_model = self.get_parameter("yolo_model").get_parameter_value().string_value
        self.model = YOLO(yolo_model)
        self.model.fuse()
        self.bridge = cv_bridge.CvBridge()

        input_topic = (self.get_parameter("input_topic").get_parameter_value().string_value)
        result_topic = (self.get_parameter("result_topic").get_parameter_value().string_value)
        result_image_topic = (self.get_parameter("result_image_topic").get_parameter_value().string_value)
        self.create_subscription(Image, input_topic, self.Image_callback, 100)
        self.result_pub = self.create_publisher(YoloResult, result_topic, 100)
        self.result_image_pub = self.create_publisher(Image, result_image_topic, 100)

    def Image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        conf_thres = self.get_parameter("conf_thres").get_parameter_value().double_value
        iou_thres = self.get_parameter("iou_thres").get_parameter_value().double_value
        max_det = self.get_parameter("max_det").get_parameter_value().integer_value
        imgsz = self.get_parameter("imgsz").get_parameter_value().integer_value
        device = self.get_parameter("device").get_parameter_value().string_value or None
        results = self.model.track(
            source=cv_image,
            conf=conf_thres,
            iou=iou_thres,
            max_det=max_det,
            device=device,
        )

        if results is not None:
            yolo_result_msg = YoloResult()
            yolo_result_image_msg = Image()
            yolo_result_msg.header = msg.header
            yolo_result_image_msg.header = msg.header
            yolo_result_msg.detections = self.create_detections_array(results)
            yolo_result_image_msg = self.create_result_image(results)
            self.result_pub.publish(yolo_result_msg)
            self.result_image_pub.publish(yolo_result_image_msg)

    def create_detections_array(self, results):
        detections_msg = Detection2DArray()
        bounding_box = results[0].boxes.xywh
        classes = results[0].boxes.cls
        confidence_score = results[0].boxes.conf
        for bbox, cls, conf in zip(bounding_box, classes, confidence_score):
            detection = Detection2D()
            detection.bbox.center.position.x = float(bbox[0])
            detection.bbox.center.position.y = float(bbox[1])
            detection.bbox.size_x = float(bbox[2])
            detection.bbox.size_y = float(bbox[3])
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = results[0].names.get(int(cls))
            hypothesis.hypothesis.score = float(conf)
            detection.results.append(hypothesis)
            detections_msg.detections.append(detection)
        return detections_msg
    
    def create_result_image(self, results):
        result_conf = self.get_parameter("result_conf").get_parameter_value().bool_value
        result_line_width = (self.get_parameter("result_line_width").get_parameter_value().integer_value)
        result_font_size = (self.get_parameter("result_font_size").get_parameter_value().integer_value)
        result_font = (self.get_parameter("result_font").get_parameter_value().string_value)
        result_labels = (self.get_parameter("result_labels").get_parameter_value().bool_value)
        result_boxes = (self.get_parameter("result_boxes").get_parameter_value().bool_value)
        anotated_frame = results[0].plot(
            conf=result_conf,
            line_width=result_line_width,
            font_size=result_font_size,
            font=result_font,
            labels=result_labels,
            boxes=result_boxes,
        )
        result_image_msg = self.bridge.cv2_to_imgmsg(anotated_frame, encoding="bgr8")
        return result_image_msg


def main(args=None):
    rclpy.init(args=args)
    node = DetectNode()
    rclpy.spin(node)


if __name__ == "__main__":
    main()