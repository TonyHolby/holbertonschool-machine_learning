#!/usr/bin/env python3
"""
    A class Yolo that uses the Yolo v3 algorithm to perform object detection.
"""
import tensorflow.keras as K
import numpy as np


class Yolo:
    """
        A class that Uses the Yolo v3 algorithm to perform object detection.
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
            Initialize the Yolo v3 algorithm to perform object detection.

            Args:
                model_path (str): the path to the Darknet Keras model.
                classes_path (str): the path to file containing class labels.
                class_t (float): a box score threshold for initial filtering.
                nms_t (float): a IOU threshold for non-max suppression.
                anchors (np.ndarray): anchor boxes (outputs, anchor_boxes, 2).
        """
        self.model = K.models.load_model(model_path)

        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]

        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
