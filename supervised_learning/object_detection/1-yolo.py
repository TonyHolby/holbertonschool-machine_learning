#!/usr/bin/env python3
"""
    A script that uses the Yolo v3 algorithm to perform object detection.
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

    def process_outputs(self, outputs, image_size):
        """
            Process the outputs from the YOLO model.

            Args:
                outputs (list of np.ndarray): a list of numpy.ndarrays with
                shape (gh, gw, ab, 4 + 1 + classes).
                image_size (np.ndarray): a numpy.ndarray  containing the
                image's original size [image_height, image_width].

            Returns:
                boxes: list of np.ndarrays with shape (gh, gw, ab, 4)
                box_confidences: list of np.ndarrays with shape (gh, gw, ab, 1)
                box_class_probs: list of np.ndarrays with shape
                (gh, gw, ab, classes)
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]
        image_h, image_w = image_size

        for i, output in enumerate(outputs):
            if output.ndim == 5 and output.shape[0] == 1:
                output = output[0]

            grid_h, grid_w, anchor_boxes, _ = output.shape
            anchors = self.anchors[i]

            box = output[..., 0:4]
            tx = box[..., 0]
            ty = box[..., 1]
            tw = box[..., 2]
            th = box[..., 3]

            cx = np.arange(grid_w)
            cy = np.arange(grid_h)
            cx_grid, cy_grid = np.meshgrid(cx, cy)
            cx_grid = np.tile(cx_grid[..., np.newaxis], (1, 1, anchor_boxes))
            cy_grid = np.tile(cy_grid[..., np.newaxis], (1, 1, anchor_boxes))

            bx = 1 / (1 + np.exp(-tx)) + cx_grid
            by = 1 / (1 + np.exp(-ty)) + cy_grid
            bw = np.exp(tw) * anchors[np.newaxis, np.newaxis, :, 0]
            bh = np.exp(th) * anchors[np.newaxis, np.newaxis, :, 1]

            bx /= grid_w
            by /= grid_h
            bw /= input_w
            bh /= input_h

            x1 = (bx - bw / 2) * image_w
            y1 = (by - bh / 2) * image_h
            x2 = (bx + bw / 2) * image_w
            y2 = (by + bh / 2) * image_h

            boxes.append(np.stack([x1, y1, x2, y2], axis=-1))

            box_conf = 1 / (1 + np.exp(-output[..., 4]))
            box_conf = box_conf[..., np.newaxis]
            box_confidences.append(box_conf)

            class_probs = 1 / (1 + np.exp(-output[..., 5:]))
            box_class_probs.append(class_probs)

        return boxes, box_confidences, box_class_probs
