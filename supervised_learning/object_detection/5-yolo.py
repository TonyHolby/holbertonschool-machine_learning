#!/usr/bin/env python3
"""
    A script that uses the Yolo v3 algorithm to perform object detection.
"""
import tensorflow.keras as K
import numpy as np
import os
import cv2


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

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
            Filters boxes using the objectness and class score thresholds.

            Args:
                boxes: list of np.ndarrays of shape (gh, gw, ab, 4).
                box_confidences: list of np.ndarrays of shape (gh, gw, ab, 1).
                box_class_probs: list of np.ndarrays of shape
                (gh, gw, ab, classes).

            Returns:
                A tuple of (filtered_boxes, box_classes, box_scores):
                    filtered_boxes: np.ndarray of shape (?, 4).
                    box_classes: np.ndarray of shape (?).
                    box_scores: np.ndarray of shape (?).
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for b, c, p in zip(boxes, box_confidences, box_class_probs):
            scores = c * p
            classes = np.argmax(scores, axis=-1)
            class_scores = np.max(scores, axis=-1)

            mask = class_scores >= self.class_t

            filtered_boxes.append(b[mask])
            box_classes.append(classes[mask])
            box_scores.append(class_scores[mask])

        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
            Applies Non-Max Suppression (NMS) on the filtered boxes.

            Args:
                filtered_boxes (np.ndarray): np.ndarray of shape (?, 4).
                box_classes (np.ndarray): np.ndarray of shape (?).
                box_scores (np.ndarray): np.ndarray of shape (?).

            Returns:
                A tuple of (box_predictions, predicted_box_classes,
                predicted_box_scores).
        """
        unique_classes = np.unique(box_classes)

        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        for classe in unique_classes:
            indexes = np.where(box_classes == classe)
            boxes = filtered_boxes[indexes]
            scores = box_scores[indexes]

            sorted_indexes = np.argsort(scores)[::-1]
            boxes = boxes[sorted_indexes]
            scores = scores[sorted_indexes]

            while len(boxes) > 0:
                best_box = boxes[0]
                best_score = scores[0]

                box_predictions.append(best_box)
                predicted_box_classes.append(classe)
                predicted_box_scores.append(best_score)

                x1 = np.maximum(best_box[0], boxes[1:, 0])
                y1 = np.maximum(best_box[1], boxes[1:, 1])
                x2 = np.minimum(best_box[2], boxes[1:, 2])
                y2 = np.minimum(best_box[3], boxes[1:, 3])

                inter_w = np.maximum(0, x2 - x1)
                inter_h = np.maximum(0, y2 - y1)
                inter_area = inter_w * inter_h

                box_area = (best_box[2] - best_box[0]) * (
                    best_box[3] - best_box[1])
                other_areas = (boxes[1:, 2] - boxes[1:, 0]) * (
                    boxes[1:, 3] - boxes[1:, 1])

                union_area = box_area + other_areas - inter_area
                iou = inter_area / union_area

                keep_idxs = np.where(iou <= self.nms_t)[0]
                boxes = boxes[1:][keep_idxs]
                scores = scores[1:][keep_idxs]

        return (np.array(box_predictions),
                np.array(predicted_box_classes),
                np.array(predicted_box_scores))

    @staticmethod
    def load_images(folder_path):
        """
            Loads all images from the specified folder.

            Args:
                folder_path (str): Path to the folder holding all the images to
                load.

            Returns:
                tuple:
                    - images (list of np.ndarray): Loaded image arrays.
                    - image_paths (list of str): Paths to the individual
                    images.
        """
        image_paths = []
        images = []

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                image = cv2.imread(file_path)
                if image is not None:
                    images.append(image)
                    image_paths.append(file_path)

        return images, image_paths

    def preprocess_images(self, images):
        """
            Preprocess a list of images for YOLO model input.

            Args:
                images (list of np.ndarray): a list of images as numpy arrays.

            Returns:
                A tuple of (pimages, image_shapes).
        """
        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]

        pimages = []
        image_shapes = []

        for image in images:
            image_height, image_width = image.shape[:2]
            image_shapes.append((image_height, image_width))
            resized = cv2.resize(image,
                                 (input_w, input_h),
                                 interpolation=cv2.INTER_CUBIC)
            resized = resized.astype(np.float32) / 255.0
            pimages.append(resized)

        pimages = np.array(pimages)
        image_shapes = np.array(image_shapes)

        return pimages, image_shapes
