"""
    A class Yolo that performs tasks for neural style transfer.
"""
import tensorflow as tf
import numpy as np


class NST:
    """
        A class Yolo that performs tasks for neural style transfer.
    """

    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
            Constructor for NST.

            Args:
                style_image (np.ndarray): the preprocessed style image.
                content_image (np.ndarray): the preprocessed content image.
                alpha (float): the weight for content cost.
                beta (float): the weight for style cost.

            Return:
                The scaled image.
        """
        if (not isinstance(style_image, np.ndarray) or
                style_image.ndim != 3 or style_image.shape[2] != 3):
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")
        if (not isinstance(content_image, np.ndarray) or
                content_image.ndim != 3 or content_image.shape[2] != 3):
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def scale_image(image):
        """
            Rescales an image such that its pixels values are between 0 and 1
            and its largest side is 512 pixels.
        """
        if (not isinstance(image, np.ndarray) or
                image.ndim != 3 or image.shape[2] != 3):
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")

        h, w, _ = image.shape
        if h > w:
            new_h = 512
            new_w = int((w / h) * 512)
        else:
            new_w = 512
            new_h = int((h / w) * 512)

        image_resized = tf.image.resize(image,
                                        (new_h, new_w),
                                        method=tf.image.ResizeMethod.BICUBIC)

        image_rescaled = tf.cast(image_resized, tf.float32) / 255.0

        image_batched = tf.expand_dims(image_rescaled, axis=0)

        return image_batched
