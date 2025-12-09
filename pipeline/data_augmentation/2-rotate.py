#!/usr/bin/env python3
"""
    A function that rotates an image by 90 degrees counter-clockwise.
"""
import tensorflow as tf


def rotate_image(image):
    """
        Rotates an image by 90 degrees counter-clockwise.

        Args:
            image (tf.Tensor): a 3D tf.Tensor
                containing the image to rotate.

        Returns:
            The rotated image.
    """
    rotated_image = tf.image.rot90(image, k=1)

    return rotated_image
