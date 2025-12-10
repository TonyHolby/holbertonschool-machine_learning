#!/usr/bin/env python3
"""
    A function that randomly changes the brightness of an image.
"""
import tensorflow as tf


def change_brightness(image, max_delta):
    """
        Randomly changes the brightness of an image.

        Args:
            image (tf.Tensor): a 3D tf.Tensor containing the
                image to change.
            max_delta (float): the maximum amount the image should
                be brightened (or darkened).

        Returns:
            The altered image.
    """
    altered_image = tf.image.random_brightness(image, max_delta)

    return altered_image
