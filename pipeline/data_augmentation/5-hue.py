#!/usr/bin/env python3
"""
    A function that changes the hue of an image.
"""
import tensorflow as tf


def change_hue(image, delta):
    """
        Changes the hue of an image.

        Args:
                image (tf.Tensor): a 3D tf.Tensor containing the
                    image to change.
                delta (float): the amount the hue should change.

            Returns:
                The altered image.
    """
    altered_image = tf.image.adjust_hue(image, delta)

    return altered_image
