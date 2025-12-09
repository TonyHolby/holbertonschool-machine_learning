#!/usr/bin/env python3
"""
    A function that randomly adjusts the contrast of an image.
"""
import tensorflow as tf


def change_contrast(image, lower, upper):
    """
        Randomly adjusts the contrast of an image.

        Args:
            image (tf.Tensor): a 3D tf.Tensor representing the
                input image to adjust the contrast.
            lower (float): a float representing the lower bound
                of the random contrast factor range.
            upper (float): a float representing the upper bound
                of the random contrast factor range.

        Returns:
            The contrast-adjusted image.
    """
    new_contrast = tf.image.random_contrast(image, lower, upper)

    return new_contrast
