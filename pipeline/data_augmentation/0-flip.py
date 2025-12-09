#!/usr/bin/env python3
"""
    A function that flips an image horizontally.
"""
import tensorflow as tf


def flip_image(image):
    """
        Flips an image horizontally.

        Args:
            image (tf.Tensor): A 3D tf.Tensor
                containing the image to flip.

        Returns:
            The flipped image.
    """
    flipped_image = tf.image.flip_left_right(image)

    return flipped_image
