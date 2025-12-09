#!/usr/bin/env python3
"""
    A function that performs a random crop of an image.
"""
import tensorflow as tf


def crop_image(image, size):
    """
        Performs a random crop of an image.

        Args:
            image (tf.Tensor): a 3D tf.Tensor
                containing the image to crop.
            size (tuple): a tuple containing
                the size of the crop.

        Returns:
            The cropped image.
    """
    channels = tf.shape(image)[2]
    crop_shape = [size[0], size[1], channels]
    cropped_image = tf.image.random_crop(image, crop_shape)

    return cropped_image
