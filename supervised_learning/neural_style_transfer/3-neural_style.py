#!/usr/bin/env python3
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
                style_image (np.ndarray): the  image used as a style reference,
                    stored as a numpy.ndarray.
                content_image (np.ndarray): the image used as a content
                    reference, stored as a numpy.ndarray.
                alpha (float): the weight for content cost.
                beta (float): the weight for style cost.
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
        self.model = self.load_model()
        self.generate_features()

    @staticmethod
    def scale_image(image):
        """
            Rescales an image such that its pixels values are between 0 and 1
            and its largest side is 512 pixels.

            Args:
                image (numpy.ndarray): A numpy.ndarray of shape (h, w, 3)
                containing the image to be scaled.

            Returns:
                The scaled image.
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

        image_resized = tf.clip_by_value(image_resized, 0, 255)

        image_rescaled = tf.cast(image_resized, tf.float32)

        image_bgr = image_rescaled[..., ::-1]

        image_bgr -= tf.constant([103.939, 116.779, 123.68], shape=[1, 1, 3])

        image_batched = tf.expand_dims(image_bgr, axis=0)

        return image_batched

    def load_model(self):
        """
            Loads the model for neural style transfer.
        """
        vgg = tf.keras.applications.VGG19(include_top=False,
                                          weights='imagenet')
        vgg.trainable = False

        outputs = {}
        x = vgg.input

        for layer in vgg.layers[1:]:
            if isinstance(layer, tf.keras.layers.MaxPooling2D):
                x = tf.keras.layers.AveragePooling2D(name=layer.name)(x)
            else:
                x = layer(x)
            outputs[layer.name] = x

        style_outputs = []
        for name in self.style_layers:
            output = outputs[name]
            style_outputs.append(output)

        content_layer_name = self.content_layer
        content_output = outputs[content_layer_name]

        self.model = tf.keras.Model(inputs=vgg.input,
                                    outputs=style_outputs + [content_output])

        return self.model

    @staticmethod
    def gram_matrix(input_layer):
        """
            Calculates the Gram matrix of a given layer's output.

            Args:
                input_layer (tf.Tensor ot tf.Variable): An instance of
                tf.Tensor or tf.Variable of shape (1, h, w, c) containing
                the layer output whose gram matrix should be calculated.

            Returns:
                A tf.Tensor of shape (1, c, c) containing the gram matrix of
                input_layer.
        """
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)
                          ) or input_layer.shape.rank != 4:
            raise TypeError("input_layer must be a tensor of rank 4")

        _, h, w, c = input_layer.shape
        features = tf.reshape(input_layer, shape=[-1, c])

        compute_gram = tf.matmul(features, features, transpose_a=True)

        gram_matrix = tf.expand_dims(compute_gram, axis=0)

        return gram_matrix

    def generate_features(self):
        """
            Extracts the style and content features.
        """
        style_outputs = self.model(self.style_image)
        style_layers_outputs = style_outputs[:len(self.style_layers)]

        self.style_features = style_layers_outputs
        self.gram_style_features = []
        for layer in style_layers_outputs:
            gram = self.gram_matrix(layer)
            self.gram_style_features.append(gram)

        content_output = self.model(self.content_image)[len(self.style_layers)]
        self.content_feature = content_output
