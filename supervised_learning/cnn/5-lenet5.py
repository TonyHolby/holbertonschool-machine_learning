#!/usr/bin/env python3
"""
    A function that builds a modified version of the LeNet-5 architecture
    using keras.
"""
from tensorflow import keras as K


def lenet5(X):
    """
        Builds a modified version of the LeNet-5 architecture using keras.

        Args:
            X (np.ndarray): a K.Input of shape (m, 28, 28, 1) containing
            the input images for the network:
                m is the number of images.

        Returns:
            A K.Model compiled to use Adam optimization
            (with default hyperparameters) and accuracy metrics.
    """
    he_normal = K.initializers.HeNormal(seed=0)

    conv1 = K.layers.Conv2D(filters=6,
                            kernel_size=(5, 5),
                            padding='same',
                            activation='relu',
                            kernel_initializer=he_normal)(X)

    pool1 = K.layers.MaxPooling2D(pool_size=(2, 2),
                                  strides=(2, 2))(conv1)

    conv2 = K.layers.Conv2D(filters=16,
                            kernel_size=(5, 5),
                            padding='valid',
                            activation='relu',
                            kernel_initializer=he_normal)(pool1)

    pool2 = K.layers.MaxPooling2D(pool_size=(2, 2),
                                  strides=(2, 2))(conv2)

    flatten = K.layers.Flatten()(pool2)

    fc1 = K.layers.Dense(units=120,
                         activation='relu',
                         kernel_initializer=he_normal)(flatten)

    fc2 = K.layers.Dense(units=84,
                         activation='relu',
                         kernel_initializer=he_normal)(fc1)

    fc_with_softmax_output = K.layers.Dense(units=10,
                                            activation="softmax",
                                            kernel_initializer=he_normal)(fc2)

    model = K.Model(inputs=X,
                    outputs=fc_with_softmax_output)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
