#!/usr/bin/env python3
"""
    A function that builds  the DenseNet-121 architecture as described in
    Densely Connected Convolutional Networks.
"""
from tensorflow import keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
        Builds  the DenseNet-121 architecture as described in Densely
        Connected Convolutional Networks.

        Args:
            growth_rate (int): the growth rate.
            compression (float): the compression factor.

        Returns:
            The keras model.
    """
    he_normal = K.initializers.he_normal(seed=0)
    input_data = K.Input(shape=(224, 224, 3))

    X = K.layers.BatchNormalization()(input_data)
    X = K.layers.ReLU()(X)
    X = K.layers.Conv2D(64,
                        (7, 7),
                        strides=2, padding='same',
                        kernel_initializer=he_normal)(X)

    X = K.layers.MaxPooling2D(pool_size=(3, 3),
                              strides=2,
                              padding='same')(X)

    nb_filters = 64

    X, nb_filters = dense_block(X,
                                nb_filters,
                                growth_rate,
                                6)

    X, nb_filters = transition_layer(X,
                                     nb_filters,
                                     compression)

    X, nb_filters = dense_block(X,
                                nb_filters,
                                growth_rate,
                                12)

    X, nb_filters = transition_layer(X,
                                     nb_filters,
                                     compression)

    X, nb_filters = dense_block(X,
                                nb_filters,
                                growth_rate,
                                24)

    X, nb_filters = transition_layer(X,
                                     nb_filters,
                                     compression)

    X, nb_filters = dense_block(X,
                                nb_filters,
                                growth_rate,
                                16)

    X = K.layers.BatchNormalization()(X)
    X = K.layers.ReLU()(X)
    X = K.layers.GlobalAveragePooling2D()(X)
    outputs = K.layers.Dense(1000,
                             activation='softmax',
                             kernel_initializer=he_normal)(X)

    model = K.Model(inputs=input_data, outputs=outputs)

    return model
