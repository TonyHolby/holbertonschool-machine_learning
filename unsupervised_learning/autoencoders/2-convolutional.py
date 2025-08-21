#!/usr/bin/env python3
"""
    A function that creates a convolutional autoencoder.
"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
        Creates a convolutional autoencoder.

        Args:
            input_dims (tuple): a tuple of integers containing the dimensions
                of the model input.
            filters (list): a list containing the number of filters for each
                convolutional layer in the encoder, respectively:
                    the filters should be reversed for the decoder.
            latent_dims (tuple): a tuple of integers containing the dimensions
                of the latent space representation.

        Returns:
            encoder, decoder, auto:
                encoder is the encoder model.
                decoder is the decoder model.
                auto is the full autoencoder model.
    """
    input_layer = keras.Input(shape=input_dims)
    x = input_layer
    for number_of_filters in filters:
        x = keras.layers.Conv2D(number_of_filters,
                                (3, 3),
                                padding="same",
                                activation="relu")(x)
        x = keras.layers.MaxPooling2D((2, 2),
                                      padding="same")(x)
    encoder = keras.Model(inputs=input_layer,
                          outputs=x,
                          name="encoder")

    latent_input = keras.Input(shape=latent_dims)
    x = latent_input
    for filters_index, number_of_filters in enumerate(reversed(filters)):
        if filters_index == (len(filters) - 1):
            padding_mode = "valid"
        else:
            padding_mode = "same"
        x = keras.layers.Conv2D(number_of_filters,
                                (3, 3),
                                padding=padding_mode,
                                activation="relu")(x)
        x = keras.layers.UpSampling2D((2, 2))(x)

    x = keras.layers.Conv2D(input_dims[-1],
                            (3, 3),
                            padding="same",
                            activation="sigmoid")(x)

    decoder = keras.Model(inputs=latent_input,
                          outputs=x,
                          name="decoder")

    auto_output = decoder(encoder(input_layer))
    auto = keras.Model(inputs=input_layer,
                       outputs=auto_output,
                       name="autoencoder")
    auto.compile(optimizer="adam",
                 loss="binary_crossentropy")

    return encoder, decoder, auto
