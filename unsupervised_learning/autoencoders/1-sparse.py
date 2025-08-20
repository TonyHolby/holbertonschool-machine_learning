#!/usr/bin/env python3
"""
    A function that creates a sparse autoencoder.
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """
        Creates a sparse autoencoder.

        Args:
            input_dims (int): an integer containing the dimensions of the model
                input.
            hidden_layers (list): a list containing the number of nodes for
                each hidden layer in the encoder, respectively:
                    the hidden layers should be reversed for the decoder.
            latent_dims (int): an integer containing the dimensions of the
                latent space representation.
            lambtha (float): the regularization parameter used for L1
                regularization on the encoded output.

        Returns:
            encoder, decoder, auto:
                encoder is the encoder model.
                decoder is the decoder model.
                auto is the sparse autoencoder model.
    """
    input_layer = keras.Input(shape=(input_dims,))
    x = input_layer
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation="relu")(x)

    latent = keras.layers.Dense(latent_dims,
                                activation="relu",
                                activity_regularizer=keras.regularizers.l1(
                                    lambtha))(x)
    encoder = keras.Model(inputs=input_layer,
                          outputs=latent,
                          name="encoder")

    latent_input = keras.Input(shape=(latent_dims,))
    x = latent_input
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation="relu")(x)

    output_layer = keras.layers.Dense(input_dims,
                                      activation="sigmoid")(x)
    decoder = keras.Model(inputs=latent_input,
                          outputs=output_layer,
                          name="decoder")

    auto_output = decoder(encoder(input_layer))
    auto = keras.Model(inputs=input_layer,
                       outputs=auto_output,
                       name="autoencoder")
    auto.compile(optimizer="adam",
                 loss="binary_crossentropy")

    return encoder, decoder, auto
