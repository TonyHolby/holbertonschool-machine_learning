#!/usr/bin/env python3
"""
    A script that implements a Variational Autoencoder (VAE).
"""
import tensorflow.keras as keras


class VAE(keras.layers.Layer):
    """
        A custom keras.Model that combines an encoder and a decoder, and
        overrides the call method to compute both the reconstruction loss
        and the KL divergence.
    """
    def __init__(self, input_dims, **kwargs):
        """
            Initializes the Variational Autoencoder (VAE).

            Args:
                encoder (keras.Model): The encoder model that maps input data
                    to the latent representation (z, z_mean, z_log_var).
                decoder (keras.Model): The decoder model that reconstructs
                    the input from the latent space.
                **kwargs: Additional keyword arguments passed to keras.Model.
        """
        super().__init__(**kwargs)
        self.input_dims = input_dims

    def call(self, inputs):
        """
            Encodes the inputs, decodes the latent and compute the losses.

            Args:
                inputs (tf.Tensor): the input data to be encoded and
                    reconstructed.

            Returns:
                The reconstructed input (decoder output).
        """
        x, reconstructed, z_mean, z_log_var = inputs
        reconstructed_loss = keras.losses.binary_crossentropy(x, reconstructed)
        reconstructed_loss *= self.input_dims
        kl_loss = 1 + z_log_var\
            - keras.backend.square(z_mean) - keras.backend.exp(z_log_var)
        kl_loss = -0.5 * keras.backend.sum(kl_loss, axis=-1)
        self.add_loss(keras.backend.mean(reconstructed_loss + kl_loss))

        return reconstructed


def sampling(args):
    """
        Perform the reparameterization trick to sample from the latent
        distribution.

        Args:
            inputs (tuple): A tuple (z_mean, z_log_var) where:
                - z_mean (Tensor): Mean vector of the latent Gaussian
                distribution.
                - z_log_var (Tensor): Log-variance vector of the latent
                Gaussian distribution.

        Returns:
            A latent vector z sampled from the Gaussian distribution
            defined by z_mean and z_log_var using the reparameterization
            trick.
    """
    z_mean, z_log_var = args
    batch = keras.backend.shape(z_mean)[0]
    dim = keras.backend.shape(z_mean)[1]
    epsilon = keras.backend.random_normal(shape=(batch, dim))

    return z_mean + keras.backend.exp(0.5 * z_log_var) * epsilon


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
        Creates a variational autoencoder.

        Args:
            input_dims (int): an integer containing the dimensions of the model
                input.
            hidden_layers (list): a list containing the number of nodes for
                each hidden layer in the encoder, respectively:
                    the hidden layers should be reversed for the decoder.
            latent_dims (int): an integer containing the dimensions of the
                latent space representation.

        Returns:
            encoder, decoder, auto:
                encoder is the encoder model, which should output the latent
                    representation, the mean, and the log variance.
                decoder is the decoder model.
                auto is the full autoencoder model.
    """
    inputs = keras.Input(shape=(input_dims,))
    x = inputs
    for units in hidden_layers:
        x = keras.layers.Dense(units, activation='relu')(x)
    z_mean = keras.layers.Dense(latent_dims,
                                activation=None,
                                name="z_mean")(x)
    z_log_var = keras.layers.Dense(latent_dims,
                                   activation=None,
                                   name="z_log_var")(x)
    z = keras.layers.Lambda(sampling,
                            output_shape=(latent_dims,),
                            name="z")([z_mean, z_log_var])
    encoder = keras.Model(inputs,
                          [z, z_mean, z_log_var],
                          name="encoder")

    latent_inputs = keras.Input(shape=(latent_dims,))
    x = latent_inputs
    for units in reversed(hidden_layers):
        x = keras.layers.Dense(units,
                               activation='relu')(x)
    outputs = keras.layers.Dense(input_dims,
                                 activation='sigmoid')(x)
    decoder = keras.Model(latent_inputs,
                          outputs,
                          name="decoder")

    z, z_mean, z_log_var = encoder(inputs)
    reconstructed = decoder(z)
    reconstructed = VAE(input_dims)([inputs, reconstructed, z_mean, z_log_var])

    auto = keras.Model(inputs, reconstructed, name="autoencoder")
    auto.compile(optimizer='adam')

    return encoder, decoder, auto
