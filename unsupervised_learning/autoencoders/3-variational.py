#!/usr/bin/env python3
"""
    A script that implements a Variational Autoencoder (VAE).
"""
import tensorflow.keras as keras


class Sampling(keras.layers.Layer):
    """
        A custom Keras layer that performs the reparameterization trick used
        in the Variational Autoencoder.
    """
    def call(self, inputs):
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
        z_mean, z_log_var = inputs
        batch = keras.backend.shape(z_mean)[0]
        dim = keras.backend.int_shape(z_mean)[1]
        epsilon = keras.backend.random_normal(shape=(batch, dim))

        return z_mean + keras.backend.exp(0.5 * z_log_var) * epsilon


class VAE(keras.Model):
    """
        A custom keras.Model that combines an encoder and a decoder, and
        overrides the call method to compute both the reconstruction loss
        and the KL divergence.
    """
    def __init__(self, encoder, decoder, **kwargs):
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
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        """
            Encodes the inputs, decodes the latent and compute the losses.

            Args:
                inputs (tf.Tensor): the input data to be encoded and
                    reconstructed.

            Returns:
                The reconstructed input (decoder output).
        """
        z, z_mean, z_log_var = self.encoder(inputs)
        recon = self.decoder(z)

        recon_loss = keras.losses.binary_crossentropy(inputs, recon)
        recon_loss = keras.backend.sum(recon_loss, axis=-1)

        kl_loss = -0.5 * keras.backend.sum(1 + z_log_var
                                           - keras.backend.square(z_mean)
                                           - keras.backend.exp(z_log_var),
                                           axis=-1)

        self.add_loss(keras.backend.mean(recon_loss + kl_loss))

        return recon


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
    encoder_inputs = keras.Input(shape=(input_dims,),
                                 name="encoder_input")
    x = encoder_inputs
    for i, units in enumerate(hidden_layers):
        x = keras.layers.Dense(units, activation="relu",
                               name=f"encoder_dense_{i+1}")(x)

    z_mean = keras.layers.Dense(latent_dims,
                                name="z_mean")(x)
    z_log_var = keras.layers.Dense(latent_dims,
                                   name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])

    encoder = keras.Model(encoder_inputs,
                          [z, z_mean, z_log_var],
                          name="encoder")

    decoder_inputs = keras.Input(shape=(latent_dims,),
                                 name="decoder_input")
    y = decoder_inputs
    for i, units in enumerate(reversed(hidden_layers)):
        y = keras.layers.Dense(units,
                               activation="relu",
                               name=f"decoder_dense_{i+1}")(y)
    decoder_outputs = keras.layers.Dense(input_dims,
                                         activation="sigmoid",
                                         name="reconstruction")(y)
    decoder = keras.Model(decoder_inputs,
                          decoder_outputs,
                          name="decoder")

    auto = VAE(encoder, decoder, name="autoencoder")
    auto.compile(optimizer="adam")

    return encoder, decoder, auto
