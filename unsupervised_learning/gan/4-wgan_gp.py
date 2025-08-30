#!/usr/bin/env python3
"""
    A script that implements a Wasserstein Generative Adversarial Network
    with Gradient Penalty (WGAN-GP) using methods to load pre-trained
    weights into the generator and discriminator generates fake faces.
"""
import tensorflow.keras as keras


class WGAN_GP:
    """
        A class that loads pre-trained weights into the generator and
        discriminator and generates fake faces.
    """
    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, batch_size=200, disc_iter=2,
                 learning_rate=0.0001):
        """
            Initializes the WGAN-GP model.

            Args:
                generator (keras.Model): the generator network.
                discriminator (keras.Model): the discriminator network.
                latent_generator (callable): a function that generates latent
                    vectors given a batch size.
                real_examples (tf.Tensor): a tensor containing real training
                    samples.
                batch_size (int, optional): the size of each training batch.
                disc_iter (int, optional): the number of discriminator updates
                    per generator update.
                learning_rate (float, optional): the learning rate for Adam
                    optimizers.
        """
        self.generator = generator
        self.discriminator = discriminator
        self.latent_generator = latent_generator
        self.real_ex = real_examples
        self.batch_size = batch_size
        self.disc_iter = disc_iter
        self.learning_rate = learning_rate
        self.g_optimizer = keras.optimizers.Adam(learning_rate)
        self.d_optimizer = keras.optimizers.Adam(learning_rate)

    def replace_weights(self, gen_h5, disc_h5):
        """
            Loads pre-trained weights into the generator and discriminator.

            Args:
                gen_h5 (str): the path to the .h5 file containing generator
                    weights.
                disc_h5 (str): the path to the .h5 file containing
                    discriminator weights.
        """
        self.generator.load_weights(gen_h5,
                                    by_name=True,
                                    skip_mismatch=True)
        self.discriminator.load_weights(disc_h5,
                                        by_name=True,
                                        skip_mismatch=True)

    def get_fake_sample(self, size):
        """
            Generates a batch of fake samples using the generator.

            Args:
                size (int): the number of fake samples to generate.

            Returns:
                A batch of generated fake samples.
        """
        if size is None:
            size = self.batch_size

        return self.generator(self.latent_generator(size), training=False)
