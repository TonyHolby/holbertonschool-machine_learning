#!/usr/bin/env python3
"""
    A script that implements a Wasserstein Generative Adversarial Network
    (WGAN) with weight clipping.
"""
import tensorflow as tf
from tensorflow import keras


class WGAN_clip(keras.Model):
    """
        A class that combines a generator and a discriminator into a single
        Keras Model and updates them alternately.
    """
    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, batch_size=200, disc_iter=2,
                 learning_rate=0.005):
        """
            Initializes the WGAN model.

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
        super().__init__()
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        self.beta1 = 0.5
        self.beta2 = 0.9

        self.generator.loss = lambda fake_output: \
            -tf.math.reduce_mean(fake_output)
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta1, beta_2=self.beta2)
        self.generator.compile(optimizer=self.generator.optimizer,
                               loss=self.generator.loss)

        self.discriminator.loss = lambda real_output, fake_output: \
            tf.math.reduce_mean(fake_output) - tf.math.reduce_mean(real_output)
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta1, beta_2=self.beta2)
        self.discriminator.compile(optimizer=self.discriminator.optimizer,
                                   loss=self.discriminator.loss)

    def get_real_sample(self, size=None):
        """
            Samples a random batch of real examples from the dataset.

            Args:
                size (int, optional): the number of real samples to draw.

            Returns:
                A batch of real samples drawn from the dataset.
        """
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]

        return tf.gather(self.real_examples, random_indices)

    def get_fake_sample(self, size=None, training=False):
        """
            Generates a batch of fake samples using the generator.

            Args:
                size (int, optional): the number of fake samples to generate.
                training (bool, optional): Whether the generator is in
                    training mode.

            Returns:
                A batch of generated fake samples.
        """
        if size is None:
            size = self.batch_size

        return self.generator(self.latent_generator(size), training=training)

    def train_step(self, useless_argument):
        """
            Performs one training step of the WGAN with weight clipping.

            Args:
                useless_argument: a placeholder argument (batch data passed
                    by Keras "fit", unused).

            Returns:
                A dictionary containing discriminator and generator losses:
                {"discr_loss": tf.Tensor, "gen_loss": tf.Tensor}
        """
        for _ in range(self.disc_iter):
            with tf.GradientTape() as tape:
                real_samples = self.get_real_sample()
                fake_samples = self.get_fake_sample(training=True)
                real_output = self.discriminator(real_samples, training=True)
                fake_output = self.discriminator(fake_samples, training=True)
                discr_loss = self.discriminator.loss(real_output, fake_output)
            gradients = tape.gradient(discr_loss,
                                      self.discriminator.trainable_variables)
            self.discriminator.optimizer.apply_gradients(
                zip(gradients, self.discriminator.trainable_variables))

            for variable in self.discriminator.trainable_variables:
                variable.assign(tf.clip_by_value(variable, -1, 1))

        with tf.GradientTape() as tape:
            fake_samples = self.get_fake_sample(training=True)
            fake_output = self.discriminator(fake_samples, training=True)
            gen_loss = self.generator.loss(fake_output)
        grads = tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(
            zip(grads, self.generator.trainable_variables))

        return {"discr_loss": discr_loss, "gen_loss": gen_loss}
