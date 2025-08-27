#!/usr/bin/env python3
"""
    A script that implements a Wasserstein Generative Adversarial Network with
    Gradient Penalty (WGAN-GP).
"""
import tensorflow as tf
from tensorflow import keras


class WGAN_GP(keras.Model):
    """
        A class that combines a generator and a discriminator into a single
        Keras Model and updates them alternately.
    """
    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, batch_size=200, disc_iter=2,
                 learning_rate=0.005, lambda_gp=10):
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
                lambda_gp (int): the gradient penalty coefficient.
        """
        super().__init__()
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        self.beta1 = 0.3
        self.beta2 = 0.9

        self.lambda_gp = lambda_gp
        self.dims = self.real_examples.shape
        self.len_dims = tf.size(self.dims)
        self.axis = tf.range(1, self.len_dims, delta=1, dtype='int32')
        self.scal_shape = self.dims.as_list()
        self.scal_shape[0] = self.batch_size
        for i in range(1, self.len_dims):
            self.scal_shape[i] = 1
        self.scal_shape = tf.convert_to_tensor(self.scal_shape)

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

    def get_interpolated_sample(self, real_sample, fake_sample):
        """
            Generates interpolated samples between real and fake data.

            Args:
                real_sample (tf.Tensor): A batch of real samples,
                    shape (batch_size, ...).
                fake_sample (tf.Tensor): A batch of fake samples,
                    shape (batch_size, ...).

            Returns:
                A batch of interpolated samples with the same shape as inputs.
        """
        u = tf.random.uniform(self.scal_shape)
        v = tf.ones(self.scal_shape) - u

        return u * real_sample + v * fake_sample

    def gradient_penalty(self, interpolated_sample):
        """
            Computes the gradient penalty.

            Args:
                interpolated_sample (tf.Tensor): A batch of interpolated
                    samples between real and fake data.

            Returns:
                A scalar tensor representing the gradient penalty value.
        """
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated_sample)
            pred = self.discriminator(interpolated_sample, training=True)
        grads = gp_tape.gradient(pred, [interpolated_sample])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=self.axis))

        return tf.reduce_mean((norm - 1.0) ** 2)

    def train_step(self, useless_argument):
        """
            Performs one training step of the WGAN-GP.

            Args:
                useless_argument: a placeholder argument (batch data passed
                    by Keras "fit", unused).

            Returns:
                A dictionary with discriminator loss, generator loss, and
                gradient penalty:
                    {
                        "discr_loss": tf.Tensor,
                        "gen_loss": tf.Tensor,
                        "gp": tf.Tensor
                    }
        """
        for _ in range(self.disc_iter):
            with tf.GradientTape() as tape:
                real_samples = self.get_real_sample()
                fake_samples = self.get_fake_sample(training=True)
                real_output = self.discriminator(real_samples, training=True)
                fake_output = self.discriminator(fake_samples, training=True)
                discr_loss = tf.reduce_mean(fake_output) \
                    - tf.reduce_mean(real_output)

                epsilon = tf.random.uniform([self.batch_size, 1], 0.0, 1.0)
                interpolated = epsilon * real_samples \
                    + (1 - epsilon) * fake_samples
                with tf.GradientTape() as gp_tape:
                    gp_tape.watch(interpolated)
                    pred = self.discriminator(interpolated, training=True)
                grads = gp_tape.gradient(pred, [interpolated])[0]
                grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1]))
                gp = tf.reduce_mean((grad_norm - 1.0) ** 2)

                new_discr_loss = discr_loss + self.lambda_gp * gp

            gradients = tape.gradient(
                new_discr_loss, self.discriminator.trainable_variables)
            self.discriminator.optimizer.apply_gradients(
                zip(gradients, self.discriminator.trainable_variables))

        with tf.GradientTape() as tape:
            fake_samples = self.get_fake_sample(training=True)
            fake_output = self.discriminator(fake_samples, training=True)
            gen_loss = -tf.reduce_mean(fake_output)

        grads = tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(
            zip(grads, self.generator.trainable_variables))

        return {"discr_loss": discr_loss, "gen_loss": gen_loss, "gp": gp}
