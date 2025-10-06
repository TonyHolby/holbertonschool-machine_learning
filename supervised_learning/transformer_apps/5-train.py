#!/usr/bin/env python3
"""
    A script that creates and trains a transformer model for machine
    translation of Portuguese to English using a custom training loop.
"""
import tensorflow as tf
Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
        Implements the learning rate schedule from Vaswani et al. 2017.
    """
    def __init__(self, dm, warmup_steps=4000):
        """
            Initializes the custom learning rate schedule.

            Args:
                dm (int): the dimensionality of the model embeddings.
                warmup_steps (int): the number of warmup steps during which
                    the learning rate increases linearly.
        """
        super(CustomSchedule, self).__init__()
        self.dm = tf.cast(dm, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        """
            Computes the learning rate at a given step.

            Args:
                step (int or tf.Tensor): the current training step.

            Returns:
                The computed learning rate for this step.
        """
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        learning_rate = tf.math.rsqrt(self.dm) * tf.math.minimum(arg1, arg2)

        return learning_rate


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                            reduction='none')


def loss_function(real, pred):
    """
        Computes masked loss for a batch.

        Args:
            real (tf.Tensor): the true target sequences, shape
                (batch_size, seq_len).
            pred (tf.Tensor): the predicted logits, shape
                (batch_size, seq_len, vocab_size).

        Returns:
            The average loss for the batch, ignoring padding tokens.
    """
    mask = tf.cast(tf.not_equal(real, 0), tf.float32)
    per_token_loss = loss_object(real, pred) * mask

    return tf.reduce_sum(per_token_loss) / (tf.reduce_sum(mask) + 1e-9)


def accuracy_function(real, pred):
    """
        Computes masked accuracy for a batch.

        Args:
            real (tf.Tensor): the true target sequences, shape
                (batch_size, seq_len).
            pred (tf.Tensor): the predicted logits, shape
                (batch_size, seq_len, vocab_size).

        Returns:
            The accuracy of predictions, ignoring padding tokens.
    """
    pred_ids = tf.argmax(pred, axis=-1, output_type=tf.int32)
    real = tf.cast(real, tf.int32)
    mask = tf.cast(tf.not_equal(real, 0), tf.float32)
    matches = tf.cast(tf.equal(real, pred_ids), tf.float32) * mask

    return tf.reduce_sum(matches) / (tf.reduce_sum(mask) + 1e-9)


def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    """
        Creates the dataset, build the Transformer model, and train it.

        Args:
            N (int): the number of encoder/decoder layers.
            dm (int): the model dimensionality.
            h (int): the number of attention heads.
            hidden (int): the number of units in feed-forward network hidden
                layer.
            max_len (int): the maximum sequence length for input and target.
            batch_size (int): the batch size.
            epochs (int): the number of training epochs.

        Returns:
            The trained Transformer model.
    """
    dataset = Dataset(batch_size, max_len)

    input_vocab = dataset.input_vocab
    target_vocab = dataset.target_vocab

    transformer = Transformer(N, dm, h, hidden, input_vocab, target_vocab,
                              max_len)

    lr_schedule = CustomSchedule(dm)
    optimizer = tf.keras.optimizers.Adam(lr_schedule, beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)

    @tf.function
    def train_step(inp, tar):
        """
            Performs a single training step on a batch.

            Args:
                inp (tf.Tensor): the input batch (source sentences).
                tar (tf.Tensor): the target batch (shifted for decoder).

            Returns:
                The loss and accuracy for the batch.
        """
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        enc_mask, comb_mask, dec_mask = create_masks(inp, tar_inp)

        with tf.GradientTape() as tape:
            predictions = transformer(inp, tar_inp, True, enc_mask, comb_mask,
                                      dec_mask)
            loss = loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients,
                                      transformer.trainable_variables))
        acc = accuracy_function(tar_real, predictions)
        return loss, acc

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0

        for batch_num, (inp, tar) in enumerate(dataset.data_train, start=1):
            loss, acc = train_step(inp, tar)
            epoch_loss += loss
            epoch_acc += acc
            num_batches += 1

            if batch_num % 50 == 0:
                print(f"Epoch {epoch}, "
                      f"batch {batch_num}: loss {loss:.4f} accuracy {acc:.4f}")

        epoch_loss /= num_batches
        epoch_acc /= num_batches
        print(f"Epoch {epoch}: loss {epoch_loss:.4f} accuracy {epoch_acc:.4f}")

    return transformer
