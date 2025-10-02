#!/usr/bin/env python3
"""
    A function that creates all masks for training/validation.
"""
import tensorflow as tf


def create_masks(inputs, target):
    """
        Creates all masks for training/validation.

        Args:
            inputs (tf.Tensor): a tf.Tensor of shape (batch_size,
                seq_len_in) that contains the input sentence.
            target (tf.Tensor): a tf.Tensor of shape (batch_size,
                seq_len_out) that contains the target sentence.

        Returns:
            encoder_mask, combined_mask, decoder_mask:
                encoder_mask is the tf.Tensor padding mask of shape
                    (batch_size, 1, 1, seq_len_in) to be applied in
                    the encoder.
                combined_mask is the tf.Tensor of shape
                    (batch_size, 1, seq_len_out, seq_len_out) used in
                    the 1st attention block in the decoder to pad and mask
                    future tokens in the input received by the decoder.
                decoder_mask is the tf.Tensor padding mask of shape
                    (batch_size, 1, 1, seq_len_in) used in the 2nd
                    attention block in the decoder.
    """
    def create_padding_mask(seq):
        """
            Creates a padding mask to ignore empty tokens (value = 0).

            Args:
                seq (tf.Tensor): a tf.Tensor of shape (batch_size, seq_len)
                representing a sequence of tokens.

            Returns:
                A Binary mask of shape (batch_size, 1, 1, seq_len),
                with 1s where "seq" contains padding tokens (0),
                and 0s elsewhere. The extra dimensions allow broadcasting
                in attention mechanisms.
        """
        mask = tf.cast(tf.math.equal(seq, 0), tf.float32)

        return mask[:, tf.newaxis, tf.newaxis, :]

    def create_look_ahead_mask(size):
        """
            Creates a look-ahead mask to prevent a position from attending
            to future tokens.

            Args:
                size (int): the length of the target sequence (seq_len_out).

            Returns:
                A Tensor of shape (size, size), upper triangular
                with 1s (masked positions) and 0s elsewhere.
        """
        return 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)

    encoder_mask = create_padding_mask(inputs)
    decoder_mask = create_padding_mask(inputs)

    seq_output_len = tf.shape(target)[1]
    look_ahead_mask = create_look_ahead_mask(seq_output_len)
    decoder_target_padding_mask = create_padding_mask(target)

    combined_mask = tf.maximum(decoder_target_padding_mask,
                               look_ahead_mask[tf.newaxis, tf.newaxis, :, :])

    return encoder_mask, combined_mask, decoder_mask
