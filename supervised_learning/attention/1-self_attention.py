#!/usr/bin/env python3
"""
    A script that implements a class SelfAttention that inherits from
    tensorflow.keras.layers.Layer to calculate the attention for machine
    translation.
"""
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """
        A class SelfAttention that inherits from tensorflow.keras.layers.Layer
        to calculate the attention for machine translation.
    """
    def __init__(self, units):
        """
            Initializes the attention layer.

            Args:
                units (int): the number of hidden units in the alignment model.
        """
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """
            Applies the attention mechanism.

            Args:
                s_prev (tf.Tensor): a tensor of shape (batch, units)
                    containing the previous decoder hidden state.
                hidden_states (tf.Tensor): a tensor of shape
                    (batch, input_seq_len, units) containing the outputs
                    of the encoder.

            Returns:
                context, weights:
                    context is a tensor of shape (batch, units) that
                        contains the context vector for the decoder.
                    weights is a tensor of shape (batch, input_seq_len, 1)
                        that contains the attention weights.
        """
        s_prev_expanded = tf.expand_dims(s_prev, axis=1)
        score = self.V(
            tf.nn.tanh(self.W(s_prev_expanded) + self.U(hidden_states)))
        weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(weights * hidden_states, axis=1)

        return context, weights
