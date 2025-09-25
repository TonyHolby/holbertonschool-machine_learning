#!/usr/bin/env python3
"""
    A script that implements a class MultiHeadAttention that inherits from
    tensorflow.keras.layers.Layer to perform multi-head attention.
"""
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """
        A class MultiHeadAttention that inherits from
        tensorflow.keras.layers.Layer to perform multi-head
        attention.
    """
    def __init__(self, dm, h):
        """
            Initializes the Multi-Head Attention layer.

            Args:
                dm (int): an integer representing the dimensionality
                    of the model. dm is divisible by h.
                h (int): an integer representing the number of heads.
        """
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(units=dm)
        self.Wk = tf.keras.layers.Dense(units=dm)
        self.Wv = tf.keras.layers.Dense(units=dm)
        self.linear = tf.keras.layers.Dense(units=dm)

    def call(self, Q, K, V, mask):
        """
            Applies multi-head attention.

            Args:
                Q (tf.Tensor): a tensor of shape (batch, seq_len_q, dk)
                    containing the input to generate the query matrix.
                K (tf.Tensor): a tensor of shape (batch, seq_len_v, dk)
                    containing the input to generate the key matrix.
                V (tf.Tensor): a tensor of shape (batch, seq_len_v, dv)
                    containing the input to generate the value matrix.
                mask is always None.

            Returns:
                output, weights:
                    outputa tensor with its last two dimensions as (...,
                    seq_len_q, dm) containing the scaled dot product attention.
                    weights a tensor with its last three dimensions as (..., h,
                    seq_len_q, seq_len_v) containing the attention weights.
        """
        batch_size = tf.shape(Q)[0]
        Q = self.Wq(Q)
        K = self.Wk(K)
        V = self.Wv(V)

        def split_heads(x):
            """
                Splits the last dimension into (h, depth) and transpose
                to shape (batch, h, seq_len, depth).

                Args:
                    x (tf.Tensor): a tensor of shape (batch, seq_len, dm).

                Returns:
                    A tensor of shape (batch, h, seq_len, depth).
            """
            x = tf.reshape(x, (batch_size, -1, self.h, self.depth))

            return tf.transpose(x, perm=[0, 2, 1, 3])

        Q = split_heads(Q)
        K = split_heads(K)
        V = split_heads(V)
        output, weights = sdp_attention(Q, K, V, mask)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        concat_output = tf.reshape(output, (batch_size, -1, self.dm))
        final_output = self.linear(concat_output)

        return final_output, weights
