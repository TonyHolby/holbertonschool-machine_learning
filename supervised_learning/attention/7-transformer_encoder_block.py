#!/usr/bin/env python3
"""
    A script that implements a class EncoderBlock that inherits from
    tensorflow.keras.layers.Layer to create an encoder block for a transformer.
"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """
        A class EncoderBlock that inherits from tensorflow.keras.layers.Layer
        to create an encoder block for a transformer.
    """
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
            Initializes the encoder block.

            Args:
                dm (int): the dimensionality of the model.
                h (int): the number of heads.
                hidden (int): the number of hidden units in
                    the fully connected layer.
                drop_rate (float): the dropout rate.
        """
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(units=hidden,
                                                  activation="relu")
        self.dense_output = tf.keras.layers.Dense(units=dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate=drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(rate=drop_rate)

    def call(self, x, training, mask=None):
        """
            Forward pass of the encoder block.

            Args:
                x (tf.Tensor):  a tensor of shape (batch, input_seq_len, dm)
                    containing the input to the encoder block.
                training (bool): a boolean to determine if the model is
                    training.
                mask (tf.Tensor): the mask to be applied for multi head
                    attention.

            Returns:
                A tensor of shape (batch, input_seq_len, dm) containing
                the block's output.
        """
        attention_output, _ = self.mha(x, x, x, mask)
        attention_output = self.dropout1(attention_output, training=training)
        skip_connection = x + attention_output
        output_1 = self.layernorm1(skip_connection)
        ffn_hidden = self.dense_hidden(output_1)
        ffn_output = self.dense_output(ffn_hidden)
        ffn_output = self.dropout2(ffn_output, training=training)
        encoder_output = self.layernorm2(output_1 + ffn_output)

        return encoder_output
