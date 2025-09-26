#!/usr/bin/env python3
"""
    A script that implements a class DecoderBlock that inherits from
    tensorflow.keras.layers.Layer to create a decoder block for a transformer.
"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """
        A class DecoderBlock that inherits from tensorflow.keras.layers.Layer
        to create a decoder block for a transformer.
    """
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
            Initializes the decoder block.

            Args:
                dm (int): the dimensionality of the model.
                h (int): the number of heads.
                hidden (int): the number of hidden units in
                    the fully connected layer.
                drop_rate (float): the dropout rate.
        """
        super(DecoderBlock, self).__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(units=hidden,
                                                  activation="relu")
        self.dense_output = tf.keras.layers.Dense(units=dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate=drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(rate=drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(rate=drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
            Forward pass of the decoder block.

            Args:
                x (tf.Tensor): a tensor of shape (batch, target_seq_len, dm)
                    containing the input to the decoder block.
                encoder_output (tf.Tensor): a tensor of shape (batch,
                    input_seq_len, dm) containing the output of the encoder.
                training (bool): a boolean to determine if the model is
                    training.
                look_ahead_mask (tf.Tensor): the mask to be applied to the
                    first multi head attention layer.
                padding_mask (tf.Tensor): the mask to be applied to the second
                    multi-head attention layer.

            Returns:
                A tensor of shape (batch, target_seq_len, dm) containing the
                block's output.
        """
        masked_mha_output, _ = self.mha1(x, x, x, look_ahead_mask)
        masked_mha_output = self.dropout1(masked_mha_output, training=training)
        skip_connection_1 = x + masked_mha_output
        layer_norm_1 = self.layernorm1(skip_connection_1)
        attention_output, _ = self.mha2(layer_norm_1, encoder_output,
                                        encoder_output, padding_mask)
        attention_output = self.dropout2(attention_output, training=training)
        skip_connection_2 = layer_norm_1 + attention_output
        layer_norm_2 = self.layernorm2(skip_connection_2)
        ffn_hidden = self.dense_hidden(layer_norm_2)
        ffn_output = self.dense_output(ffn_hidden)
        ffn_output = self.dropout3(ffn_output, training=training)
        decoder_output = self.layernorm3(layer_norm_2 + ffn_output)

        return decoder_output
