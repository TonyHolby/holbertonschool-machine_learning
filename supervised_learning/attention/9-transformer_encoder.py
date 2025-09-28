#!/usr/bin/env python3
"""
    A script that implements a class Encoder that inherits from
    tensorflow.keras.layers.Layer to create the encoder for a transformer.
"""
import tensorflow as tf
import numpy as np
positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    """
        A class Encoder that inherits from tensorflow.keras.layers.Layer
        to create the encoder for a transformer.
    """
    def __init__(self, N, dm, h, hidden, input_vocab, max_seq_len,
                 drop_rate=0.1):
        """
            Initializes the encoder.

            Args:
                N (int): the number of blocks in the encoder.
                dm (int): the dimensionality of the model.
                h (int): the number of heads.
                hidden (int): the number of hidden units in the fully
                    connected layer.
                input_vocab (int): the size of the input vocabulary.
                max_seq_len (int): the maximum sequence length possible.
                drop_rate (float): the dropout rate.
        """
        super(Encoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_dim=input_vocab,
                                                   output_dim=dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = []
        for _ in range(N):
            self.blocks.append(EncoderBlock(dm, h, hidden, drop_rate))
        self.dropout = tf.keras.layers.Dropout(rate=drop_rate)

    def call(self, x, training, mask):
        """
            Forward pass of the encoder.

            Args:
                x (tf.Tensor): a tensor of shape (batch, input_seq_len, dm)
                    containing the input to the encoder.
                training (bool):  a boolean to determine if the model is
                    training.
                mask (tf.Tensor): the mask to be applied for multi-head
                    attention.

            Returns:
                A tensor of shape (batch, input_seq_len, dm) containing the
                encoder output.
        """
        token_embeddings = self.embedding(x)
        token_embeddings *= tf.math.sqrt(tf.cast(self.dm, tf.float32))

        seq_len = tf.shape(x)[1]
        pos_embeddings = token_embeddings + self.positional_encoding[:seq_len]
        encoder_output = self.dropout(pos_embeddings, training=training)

        for encoder_block in self.blocks:
            encoder_output = encoder_block(encoder_output, training, mask)

        return encoder_output
