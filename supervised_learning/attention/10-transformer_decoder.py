#!/usr/bin/env python3
"""
    A script that implements a class Decoder that inherits from
    tensorflow.keras.layers.Layer to create the decoder for a transformer.
"""
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Decoder(tf.keras.layers.Layer):
    """
        A class Decoder that inherits from tensorflow.keras.layers.Layer
        to create the decoder for a transformer.
    """
    def __init__(self, N, dm, h, hidden, target_vocab, max_seq_len,
                 drop_rate=0.1):
        """
            Initializes the decoder.

            Args:
                N (int): the number of decoder blocks.
                dm (int): the dimensionality of the model.
                h (int): the number of heads.
                hidden (int): the number of hidden units in the fully
                    connected layer.
                target_vocab (int): the size of the target vocabulary.
                max_seq_len (int): the maximum sequence length possible.
                drop_rate (float): the dropout rate.
        """
        super(Decoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_dim=target_vocab,
                                                   output_dim=dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = []
        for _ in range(N):
            self.blocks.append(DecoderBlock(dm, h, hidden, drop_rate))

        self.dropout = tf.keras.layers.Dropout(rate=drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
            Forward pass of the decoder.

            Args:
                x (tf.Tensor): a tensor of shape (batch, target_seq_len, dm)
                    containing the input to the decoder.
                encoder_output (tf.Tensor): a tensor of shape
                    (batch, input_seq_len, dm)
                    containing the output of the encoder.
                training (bool): a boolean to determine if the model is
                    training.
                look_ahead_mask (tf.Tensor or None): the mask to be applied to
                    the first multi head attention layer.
                padding_mask (tf.Tensor or None): the mask to be applied to
                    the second multi head attention layer.

            Returns:
                A  tensor of shape (batch, target_seq_len, dm)
                containing the decoder output.
        """
        token_embeddings = self.embedding(x)
        token_embeddings *= tf.math.sqrt(tf.cast(self.dm, tf.float32))

        seq_len = tf.shape(x)[1]
        pos_embeddings = token_embeddings + self.positional_encoding[:seq_len]
        decoder_output = self.dropout(pos_embeddings, training=training)

        for decoder_block in self.blocks:
            decoder_output = decoder_block(decoder_output,
                                           encoder_output,
                                           training,
                                           look_ahead_mask,
                                           padding_mask)

        return decoder_output
