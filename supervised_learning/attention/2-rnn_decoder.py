#!/usr/bin/env python3
"""
    A script that implements a class RNNDecoder that inherits from
    tensorflow.keras.layers.Layer to decode for machine translation.
"""
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """
        A class RNNDecoder that inherits from tensorflow.keras.layers.Layer
        to decode for machine translation.
    """
    def __init__(self, vocab, embedding, units, batch):
        """
            Initializes the RNNDecoder.

            Args:
                vocab (int): an integer representing the size of the output
                    vocabulary.
                embedding (int): an integer representing the dimensionality of
                    the embedding vector.
                units (int): an integer representing the number of hidden units
                    in the RNN cell.
                batch (int): an integer representing the batch size.
        """
        super(RNNDecoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab,
                                                   output_dim=embedding)
        self.gru = tf.keras.layers.GRU(units=units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer="glorot_uniform")
        self.F = tf.keras.layers.Dense(vocab)
        self.units = units

    def call(self, x, s_prev, hidden_states):
        """
            Performs one decoding step with attention.

            Args:
                x (tf.Tensor): a tensor of shape (batch, 1) containing the
                    previous word in the target sequence as an index of the
                    target vocabulary.
                s_prev (tf.Tensor): a tensor of shape (batch, units) containing
                    the previous decoder hidden state.
                hidden_states (tf.Tensor): a tensor of shape(batch,
                    input_seq_len, units) containing the outputs of the
                    encoder.

            Returns:
                y, s:
                    y is a tensor of shape (batch, vocab) containing the output
                        word as a one hot vector in the target vocabulary.
                    s is a tensor of shape (batch, units) containing the new
                        decoder hidden state.
        """
        attention = SelfAttention(self.units)
        context, _ = attention(s_prev, hidden_states)
        x = self.embedding(x)
        context = tf.expand_dims(context, 1)
        x_concat = tf.concat([context, x], axis=-1)
        outputs, s = self.gru(x_concat, initial_state=s_prev)
        y = self.F(outputs)
        y = tf.squeeze(y, axis=1)

        return y, s
