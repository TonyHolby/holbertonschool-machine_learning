#!/usr/bin/env python3
"""
    A script that implements a transformer from Dataset and create_masks.
"""
import tensorflow as tf
Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks


def positional_encoding(max_seq_len, dm):
    """
        Calculates the positional encoding for a transformer.

        Args:
            max_seq_len (int): an integer representing the maximum sequence
                length.
            dm (int): the model depth.

        Returns:
            a numpy.ndarray of shape (max_seq_len, dm) containing the
            positional encoding vectors.
    """
    positions = tf.cast(tf.range(max_seq_len)[:, tf.newaxis], tf.float32)
    dims = tf.cast(tf.range(dm)[tf.newaxis, :], tf.float32)
    angle_rates = 1 / tf.pow(10000.0,
                             (2 * tf.math.floor(dims / 2))
                             / tf.cast(dm, tf.float32))
    angle_rads = positions * angle_rates
    sines = tf.math.sin(angle_rads[:, 0::2])
    coses = tf.math.cos(angle_rads[:, 1::2])
    pos_encoding = tf.concat([sines, coses], axis=-1)

    return pos_encoding


def sdp_attention(Q, K, V, mask=None):
    """
        Calculates the scaled dot product attention.

        Args:
            Q (tf.Tensor): a tensor with its last two dimensions as
                (..., seq_len_q, dk) containing the query matrix.
            K (tf.Tensor): a tensor with its last two dimensions as
                (..., seq_len_v, dk) containing the key matrix.
            V (tf.Tensor): a tensor with its last two dimensions as
                (..., seq_len_v, dv) containing the value matrix.
            mask (tf.Tensor): a tensor that can be broadcast into
                (..., seq_len_q, seq_len_v) containing the optional
                mask, or defaulted to None.

        Returns:
            output, weights:
                outputa tensor with its last two dimensions as (...,
                seq_len_q, dv) containing the scaled dot product attention.
                weights a tensor with its last two dimensions as (...,
                seq_len_q, seq_len_v) containing the attention weights.
    """
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    matmul_scores = tf.matmul(Q, K, transpose_b=True)
    scaled_scores = matmul_scores / tf.math.sqrt(dk)

    if mask is not None:
        scaled_scores += (mask * -1e9)

    weights = tf.nn.softmax(scaled_scores, axis=-1)
    output = tf.matmul(weights, V)

    return output, weights


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
        super().__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation="relu")
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

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


class Transformer(tf.keras.Model):
    """
        A class Transformer that inherits from tensorflow.keras.Model
        to create a transformer network.
    """
    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab, max_len,
                 drop_rate=0.1):
        """
            Initializes the Transformer.

            Args:
                N (int): the number of blocks in the encoder and decoder.
                dm (int): the dimensionality of the model.
                h (int): the number of heads.
                hidden (int): the number of hidden units in the fully
                    connected layers.
                input_vocab (int): the size of the input vocabulary.
                target_vocab (int): the size of the target vocabulary.
                max_seq_input (int): the maximum sequence length possible
                    for the input.
                max_seq_target (int): the maximum sequence length possible
                    for the target.
                drop_rate (float): the dropout rate.
        """
        super(Transformer, self).__init__()
        self.encoder = Encoder(N, dm, h, hidden, input_vocab, max_len,
                               drop_rate)
        self.decoder = Decoder(N, dm, h, hidden, target_vocab, max_len,
                               drop_rate)
        self.final_linear = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training, encoder_mask, look_ahead_mask,
             decoder_mask):
        """
            Forward pass of the Transformer.

            Args:
                inputs (tf.Tensor): a tensor of shape (batch, input_seq_len)
                    containing the inputs.
                target (tf.Tensor): a tensor of shape (batch, target_seq_len)
                    containing the target.
                training (bool): a boolean to determine if the model is
                    training.
                encoder_mask (tf.Tensor): the padding mask to be applied
                    to the encoder.
                look_ahead_mask (tf.Tensor): the look ahead mask to be
                    applied to the decoder.
                decoder_mask (tf.Tensor): the padding mask to be applied
                    to the decoder.

            Returns:
                A tensor of shape (batch, target_seq_len, target_vocab)
                containing the transformer output.
        """
        encoder_output = self.encoder(inputs, training=training,
                                      mask=encoder_mask)
        decoder_output = self.decoder(target, encoder_output, training,
                                      look_ahead_mask, decoder_mask)
        final_output = self.final_linear(decoder_output)

        return final_output
