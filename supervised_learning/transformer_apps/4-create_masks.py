#!/usr/bin/env python3
"""
    A script that implements a class Dataset that loads and prepares a dataset
    for machine translation.
"""
import tensorflow_datasets as tfds
import transformers
import tensorflow as tf


class Dataset:
    """
        A class Dataset that loads and prepares a dataset for machine
        translation.
    """
    def __init__(self, batch_size, max_len):
        """
            Loads train and validation datasets, builds tokenizers,
            and encodes datasets into tokens.

            Args:
                batch_size (int): the batch size for training/validation.
                max_len (int): the maximum number of tokens allowed per example
                sentence.
        """
        self.batch_size = batch_size
        self.max_len = max_len

        data = tfds.load("ted_hrlr_translate/pt_to_en", as_supervised=True)
        train_raw = data['train']
        valid_raw = data['validation']

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(train_raw)

        train_encoded = train_raw.map(self.tf_encode,
                                      num_parallel_calls=tf.data.AUTOTUNE)
        valid_encoded = valid_raw.map(self.tf_encode,
                                      num_parallel_calls=tf.data.AUTOTUNE)

        self.data_train = (train_encoded
                           .filter(lambda pt, en:
                                   tf.logical_and(tf.size(pt) <= max_len,
                                                  tf.size(en) <= max_len))
                           .cache()
                           .shuffle(20000)
                           .padded_batch(batch_size,
                                         padded_shapes=([None], [None]))
                           .prefetch(tf.data.AUTOTUNE))

        self.data_valid = (valid_encoded
                           .filter(lambda pt, en:
                                   tf.logical_and(tf.size(pt) <= max_len,
                                                  tf.size(en) <= max_len))
                           .padded_batch(batch_size,
                                         padded_shapes=([None], [None])))

    def tokenize_dataset(self, data):
        """
            Creates subword tokenizers for the dataset.

            Args:
                data (tf.data.Dataset): a tf.data.Dataset whose examples are
                formatted as a tuple (pt, en):
                    pt is the tf.Tensor containing the Portuguese sentence.
                    en is the tf.Tensor containing the corresponding English
                    sentence.

            Returns:
                tokenizer_pt, tokenizer_en:
                    tokenizer_pt is the Portuguese tokenizer.
                    tokenizer_en is the English tokenizer.
        """
        def pt_generator():
            """
                Generates utf-8 sentences in Potuguese.
            """
            for pt, _ in tfds.as_numpy(data):
                yield pt.decode('utf-8')

        def en_generator():
            """
                Generates utf-8 sentences in English.
            """
            for _, en in tfds.as_numpy(data):
                yield en.decode('utf-8')

        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            "neuralmind/bert-base-portuguese-cased")
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            "bert-base-uncased")

        vocab_size = 2 ** 13

        tokenizer_pt = tokenizer_pt.train_new_from_iterator(
            pt_generator(), vocab_size=vocab_size)
        tokenizer_en = tokenizer_en.train_new_from_iterator(
            en_generator(), vocab_size=vocab_size)

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
            Encodes a translation into tokens.

            Args:
                pt (tf.Tensor): the tf.Tensor containing the Portuguese
                    sentence.
                en (tf.Tensor): the tf.Tensor containing the corresponding
                    English sentence.

            Returns:
                pt_tokens, en_tokens:
                    pt_tokens is a np.ndarray containing the Portuguese tokens.
                    en_tokens is a np.ndarray. containing the English tokens.
        """
        pt = pt.numpy().decode("utf-8")
        en = en.numpy().decode("utf-8")

        vocab_size_pt = self.tokenizer_pt.vocab_size
        vocab_size_en = self.tokenizer_en.vocab_size

        pt_tokens = [vocab_size_pt] + self.tokenizer_pt.encode(pt)\
            + [vocab_size_pt + 1]
        en_tokens = [vocab_size_en] + self.tokenizer_en.encode(en)\
            + [vocab_size_en + 1]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """
            TensorFlow wrapper for the encode() instance method.

            Args:
                pt (tf.Tensor): the tf.Tensor containing the Portuguese
                    sentence.
                en (tf.Tensor): the tf.Tensor containing the corresponding
                    English sentence.

            Returns:
                result_pt, result_en:
                    result_pt: a tf.Tensor containing the tokenized Portuguese
                    sentence.
                    result_en: a tf.Tensor containing the tokenized English
                    sentence.
        """
        result_pt, result_en = tf.py_function(func=self.encode,
                                              inp=[pt, en],
                                              Tout=[tf.int64, tf.int64])
        result_pt.set_shape([None])
        result_en.set_shape([None])

        return result_pt, result_en


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
