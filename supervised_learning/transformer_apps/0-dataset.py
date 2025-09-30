#!/usr/bin/env python3
"""
    A script that implements a class Dataset that loads and prepares a dataset
    for machine translation.
"""
import tensorflow_datasets as tfds
import transformers


class Dataset:
    """
        A class Dataset that loads and prepares a dataset for machine
        translation.
    """
    def __init__(self):
        """
            Loads train and validation datasets and builds tokenizers.
        """
        data = tfds.load("ted_hrlr_translate/pt_to_en", as_supervised=True)
        self.data_train = data['train']
        self.data_valid = data['validation']
        self.tokenizer_pt, self.tokenizer_en =\
            self.tokenize_dataset(self.data_train)

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
        vocab_size = 2 ** 13

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

        tokenizer_pt =\
            tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
                pt_generator(), target_vocab_size=vocab_size)
        tokenizer_en =\
            tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
                en_generator(), target_vocab_size=vocab_size)

        return tokenizer_pt, tokenizer_en
