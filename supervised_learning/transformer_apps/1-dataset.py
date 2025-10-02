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
