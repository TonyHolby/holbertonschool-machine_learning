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
        tokenizer_pt = transformers.BertTokenizer.from_pretrained(
            "neuralmind/bert-base-portuguese-cased")
        tokenizer_en = transformers.BertTokenizer.from_pretrained(
            "bert-base-uncased")

        return tokenizer_pt, tokenizer_en
