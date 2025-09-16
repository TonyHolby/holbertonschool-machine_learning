#!/usr/bin/env python3
"""
    A function that creates a bag of words embedding matrix.
"""
import numpy as np


def bag_of_words(sentences, vocab=None):
    """
        Creates a bag of words embedding matrix.

        Args:
            sentences (list): a list of sentences to analyze.
            vocab (list): a list of the vocabulary words to use for the
                analysis.

        Returns:
            embeddings, features:
                embeddings is a numpy.ndarray of shape (s, f) containing the
                    embeddings:
                        s is the number of sentences in sentences.
                        f is the number of features analyzed.
                features is a list of the features used for embeddings.
    """
    tokenized_sentences = []
    for sentence in sentences:
        sentence = sentence.lower()
        cleaned = "".join(character
                          if character.isalnum() or character in ["'", " "]
                          else " "
                          for character in sentence)
        tokens = cleaned.split()
        tokens = [tok[:-2] if tok.endswith("'s") else tok for tok in tokens]
        tokenized_sentences.append(tokens)

    if vocab is None:
        vocab_list = sorted(set(word
                                for tokens in tokenized_sentences
                                for word in tokens))
        vocab = np.array(vocab_list)
    else:
        vocab = list(vocab)

    features = vocab
    f = len(features)
    s = len(sentences)
    embeddings = np.zeros((s, f), dtype=int)
    word_index = {word: idx for idx, word in enumerate(features)}

    for i, tokens in enumerate(tokenized_sentences):
        for token in tokens:
            if token in word_index:
                embeddings[i, word_index[token]] += 1

    return embeddings, features
