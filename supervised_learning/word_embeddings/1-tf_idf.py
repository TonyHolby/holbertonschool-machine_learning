#!/usr/bin/env python3
"""
    A function that creates a TF-IDF embedding.
"""
import numpy as np


def tf_idf(sentences, vocab=None):
    """
        Creates a TF-IDF embedding.

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
        vocab = np.array(vocab)

    features = vocab
    f = len(features)
    s = len(sentences)
    tf = np.zeros((s, f))
    word_index = {word: idx for idx, word in enumerate(features)}
    for i, tokens in enumerate(tokenized_sentences):
        for token in tokens:
            if token in word_index:
                j = word_index[token]
                tf[i, j] += 1

    df = np.zeros(f)
    for j, word in enumerate(features):
        df[j] = sum(1 for tokens in tokenized_sentences if word in tokens)

    idf = np.log((1 + s) / (1 + df)) + 1
    embeddings = tf * idf
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = embeddings / norms

    return embeddings, features
