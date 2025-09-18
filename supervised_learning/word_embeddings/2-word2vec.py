#!/usr/bin/env python3
"""
    A function that creates, builds and trains a gensim word2vec model.
"""
import gensim


def word2vec_model(sentences, vector_size=100, min_count=5, window=5,
                   negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """
        Creates, builds and trains a gensim word2vec model.

        Args:
            sentences (list of list): a list of sentences to be trained on.
            vector_size (int): the dimensionality of the embedding layer.
            min_count (int): the minimum number of occurrences of a word for
                use in training.
            window (int): the maximum distance between the current and
                predicted word within a sentence.
            negative (int): the size of negative sampling.
            cbow (bool): a boolean to determine the training type.
                True (0) is for CBOW and False (1) is for Skip-gram.
            epochs (int): the number of iterations to train over.
            seed (int): the seed for the random number generator.
            workers (int): the number of worker threads to train the model.

        Returns:
            The trained model.
    """
    sg = 0 if cbow else 1

    w2v_model = gensim.models.Word2Vec(sentences=sentences,
                                       vector_size=vector_size,
                                       min_count=min_count,
                                       window=window,
                                       negative=negative,
                                       sg=sg,
                                       seed=seed,
                                       workers=workers,
                                       epochs=epochs,
                                       hs=0,
                                       sample=0)

    return w2v_model
