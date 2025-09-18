#!/usr/bin/env python3
"""
    A function that converts a gensim word2vec model to a keras Embedding layer
"""
import tensorflow


def gensim_to_keras(model):
    """
        Converts a gensim word2vec model to a keras Embedding layer.

        Args:
            model (gensim.models.Word2Vec): a trained gensim word2vec models.

        Returns:
            The trainable keras Embedding.
    """
    vocab_size = len(model.wv)
    vector_size = model.vector_size
    weights = model.wv.vectors

    embedding_layer = tensorflow.keras.layers.Embedding(input_dim=vocab_size,
                                                        output_dim=vector_size,
                                                        weights=[weights],
                                                        trainable=True)

    return embedding_layer
