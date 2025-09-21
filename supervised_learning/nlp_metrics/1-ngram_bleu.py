#!/usr/bin/env python3
"""
    A function that calculates the n-gram BLEU score for a sentence.
"""
import numpy as np


def ngram_bleu(references, sentence, n):
    """
        Calculates the n-gram BLEU score for a sentence.

        Args:
            references (list): a list of reference translations:
                each reference translation is a list of the words in the
                translation.
            sentence (list): a list containing the model proposed sentence.
            n (int): the size of the n-gram to use for evaluation.

        Returns:
            The n-gram BLEU score.
    """
    def get_ngrams(words, n):
        ngrams = []
        for i in range(len(words) - n + 1):
            ngram = tuple(words[i:i+n])
            ngrams.append(ngram)

        return ngrams

    candidate_ngrams = get_ngrams(sentence, n)
    candidate_counts = {}
    for ngram in candidate_ngrams:
        candidate_counts[ngram] = candidate_counts.get(ngram, 0) + 1

    clipped_count = 0
    for ngram in candidate_counts:
        max_ref_count = max(
            ref.count(ngram)
            for ref in (get_ngrams(ref, n)
                        for ref in references))
        clipped_count += min(candidate_counts[ngram], max_ref_count)

    if candidate_ngrams:
        precision = clipped_count / len(candidate_ngrams)
    else:
        precision = 0

    ref_lens = []
    for ref in references:
        ref_lens.append(len(ref))

    candidate_len = len(sentence)
    closest_ref_len = min(
        ref_lens, key=lambda ref_len: (abs(ref_len - candidate_len), ref_len))

    if candidate_len > 0:
        if candidate_len > closest_ref_len:
            BP = 1
        else:
            BP = np.exp(1 - closest_ref_len / candidate_len)
    else:
        BP = 0

    bleu = BP * precision

    return bleu
