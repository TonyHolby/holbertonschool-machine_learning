#!/usr/bin/env python3
"""
    A function that calculates the unigram BLEU score for a sentence.
"""


def uni_bleu(references, sentence):
    """
        Calculates the unigram BLEU score for a sentence.

        Args:
            references (list): a list of reference translations:
                each reference translation is a list of the words in the
                translation.
            sentence (list): a list containing the model proposed sentence.

        Returns:
            The unigram BLEU score.
    """
    word_counts = {}
    for word in sentence:
        word_counts[word] = word_counts.get(word, 0) + 1

    clipped_count = 0
    for word in word_counts:
        max_ref_count = max(ref.count(word) for ref in references)
        clipped_count += min(word_counts[word], max_ref_count)

    if sentence:
        precision = clipped_count / len(sentence)
    else:
        precision = 0

    ref_lens = [len(ref) for ref in references]
    candidate_len = len(sentence)
    closest_ref_len = min(
        ref_lens, key=lambda ref_len: (abs(ref_len - candidate_len), ref_len))

    if candidate_len > 0:
        if candidate_len > closest_ref_len:
            BP = 1
        else:
            BP = pow(2.718281828, 1 - closest_ref_len / candidate_len)
    else:
        BP = 0

    bleu = BP * precision
    return bleu
