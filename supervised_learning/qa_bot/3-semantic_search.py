#!/usr/bin/env python3
"""
    A script that performs semantic search on a corpus of documents.
"""
import os
import re
import numpy as np
import tensorflow_hub as hub


def semantic_search(corpus_path, sentence):
    """
        Performs semantic search on a corpus of documents.

        Args:
            corpus_path (str): the path to the corpus of reference documents
                on which to perform semantic search.
            sentence (str): the sentence from which to perform semantic search.

        Returns:
            The reference text of the document most similar to sentence.
    """
    model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    documents = []

    for document in os.listdir(corpus_path):
        file_path = os.path.join(corpus_path, document)
        if os.path.isfile(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content.strip():
                        documents.append(content)
            except Exception:
                continue

    if not documents:
        return None

    def extract_keywords(text):
        """
            Extracts keywords from a given sentence by removing common stop
            words and keeping only words of at least three characters.

            Args:
                text (str): the input text string from which to extract
                keywords.

            Returns:
                A list of lowercase keywords extracted from the text.
        """
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        stop_words = {'the', 'are', 'and', 'for', 'with', 'this', 'that',
                      'from', 'have', 'has', 'was', 'were', 'will', 'would',
                      'can', 'could', 'should', 'may', 'might', 'must', 'what'}

        return [w for w in words if w not in stop_words]

    keywords = extract_keywords(sentence)

    keyword_scores = []
    for document in documents:
        text_lower = document.lower()
        doc_score = sum(1 for keyword in keywords if keyword in text_lower)
        keyword_scores.append(doc_score)

    keyword_scores = np.array(keyword_scores)

    if keyword_scores.max() > 0:
        candidates_indices = np.where(keyword_scores > 0)[0]

        if len(candidates_indices) > 0:
            candidates = [documents[i] for i in candidates_indices]
            all_texts = [sentence] + candidates
            embeddings = model(all_texts).numpy()
            sentence_embedding = embeddings[0:1]
            candidate_embeddings = embeddings[1:]
            sentence_norm = sentence_embedding / np.linalg.norm(
                sentence_embedding, axis=1, keepdims=True)
            candidates_norm = candidate_embeddings / np.linalg.norm(
                candidate_embeddings, axis=1, keepdims=True)
            similarities = np.dot(sentence_norm, candidates_norm.T)[0]
            normalized_keyword_scores =\
                keyword_scores[candidates_indices] / keyword_scores.max()
            combined_scores =\
                0.5 * normalized_keyword_scores + 0.5 * similarities
            best_candidate_idx = np.argmax(combined_scores)

            return candidates[best_candidate_idx]

    all_texts = [sentence] + documents
    embeddings = model(all_texts).numpy()
    sentence_embedding = embeddings[0:1]
    document_embeddings = embeddings[1:]
    sentence_norm = sentence_embedding / np.linalg.norm(
        sentence_embedding, axis=1, keepdims=True)
    documents_norm = document_embeddings / np.linalg.norm(
        document_embeddings, axis=1, keepdims=True)
    similarities = np.dot(sentence_norm, documents_norm.T)[0]
    most_similar_idx = np.argmax(similarities)

    return documents[most_similar_idx]
