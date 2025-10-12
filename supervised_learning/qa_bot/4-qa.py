#!/usr/bin/env python3
"""
    A script that answers questions from multiple reference texts
    and interacts with the user in a loop.
"""
import os
import re
import numpy as np
import tensorflow_hub as hub
from transformers import BertTokenizer


qa_model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")
tokenizer = BertTokenizer.from_pretrained(
    "bert-large-uncased-whole-word-masking-finetuned-squad")


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
    top_k = 3
    model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    documents, filenames = [], []
    for filename in os.listdir(corpus_path):
        filepath = os.path.join(corpus_path, filename)
        if os.path.isfile(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                    if content.strip():
                        documents.append(content)
                        filenames.append(filename)
            except Exception:
                continue

    if not documents:
        return None, None

    sentence = re.sub(r"[^\w\s]", "", sentence).strip()
    all_words = re.findall(r"\b[A-Za-z]+\b", sentence)
    stop_words = {
        "what", "when", "where", "who", "how", "why", "are", "is", "the",
        "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "up", "about", "into", "through"}
    keywords = [w.lower() for w in all_words
                if (w.isupper() and len(w) >= 2)
                or (len(w) >= 4 and w.lower() not in stop_words)]

    all_texts = [sentence] + documents
    embeddings = model(all_texts).numpy()

    sentence_embedding = embeddings[0:1]
    document_embeddings = embeddings[1:]

    sentence_norm = sentence_embedding / np.linalg.norm(
        sentence_embedding, axis=1, keepdims=True)
    documents_norm = document_embeddings / np.linalg.norm(
        document_embeddings, axis=1, keepdims=True)

    semantic_similarities = np.dot(sentence_norm, documents_norm.T)[0]

    keyword_scores = np.zeros(len(documents))
    for i, (doc, filename) in enumerate(zip(documents, filenames)):
        doc_lower, filename_lower = doc.lower(), filename.lower()
        content_score = sum(1 for kw in keywords if kw in doc_lower)
        filename_score = sum(2 for kw in keywords if kw in filename_lower)
        keyword_scores[i] = content_score + filename_score

    keyword_scores_norm = (
        keyword_scores / keyword_scores.max()
        if keyword_scores.max() > 0 else keyword_scores)

    combined_scores = (
        0.5 * keyword_scores_norm + 0.5 * semantic_similarities
        if keyword_scores.max() > 0
        else semantic_similarities)

    top_indices = np.argsort(combined_scores)[::-1][:top_k]
    top_documents = [documents[i] for i in top_indices]
    top_filenames = [filenames[i] for i in top_indices]

    return top_documents, top_filenames


def extract_answer(question, reference):
    """
            Extracts the most probable answer from a reference text given a
            question.

            Args:
                question (str): the question to be answered.
                reference (str): the reference text from which the answer
                    should be extracted.

            Returns:
                The extracted answer as a string, or None if no answer was
                found.
                The model's confidence score.
        """
    if not question or not reference:
        return (None, 0.0)

    tokenized_inputs = tokenizer.encode_plus(question,
                                             reference,
                                             add_special_tokens=True,
                                             return_tensors="tf")
    input_ids = tokenized_inputs["input_ids"]
    attention_mask = tokenized_inputs["attention_mask"]
    token_type_ids = tokenized_inputs["token_type_ids"]

    qa_model_outputs = qa_model([input_ids, attention_mask, token_type_ids])

    start_logits = qa_model_outputs[0]
    end_logits = qa_model_outputs[1]

    start_logits = start_logits.numpy()[0]
    end_logits = end_logits.numpy()[0]

    context_start = np.where(token_type_ids[0].numpy() == 1)[0]
    if len(context_start) == 0:
        return (None, 0.0)

    context_start_index = context_start[0]
    start_logits[:context_start_index] = -float("inf")
    end_logits[:context_start_index] = -float("inf")

    start_idx = np.argmax(start_logits)
    end_idx = np.argmax(end_logits)
    confidence_score = float(start_logits[start_idx] + end_logits[end_idx])

    if confidence_score < 0 or start_idx > end_idx or start_idx == 0:
        return (None, 0.0)

    all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].numpy())
    answer_tokens = all_tokens[start_idx:end_idx + 1]
    answer_text = tokenizer.convert_tokens_to_string(answer_tokens).strip()

    if answer_text:
        return (answer_text, confidence_score)
    else:
        return (None, 0.0)


def question_answer(corpus_path):
    """
        Interacts with the user in a loop, performs semantic search and
        answers questions from multiple reference texts.

        Args:
            corpus_path (str): the path to the corpus of reference documents
            on which to perform semantic search.
    """
    while True:
        question = input("Q: ").strip()
        if question.lower() in ("exit", "quit", "goodbye", "bye"):
            print("A: Goodbye")
            break

        references, filenames = semantic_search(corpus_path, question)
        if references is None:
            print("A: Sorry, I do not understand your question.")
            continue

        best_answer = None
        best_confidence = -float("inf")
        all_candidates = []

        for i, (reference, filename) in enumerate(zip(references, filenames)):
            answer, confidence = extract_answer(question, reference)

            if answer is not None:
                all_candidates.append((answer, confidence, filename, i + 1))

                if confidence > best_confidence:
                    best_confidence = confidence
                    best_answer = answer

        if best_answer is None:
            print("A: Sorry, I do not understand your question.")
        else:
            print(f"A: {best_answer}")
