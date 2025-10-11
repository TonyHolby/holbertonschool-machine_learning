#!/usr/bin/env python3
"""
    A function that finds a snippet of text within a reference document
    to answer a question.
"""
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


qa_model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")
tokenizer = BertTokenizer.from_pretrained(
    "bert-large-uncased-whole-word-masking-finetuned-squad")


def question_answer(question: str, reference: str):
    """
        Finds a snippet of text within a reference document to answer a
        question.

        Args:
            question (str): a string containing the question to answer.
            reference (str): a string containing the reference document
                from which to find the answer.

        Returns:
            A string containing the answer.
    """
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
        return None

    context_start_index = context_start[0]

    start_logits[:context_start_index] = -float('inf')
    end_logits[:context_start_index] = -float('inf')

    start_idx = np.argmax(start_logits)
    end_idx = np.argmax(end_logits)

    if start_idx > end_idx:
        best_score = -float('inf')
        best_start = start_idx
        best_end = end_idx

        top_starts = np.argsort(start_logits)[-20:][::-1]
        top_ends = np.argsort(end_logits)[-20:][::-1]

        for start in top_starts:
            for end in top_ends:
                if start <= end and (end - start) < 30:
                    score = start_logits[start] + end_logits[end]
                    if score > best_score:
                        best_score = score
                        best_start = start
                        best_end = end

        start_idx = best_start
        end_idx = best_end

    if start_idx > end_idx or start_idx == 0:
        return None

    all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].numpy())
    answer_tokens = all_tokens[start_idx:end_idx + 1]
    answer_text = tokenizer.convert_tokens_to_string(answer_tokens)
    chatbot_answer = answer_text.strip()

    if chatbot_answer:
        return chatbot_answer

    return None
