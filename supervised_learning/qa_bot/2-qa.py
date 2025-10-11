#!/usr/bin/env python3
"""
    A function that finds a snippet of text within a reference document
    to answer a question and interacts in a loop.
"""
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


qa_model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")
tokenizer = BertTokenizer.from_pretrained(
    "bert-large-uncased-whole-word-masking-finetuned-squad")


def question_answer(question, reference):
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
    inputs = tokenizer.encode_plus(question,
                                   reference,
                                   add_special_tokens=True,
                                   return_tensors="tf")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]

    outputs = qa_model([input_ids, attention_mask, token_type_ids])
    start_logits = outputs[0]
    end_logits = outputs[1]

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

    start_score = start_logits[start_idx]
    end_score = end_logits[end_idx]
    total_score = start_score + end_score

    if total_score < 0:
        return None

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
        total_score = best_score

    if total_score < 0:
        return None

    if start_idx > end_idx or start_idx == 0:
        return None

    all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].numpy())
    answer_tokens = all_tokens[start_idx:end_idx + 1]
    answer_text = tokenizer.convert_tokens_to_string(answer_tokens)
    chatbot_answer = answer_text.strip()

    if chatbot_answer:
        return chatbot_answer

    return None


def answer_loop(reference):
    """
        Interacts with the user to answers questions from a reference text.

        Args:
            reference (str): the reference text.
    """
    while True:
        question = input("Q: ").strip()
        if question.lower() in ("exit", "quit", "goodbye", "bye"):
            print("A: Goodbye")
            break

        answer = question_answer(question, reference)

        if answer is None:
            print("A: Sorry, I do not understand your question.")
        else:
            print(f"A: {answer}")
