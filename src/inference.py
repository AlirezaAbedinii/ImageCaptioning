"""Greedy caption generation for a single image.

This is the exact decoding loop used by the inference server: starting from
``startseq`` the decoder repeatedly predicts the most likely next token until it
emits ``endseq`` or hits ``max_length``.
"""
from __future__ import annotations

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

from .config import CONFIG


def generate_caption(tokenizer, picture: np.ndarray, model,
                     max_length: int = CONFIG.max_length) -> str:
    """Greedily decode a caption for one ``(1, 2048)`` image feature vector."""
    generated_text = "startseq"
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([generated_text])
        sequence = pad_sequences(sequence, maxlen=max_length)

        word_probs = model.predict([picture, sequence], verbose=0)
        predicted_index = [int(np.argmax(word_probs))]
        new_word = tokenizer.sequences_to_texts([predicted_index])[0]

        generated_text += " " + new_word
        if new_word == "endseq":
            break

    # Kept as in the notebook (including the leading space), so BLEU scores
    # computed on this output match the reported ones.
    return generated_text.replace("startseq", "").replace(" endseq", "")
