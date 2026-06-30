"""Vocabulary: Keras tokenizer + GloVe embedding matrix.

Captions are lower-cased and wrapped with ``startseq`` / ``endseq`` markers
before tokenisation. The tokenizer keeps only the ``num_words`` most frequent
tokens; an ``OOV`` token covers the rest. A GloVe-100d matrix gives the decoder
a pre-trained embedding layer.

Mirrors the tokenisation and GloVe cells of the notebook.
"""
from __future__ import annotations

import pickle

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer

from .config import CONFIG, GLOVE_FILE

TOKENIZER_FILTERS = "!'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n"


def add_markers(caption: str) -> str:
    """``'A dog runs'`` -> ``'startseq a dog runs endseq'``."""
    return "startseq " + caption.lower().strip() + " endseq"


def build_tokenizer(captions, num_words: int = CONFIG.num_words) -> Tokenizer:
    """Fit a tokenizer on an iterable of (already lower-cased) caption strings."""
    tokenizer = Tokenizer(
        oov_token="OOV", filters=TOKENIZER_FILTERS, num_words=num_words
    )
    tokenizer.fit_on_texts(captions)
    return tokenizer


def save_tokenizer(tokenizer: Tokenizer, path: str) -> None:
    with open(path, "wb") as fh:
        pickle.dump(tokenizer, fh)


def load_tokenizer(path: str) -> Tokenizer:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def build_embedding_matrix(word_index: dict, glove_file: str = GLOVE_FILE,
                           num_words: int = CONFIG.num_words,
                           embedding_dim: int = CONFIG.embedding_dim) -> np.ndarray:
    """Map the vocabulary onto pre-trained GloVe vectors.

    Words missing from GloVe keep their all-zero row.
    """
    embeddings_index = {}
    with open(glove_file, encoding="utf-8") as fh:
        for line in fh:
            values = line.split()
            embeddings_index[values[0]] = np.asarray(values[1:], dtype="float32")

    matrix = np.zeros((num_words, embedding_dim))
    for word, i in word_index.items():
        if i >= num_words:
            continue
        vector = embeddings_index.get(word)
        if vector is not None:
            matrix[i] = vector
    return matrix
