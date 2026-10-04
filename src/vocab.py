"""Captions and vocabulary.

Every COCO caption is wrapped with ``startseq`` / ``endseq`` markers and grouped
per image id. A Keras ``Tokenizer`` (which lower-cases and strips punctuation)
keeps the ``num_words`` most frequent tokens; an ``OOV`` token covers the rest.

Mirrors the "Captions Processing" cells of the notebook. The word embedding is
learned by the decoder itself; no pre-trained (e.g. GloVe) vectors are used.
"""
from __future__ import annotations

import os
import pickle

from tensorflow.keras.preprocessing.text import Tokenizer

from .config import CONFIG
from .data import img_url_to_id

TOKENIZER_FILTERS = "!'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n"


def add_start_and_endseq(caption: str) -> str:
    """``'A dog runs'`` -> ``'startseq A dog runs endseq'``."""
    return "startseq " + caption + " endseq"


def remove_start_and_endseq(captions: list[str]) -> list[str]:
    captions = [c.replace("startseq ", "") for c in captions]
    return [c.replace(" endseq", "") for c in captions]


def create_captions_dic(coco_caps, dir_path: str) -> dict[int, list[str]]:
    """``{image_id: [marked captions]}`` for every image file in ``dir_path``."""
    captions_dic: dict[int, list[str]] = {}
    for img_url in os.listdir(dir_path):
        img_id = img_url_to_id(img_url)
        anns = coco_caps.loadAnns(coco_caps.getAnnIds(imgIds=img_id))
        for ann in anns:
            captions_dic.setdefault(ann["image_id"], []).append(
                add_start_and_endseq(ann["caption"])
            )
    return captions_dic


def flatten_captions(captions_dic: dict[int, list[str]]) -> list[str]:
    return [cap for caps in captions_dic.values() for cap in caps]


def build_tokenizer(captions, num_words: int = CONFIG.num_words) -> Tokenizer:
    """Fit a tokenizer on an iterable of caption strings."""
    tokenizer = Tokenizer(
        oov_token="OOV", filters=TOKENIZER_FILTERS, num_words=num_words
    )
    tokenizer.fit_on_texts(captions)
    return tokenizer


def save_tokenizer(tokenizer: Tokenizer, path: str) -> None:
    with open(path, "wb") as fh:
        pickle.dump(tokenizer, fh, protocol=pickle.HIGHEST_PROTOCOL)


def load_tokenizer(path: str) -> Tokenizer:
    with open(path, "rb") as fh:
        return pickle.load(fh)
