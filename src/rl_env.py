"""Reinforcement-learning environment for caption generation.

``ICEnv`` follows the OpenAI Gym interface and turns caption generation into a
sequential decision problem, so the decoder can be refined with SCST. Ported
from the "Reinforcement Learning" section of the notebook.

* **state**  — ``{"image": 2048-d features, "words": token ids, zero-padded}``
* **action** — the next vocabulary index (``Discrete(num_words, start=1)``)
* **reward** — BLEU of the caption built so far against the image's training
  captions (see ``src/metrics.py`` for how BLEU is computed).
"""
from __future__ import annotations

import numpy as np
from gym import Env, spaces

from .config import CONFIG
from .metrics import compute_bleu


class ICEnv(Env):
    """Image-Captioning environment.

    ``train_caps`` is the ``{image_id: [marked captions]}`` dictionary built by
    ``vocab.create_captions_dic``.
    """

    def __init__(self, image, img_id, tokenizer, train_caps,
                 max_len: int = CONFIG.max_length,
                 num_words: int = CONFIG.num_words,
                 startseq: int | None = None, endseq: int | None = None):
        self.prev_rewards = []
        self.reward = 0
        self.total_rewards = 0
        self.max_len = max_len
        self.num_words = num_words
        self.startseq = startseq if startseq is not None else tokenizer.word_index["startseq"]
        self.endseq = endseq if endseq is not None else tokenizer.word_index["endseq"]
        self.done = False
        self.img_id = img_id
        self.tokenizer = tokenizer
        self.train_caps = train_caps
        self.seq_len = 1

        self.observation_space = {
            "image": spaces.Box(low=0, high=255, shape=(2048,)),
            "words": spaces.Box(low=0, high=num_words, shape=(self.max_len,)),
        }
        self.action_space = spaces.Discrete(self.num_words, start=1)

        self.state = {"image": image, "words": self._initial_words()}

    def _initial_words(self):
        return np.pad([self.startseq], (0, self.max_len - 1),
                      "constant", constant_values=(0, 0))

    # ------------------------------------------------------------------ #
    def step(self, action):
        info = {}
        self.prev_rewards.append(self.reward)

        # Place the chosen word in the first free (zero) slot.
        for i, word in enumerate(self.state["words"]):
            if word == 0:
                self.state["words"][i] = action
                self.seq_len += 1
                break

        # Decode the caption so far and score it.
        ind_caption = []
        for word in self.state["words"]:
            if word == self.endseq or word == 0:
                break
            elif word == self.startseq:
                continue
            else:
                ind_caption.append(word)
        caption = self.tokenizer.sequences_to_texts([ind_caption])[0]
        self.reward = compute_bleu(self.train_caps[self.img_id], caption)
        self.total_rewards += self.reward

        if action == self.endseq or self.seq_len == self.max_len:
            self.done = True
        return self.state, self.reward, self.done, info

    # ------------------------------------------------------------------ #
    def reset(self, image=None, img_id=None):
        self.prev_rewards = []
        self.reward = 0
        self.total_rewards = 0
        self.done = False
        self.seq_len = 1
        self.state["words"] = self._initial_words()
        if image is not None and img_id is not None:
            self.state["image"] = image
            self.img_id = img_id
        return self.state

    # ------------------------------------------------------------------ #
    def render(self):
        ind_caption = []
        for word in self.state["words"]:
            if word == 0:
                break
            ind_caption.append(word)
        caption = self.tokenizer.sequences_to_texts([ind_caption])[0]
        print(caption)
        return caption
