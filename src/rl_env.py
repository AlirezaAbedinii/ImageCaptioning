"""Reinforcement-learning environment for caption generation.

``ICEnv`` follows the OpenAI Gym interface and turns caption generation into a
sequential decision problem, so the generator can be refined with policy-gradient
methods (SCST). It is transcribed from the thesis (figures 16-20).

* **state**  — ``{"image": 2048-d features, "words": token-id sequence}``
* **action** — choosing the next vocabulary index (``Discrete(num_words, start=1)``)
* **reward** — sentence-level BLEU of the caption built so far against the
  image's ground-truth captions.

``train_caps`` is the ``{image_url: "caption text"}`` mapping used to score
candidate captions; pass it in when constructing the environment.
"""
from __future__ import annotations

import gym
import numpy as np
from gym import spaces
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

from .config import CONFIG


def compute_bleu(references: list[str], candidate: str) -> float:
    """Sentence-level BLEU of ``candidate`` against one or more reference strings."""
    refs = [r.split() for r in references]
    hyp = candidate.split()
    if not hyp:
        return 0.0
    return sentence_bleu(refs, hyp, smoothing_function=SmoothingFunction().method1)


class ICEnv(gym.Env):
    """Image-Captioning environment."""

    def __init__(self, image, img_id, tokenizer, train_caps,
                 max_len: int = CONFIG.max_length,
                 num_words: int = CONFIG.num_words,
                 startseq: int | None = None, endseq: int | None = None):
        super().__init__()
        self.image = image
        self.img_id = img_id
        self.tokenizer = tokenizer
        self.train_caps = train_caps
        self.max_len = max_len
        self.num_words = num_words
        self.startseq = startseq if startseq is not None else tokenizer.word_index["startseq"]
        self.endseq = endseq if endseq is not None else tokenizer.word_index["endseq"]

        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(2048,)),
            "words": spaces.Box(low=0, high=num_words, shape=(self.max_len,)),
        })
        self.action_space = spaces.Discrete(num_words, start=1)

        self.state = {"image": image, "words": np.zeros(self.max_len)}
        self.reset(image, img_id)

    # ------------------------------------------------------------------ #
    def reset(self, image=None, img_id=None):
        self.prev_rewards = []
        self.reward = 0
        self.total_rewards = 0
        self.done = False
        self.seq_len = 1
        self.state["words"] = np.pad(
            [self.startseq], (0, self.max_len - 1), "constant", constant_values=(0, 0)
        )
        if image is not None and img_id is not None:
            self.state["image"] = image
            self.img_id = img_id
        return self.state

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

        # Decode the caption so far and score it with BLEU.
        ind_caption = []
        for word in self.state["words"]:
            if word == self.endseq or word == 0:
                break
            elif word == self.startseq:
                continue
            else:
                ind_caption.append(int(word))
        caption = self.tokenizer.sequences_to_texts([ind_caption])[0]
        references = [self.train_caps[self.img_id]]
        self.reward = compute_bleu(references, caption)
        self.total_rewards += self.reward

        if action == self.endseq or self.seq_len == self.max_len:
            self.done = True
        return self.state, self.reward, self.done, info

    # ------------------------------------------------------------------ #
    def render(self, with_image: bool = False):
        ind_caption = [int(w) for w in self.state["words"] if w != 0]
        caption = self.tokenizer.sequences_to_texts([ind_caption])[0]
        print(caption)
        return caption
