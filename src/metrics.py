"""BLEU scoring, as used for both evaluation and the RL reward.

Note: the notebook passes raw strings to NLTK's ``sentence_bleu``. NLTK
iterates over a string's characters, so the score is a **character n-gram
BLEU**, not the word-level BLEU-4 used in captioning papers. The default here
reproduces that behaviour so the reported numbers can be reproduced; set
``word_level=True`` for conventional word-level BLEU-4.
"""
from __future__ import annotations

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

from .vocab import remove_start_and_endseq

_SMOOTHING = SmoothingFunction(epsilon=1e-12).method1


def compute_bleu(references: list[str], candidate: str,
                 word_level: bool = False) -> float:
    """Sentence BLEU of ``candidate`` against ``references``.

    ``startseq`` / ``endseq`` markers are stripped from both sides first.
    """
    refs = remove_start_and_endseq(references)
    cand = remove_start_and_endseq([candidate])[0]
    if word_level:
        return sentence_bleu([r.split() for r in refs], cand.split(),
                             smoothing_function=_SMOOTHING)
    return sentence_bleu(refs, cand, smoothing_function=_SMOOTHING)
