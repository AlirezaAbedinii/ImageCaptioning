# Image Captioning with Conditional GANs

> Generating natural-language descriptions of images with a CNN–LSTM
> encoder–decoder, refined adversarially with a discriminator and **Self-Critical
> Sequence Training (SCST)**.

<p align="left">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.7%2B-3776AB?logo=python&logoColor=white">
  <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white">
  <img alt="Dataset" src="https://img.shields.io/badge/Dataset-MS--COCO-blue">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green">
</p>

This is my undergraduate (B.Sc.) thesis project — an implementation of
**“Improving Image Captioning with Conditional Generative Adversarial Nets”**
(Chen *et al.*, AAAI 2019). A generator describes an image; a discriminator
learns to tell human captions from machine ones; the generator is then trained
with reinforcement learning to fool the discriminator *and* maximise BLEU.

---

## Demo

The best-scoring captions on the held-out test images (the thesis's "very good"
group, BLEU > 0.8). A typical caption scores around 0.55–0.6.

![Result gallery](assets/demo/gallery.png)

| Generated caption | BLEU |
| ----------------- | :--: |
| a man is riding a surfboard in the ocean | **0.963** |
| a bathroom with a sink and a shower curtain | **0.927** |
| a man is riding a skateboard on a cement wall | **0.922** |
| a group of motorcycles parked on the side of a road | **0.863** |
| a man is swinging a tennis racket at a ball | **0.835** |

**Average test-set BLEU ≈ 0.57** (~10 % over the plain encoder–decoder baseline).
More examples in [`assets/demo/`](assets/demo/README.md).

> **How to read these scores:** BLEU here is computed with NLTK's
> `sentence_bleu` on raw strings, which compares **character** n-grams rather than
> words (see [`src/metrics.py`](src/metrics.py)). The numbers are consistent
> within this project, but they are **not** comparable to the word-level BLEU-4
> reported in captioning papers.

---

## Table of contents

- [Highlights](#highlights)
- [How it works](#how-it-works)
- [Repository structure](#repository-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Interactive dashboard](#interactive-dashboard)
- [Notebook](#notebook)
- [Citation](#citation)
- [Author](#author)

---

## Highlights

- **Conditional-GAN captioning** — a CNN-encoder/LSTM-decoder *generator* plus an
  RNN *discriminator*, following Chen *et al.* (2019).
- **SCST reinforcement learning** — the non-differentiable BLEU/discriminator
  reward is optimised with a self-critical policy gradient, using a custom
  OpenAI-Gym environment ([`src/rl_env.py`](src/rl_env.py)).
- **Memory-efficient training pipeline** — instead of carrying raw images
  through training, each image is encoded **once** into a **2048-d InceptionV3
  feature vector** and cached to disk; preprocessed data is serialised for
  repeatable experiments. This is what makes the whole pipeline fit in the
  RAM/time budget of a free Google Colab GPU.
- **End-to-end** — data prep, vocabulary, model definitions, training and a small
  **web dashboard** for live captioning.

## How it works

```
          ┌──────────────┐   2048-d    ┌───────────────────────────┐
 image ──▶│ InceptionV3  │────feature─▶│ Generator (LSTM decoder)  │──▶ caption
          │  (frozen)    │             └───────────────────────────┘
          └──────────────┘                        │  caption + image features
                                                   ▼
                                       ┌───────────────────────────┐
                                       │ Discriminator (LSTM)      │──▶ real / fake
                                       └───────────────────────────┘
```

1. **Encoder** — InceptionV3 with its classification head removed maps each image
   to a 2048-d vector ([`src/features.py`](src/features.py)).
2. **Generator (decoder)** — a "merge" model: the image vector and the partial
   caption (learned embedding → LSTM) are added and projected to a softmax over the
   vocabulary ([`src/model.py`](src/model.py)). Pre-trained with maximum
   likelihood (categorical cross-entropy).
3. **Discriminator** — an LSTM that scores an `(image-features, caption)` pair as
   real or fake. Pre-trained on real / generated / mismatched captions.
4. **Adversarial refinement** — the generator is updated with **SCST**
   ([`src/train.py`](src/train.py)) to maximise a mixed reward
   `r = λ·D(caption | image) + (1 − λ)·BLEU` (λ = 0.2), while the discriminator is
   periodically refreshed.

A full walk-through of the method (with the equations) is in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). How this implementation differs
from the paper (one RNN discriminator, no ensemble, frozen CNN, no attention) is
covered in [Relation to the paper](docs/ARCHITECTURE.md#8-relation-to-the-paper).

## Repository structure

```
ImageCaptioning/
├── Image_Captioning.ipynb     # complete Colab notebook (full pipeline, with outputs)
├── src/                       # refactored, documented pipeline
│   ├── config.py              #   paths + hyper-parameters
│   ├── data.py                #   COCO loading, train/test split, dataframes
│   ├── features.py            #   InceptionV3 encoder + 2048-d feature caching
│   ├── vocab.py               #   caption dictionaries + tokenizer
│   ├── model.py               #   generator (decoder) + discriminator
│   ├── metrics.py             #   BLEU (evaluation and RL reward)
│   ├── rl_env.py              #   Gym environment for SCST
│   ├── train.py               #   MLE pre-training + GAN/SCST loop
│   └── inference.py           #   greedy caption decoding
├── app/                       # interactive dashboard
│   ├── server.py              #   HTTP inference server
│   └── web/                   #   front-end (HTML/CSS/JS)
├── assets/demo/               # result gallery + scores
├── docs/ARCHITECTURE.md       # method write-up
├── requirements.txt
└── LICENSE
```

## Dataset

The project uses only the **MS-COCO 2017 validation split**: 5 000 images with
5–6 human captions each. The full training split was too large for a free
Colab instance. The images are split randomly 80/20 into **4 000 train /
1 000 test** images.

| Download | Link | What is used |
| -------- | ---- | ------------ |
| `val2017.zip` (~1 GB) | <http://images.cocodataset.org/zips/val2017.zip> | the 5 000 images |
| `annotations_trainval2017.zip` | <http://images.cocodataset.org/annotations/annotations_trainval2017.zip> | `annotations/captions_val2017.json` |

Unzip both into one folder:

```
data/
├── annotations/captions_val2017.json
├── val2017/            # 5 000 .jpg files
├── Train/              # created by the train/test split
└── Test/               # created by the train/test split
```

[`src/config.py`](src/config.py) expects this layout under `data/` (override it
with `IC_DATA_DIR`). The notebook's paths point at Google Drive. Edit the
variables in its *Globals* section before running it; the notebook's first cell
lists the one-off steps (split, feature caching, …) to run on a fresh setup.

## Installation

```bash
git clone https://github.com/AlirezaAbedinii/ImageCaptioning.git
cd ImageCaptioning
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
```

To run inference without retraining you also need the **trained weights**
(`final_model_V4.h5`, `flat_train_caps.pickle`) in `models/`. They are too large
for git and are available on request.

## Usage

```bash
# 1. Build the train/test dataframes from COCO (run once)
python -m src.data

# 2. Cache 2048-d image features (run once; see src/features.py)
python - <<'PY'
from src import config, data, features
enc = features.build_encoder()
train_df, _ = data.read_dataframes()
features.cache_features(enc, config.TRAIN_DIR, train_df["url"], config.TRAIN_FEATURES)
PY

# 3. Train (MLE pre-training, then GAN/SCST) — see src/train.py
```

> **Note** Training was performed on a free Google Colab GPU under tight
> RAM/time limits. The training code in [`src/train.py`](src/train.py) is the
> documented version of that pipeline; the released weights already contain its
> result, so you do not need to retrain to try the model.

## Interactive dashboard

A small web app lets you upload an image and see the caption.

```bash
# Terminal 1 — model server (needs models/final_model_V4.h5)
python -m app.server

# Terminal 2 — serve the front-end
cd app/web && python -m http.server 5500
# open http://localhost:5500
```

The page (`app/web/`) POSTs the image to the server (`localhost:8000`), which
encodes it, decodes a caption greedily and returns the text.

## Notebook

[`Image_Captioning.ipynb`](Image_Captioning.ipynb) is the complete Colab
notebook used for the thesis. It covers data preparation, feature caching, the
MLE decoder, discriminator pre-training, GAN/SCST training and evaluation, and
keeps its original outputs. Its first cell explains the dataset and the one-off
setup steps. The `src/` package is the same pipeline as importable modules.

## Citation

Base paper:

```bibtex
@inproceedings{chen2019improving,
  title     = {Improving Image Captioning with Conditional Generative Adversarial Nets},
  author    = {Chen, Chen and Mu, Shuai and Xiao, Wanpeng and Ye, Zexiong and Wu, Liesi and Ju, Qi},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2019},
  doi       = {10.1609/aaai.v33i01.33018142}
}
```

Key building blocks: SCST (Rennie *et al.*, CVPR 2017), BLEU (Papineni *et al.*,
2002), MS-COCO (Lin *et al.*, 2014), InceptionV3 (Szegedy *et al.*, 2016).

## Author

**Alireza Abedini** — B.Sc. thesis project.
GitHub: [@AlirezaAbedinii](https://github.com/AlirezaAbedinii)

Released under the [MIT License](LICENSE).
