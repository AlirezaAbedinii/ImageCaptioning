"""Central configuration: filesystem paths and model hyper-parameters.

All paths default to the values used while developing the project on Google
Colab (the dataset lived on Google Drive). Override them through environment
variables or by editing this file before running the pipeline locally.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


# --------------------------------------------------------------------------- #
# Paths                                                                        #
# --------------------------------------------------------------------------- #
def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


# Root of the (unzipped) MS-COCO 2017 annotations + images.
DATA_DIR = _env("IC_DATA_DIR", "data")

# COCO split used as the working set. The project uses the *validation* set
# (val2017, ~5 000 images) because of the limited RAM/time budget on Colab.
DATA_TYPE = _env("IC_DATA_TYPE", "val2017")

ANNOTATIONS_FILE = os.path.join(DATA_DIR, "annotations", f"captions_{DATA_TYPE}.json")
IMAGES_DIR = os.path.join(DATA_DIR, DATA_TYPE)

TRAIN_DIR = os.path.join(DATA_DIR, "Train")
TEST_DIR = os.path.join(DATA_DIR, "Test")

TRAIN_DF = os.path.join(DATA_DIR, "train_data.csv")
TEST_DF = os.path.join(DATA_DIR, "test_data.csv")

# Cached 2048-d InceptionV3 feature vectors ({image_url: np.ndarray}).
TRAIN_FEATURES = os.path.join(DATA_DIR, "images1.pkl")
TEST_FEATURES = os.path.join(DATA_DIR, "test_images.pkl")

# Captions produced by the MLE generator, used to pre-train the discriminator.
FAKE_CAPTIONS = os.path.join(DATA_DIR, "fake_captions.pkl")

# Trained artefacts (kept out of git — see .gitignore).
MODELS_DIR = _env("IC_MODELS_DIR", "models")
GENERATOR_WEIGHTS = os.path.join(MODELS_DIR, "final_model_V4.h5")
DISCRIMINATOR_WEIGHTS = os.path.join(MODELS_DIR, "Disc_V1.h5")
TOKENIZER_PICKLE = os.path.join(MODELS_DIR, "tokenizer.pickle")
FLAT_CAPTIONS_PICKLE = os.path.join(MODELS_DIR, "flat_train_caps.pickle")


# --------------------------------------------------------------------------- #
# Hyper-parameters                                                             #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Config:
    # Vocabulary / sequences
    num_words: int = 2075          # vocabulary size kept by the tokenizer
    embedding_dim: int = 2048      # learned word embedding (same size as image features)
    max_length: int = 16           # maximum caption length (in tokens)

    # Image encoder
    feature_dim: int = 2048        # InceptionV3 penultimate-layer size
    inception_input: tuple = (299, 299)

    train_split: float = 0.8       # train / test ratio for the image split

    # Generator (decoder) MLE pre-training
    gen_epochs: int = 10
    gen_batch_size: int = 3
    gen_lr: float = 1e-3

    # Discriminator pre-training
    disc_epochs: int = 15
    disc_batch_size: int = 16

    # Adversarial (GAN + SCST) training
    gan_epochs: int = 25
    mini_batch: int = 8            # generated samples examined per GAN step
    disc_data_batch: int = 128     # samples produced to refresh the discriminator
    disc_training_batch: int = 8   # discriminator batch size during GAN
    disc_inner_epochs: int = 4     # discriminator epochs per GAN step (thesis run; the notebook's last run used 1)
    lambda_val: float = 0.2        # reward mix: r = λ·D + (1-λ)·BLEU
    scst_lr: float = 5e-5          # Adam learning rate for the SCST update
    scst_clip: float = 0.5         # gradient clip value


CONFIG = Config()
