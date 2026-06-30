"""Image encoder.

A pre-trained InceptionV3 (ImageNet weights) with its classification head
removed turns each image into a single **2048-d feature vector**. These vectors
are cached to disk so that the rest of the pipeline never has to touch the raw
images again — the key memory/time optimisation that makes training feasible on
a free Colab instance.

Mirrors the "New way" / feature-extraction cells of the notebook and the
``encode`` helper used by the inference server.
"""
from __future__ import annotations

import pickle

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from .config import CONFIG


def build_encoder() -> tf.keras.Model:
    """InceptionV3 truncated at its penultimate layer (output: 2048-d)."""
    base = tf.keras.applications.InceptionV3(weights="imagenet")
    return tf.keras.models.Model(base.input, base.layers[-2].output)


def preprocess(image_path: str) -> np.ndarray:
    """Load + preprocess an image into InceptionV3's expected input tensor."""
    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=CONFIG.inception_input
    )
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return tf.keras.applications.inception_v3.preprocess_input(x)


def encode(encoder: tf.keras.Model, image_path: str) -> np.ndarray:
    """Encode one image into a flat ``(2048,)`` feature vector."""
    fea_vec = encoder.predict(preprocess(image_path), verbose=0)
    return np.reshape(fea_vec, fea_vec.shape[1])


def cache_features(encoder: tf.keras.Model, image_dir: str, urls,
                   out_path: str) -> dict[str, np.ndarray]:
    """Encode every image in ``urls`` and pickle the {url: vector} mapping."""
    encoding = {}
    for url in tqdm(urls, desc="encoding"):
        encoding[url] = encode(encoder, f"{image_dir}/{url}")
    with open(out_path, "wb") as fh:
        pickle.dump(encoding, fh)
    return encoding


def load_features(path: str) -> dict[str, np.ndarray]:
    with open(path, "rb") as fh:
        return pickle.load(fh)
