"""Network definitions, as in the notebook.

* ``build_generator``     — the LSTM decoder on top of cached InceptionV3
                            features (a "merge" architecture). Trained with MLE,
                            then refined with SCST inside the GAN. Matches the
                            deployed ``final_model_V4.h5``.
* ``build_discriminator`` — an LSTM that scores an (image-features, caption)
                            pair as real (1) or fake/mismatched (0).
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Embedding,
    Input,
    LSTM,
    add,
)
from tensorflow.keras.models import Model

from .config import CONFIG


def build_generator(num_words: int = CONFIG.num_words,
                    max_length: int = CONFIG.max_length,
                    embedding_dim: int = CONFIG.embedding_dim) -> Model:
    """LSTM caption decoder.

    Inputs
    ------
    inputs1 : (2048,)        cached InceptionV3 image feature vector
    inputs2 : (max_length,)  partial caption (token ids, left-padded)

    Output
    ------
    softmax distribution over the vocabulary for the next token.
    """
    # Image branch
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(512, activation="relu")(fe1)

    # Sequence branch
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(num_words, embedding_dim, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(512)(se2)

    # Merge + classify
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation="relu")(decoder1)
    outputs = Dense(num_words, activation="softmax")(decoder2)

    return Model(inputs=[inputs1, inputs2], outputs=outputs)


def build_discriminator(generator: Model | None = None,
                        num_words: int = CONFIG.num_words,
                        max_length: int = CONFIG.max_length,
                        embedding_dim: int = CONFIG.embedding_dim) -> Model:
    """LSTM discriminator.

    The image feature vector is fed to the LSTM as the first time step,
    followed by the embedded caption words. This is why ``embedding_dim``
    equals the 2048-d feature size.

    If ``generator`` is given, its word embedding is copied into the
    discriminator and frozen, as in the notebook.
    """
    img_features = Input(shape=(1, 2048))
    caption = Input(shape=(max_length,))
    embedded_caption = Embedding(num_words, embedding_dim, mask_zero=True)(caption)

    disc_input = tf.concat([img_features, embedded_caption], axis=1)

    disc_lstm = LSTM(512)(disc_input)
    disc_dense = Dense(512, activation="leaky_relu")(disc_lstm)
    # The notebook also creates Dropout(0.4) here but never applies it to a
    # tensor, so it is not part of the trained model and is omitted.
    outputs = Dense(1, activation="sigmoid")(disc_dense)

    disc = Model(inputs=[img_features, caption], outputs=outputs)

    if generator is not None:
        disc_emb = _first_embedding(disc)
        disc_emb.set_weights(_first_embedding(generator).get_weights())
        disc_emb.trainable = False
    return disc


def _first_embedding(model: Model) -> Embedding:
    return next(layer for layer in model.layers if isinstance(layer, Embedding))
