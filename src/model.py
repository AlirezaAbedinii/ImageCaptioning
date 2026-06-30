"""Network definitions.

* ``build_generator``     — the CNN-encoder / LSTM-decoder caption generator
                            (a classic "merge" architecture). Pre-trained with
                            MLE, then refined with SCST inside the GAN.
* ``build_discriminator`` — an RNN that scores an (image-features, caption)
                            pair as real or fake.

Both architectures are transcribed from the thesis (figures 13 and 21). The
generator matches the deployed ``final_model_V4.h5`` used by the inference
server.
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
                    embedding_dim: int = CONFIG.embedding_dim,
                    embedding_matrix=None) -> Model:
    """CNN-encoder + LSTM-decoder caption model (thesis fig. 13).

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
    f_layer1 = Dropout(0.5)(inputs1)
    f_layer2 = Dense(512, activation="relu")(f_layer1)

    # Sequence branch
    inputs2 = Input(shape=(max_length,))
    s_layer1 = Embedding(num_words, embedding_dim, mask_zero=True)(inputs2)
    s_layer2 = Dropout(0.5)(s_layer1)
    s_layer3 = LSTM(512)(s_layer2)

    # Merge + classify
    decoder1 = add([f_layer2, s_layer3])
    decoder2 = Dense(256, activation="relu")(decoder1)
    outputs = Dense(num_words, activation="softmax")(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)

    if embedding_matrix is not None:
        # Use the GloVe matrix and freeze the embedding layer.
        model.layers[2].set_weights([embedding_matrix])
        model.layers[2].trainable = False
    return model


def build_discriminator(num_words: int = CONFIG.num_words,
                        max_length: int = CONFIG.max_length,
                        embedding_dim: int = CONFIG.embedding_dim) -> Model:
    """RNN discriminator (thesis fig. 21).

    Scores an (image-features, caption) pair in [0, 1]: 1 = real human caption,
    0 = generated / mismatched. Pre-trained with binary cross-entropy on real,
    fake (generator) and "wrong" (mismatched) pairs.
    """
    img_features = Input(shape=(1, 2048))
    caption = Input(shape=(max_length,))
    embedded_caption = Embedding(num_words, embedding_dim, mask_zero=True)(caption)

    disc_input = tf.concat([img_features, embedded_caption], axis=1)

    disc_lstm = LSTM(512)(disc_input)
    disc_dense = Dense(512, activation="leaky_relu")(disc_lstm)
    disc_dropout = Dropout(0.4)(disc_dense)
    outputs = Dense(1, activation="sigmoid")(disc_dropout)

    return Model(inputs=[img_features, caption], outputs=outputs)
