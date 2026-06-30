"""Training pipeline: MLE pre-training followed by adversarial (GAN + SCST)
refinement.

The training code was developed and run on Google Colab; the functions below
are transcribed/reconstructed from the thesis (figures 9, 10, 14, 23-26) so the
repository documents the full method end-to-end. Running them requires the
prepared dataset (cached features, tokenizer, GloVe matrix) and a GPU — the
released weights (``final_model_V4.h5``) already contain the result of this
process.

Stages
------
1. ``pretrain_generator``      MLE (categorical cross-entropy) on (image, partial
                               caption) -> next-token pairs.
2. ``pretrain_discriminator``  Binary cross-entropy on real / fake / wrong pairs.
3. ``train_gan``               Alternating SCST generator updates (self-critical
                               policy gradient, reward = λ·D + (1-λ)·BLEU) and
                               discriminator updates.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences

from .config import CONFIG
from .data import img_id_to_url
from .rl_env import ICEnv, compute_bleu


# --------------------------------------------------------------------------- #
# 1. Generator (decoder) MLE pre-training                                      #
# --------------------------------------------------------------------------- #
def mle_data_generator(tokenizer, train_caps, photos,
                       batch_size: int = CONFIG.gen_batch_size,
                       max_length: int = CONFIG.max_length,
                       num_words: int = CONFIG.num_words):
    """Yield ``([image_features, partial_sequence], next_word)`` batches.

    For every caption, each ``n``-gram prefix becomes one training example whose
    target is the following token (thesis fig. 14).
    """
    image_inp, sequence_inp, out_word = [], [], []
    n = 0
    while True:
        for img_id, caps_list in train_caps.items():
            n += 1
            photo = photos[img_id_to_url(img_id)]
            for cap in caps_list:
                seq = tokenizer.texts_to_sequences([cap])[0]
                for i in range(1, min(max_length + 1, len(seq))):
                    in_seq = pad_sequences([seq[:i]], maxlen=max_length)[0]
                    out_seq = tf.keras.utils.to_categorical(
                        [seq[i]], num_classes=num_words
                    )[0]
                    image_inp.append(photo)
                    sequence_inp.append(in_seq)
                    out_word.append(out_seq)
            if n == batch_size:
                yield ([np.array(image_inp), np.array(sequence_inp)],
                       np.array(out_word))
                image_inp, sequence_inp, out_word = [], [], []
                n = 0


def pretrain_generator(generator, tokenizer, train_caps, photos,
                       epochs: int = CONFIG.gen_epochs,
                       batch_size: int = CONFIG.gen_batch_size,
                       steps_per_epoch: int | None = None):
    """Compile + fit the decoder with categorical cross-entropy (Adam)."""
    generator.compile(loss="categorical_crossentropy",
                      optimizer=Adam(learning_rate=CONFIG.gen_lr))
    gen = mle_data_generator(tokenizer, train_caps, photos, batch_size)
    steps = steps_per_epoch or max(1, len(train_caps) // batch_size)
    generator.fit(gen, epochs=epochs, steps_per_epoch=steps, verbose=1)
    return generator


# --------------------------------------------------------------------------- #
# 2. Discriminator pre-training                                                #
# --------------------------------------------------------------------------- #
def pretrain_discriminator(discriminator, features, padded_captions, labels,
                           epochs: int = CONFIG.disc_epochs,
                           batch_size: int = CONFIG.disc_batch_size):
    """Binary cross-entropy on real (1) / fake & wrong (0) pairs (thesis §4-6)."""
    discriminator.compile(loss="binary_crossentropy", optimizer=Adam(),
                          metrics=["accuracy"])
    discriminator.fit(x=[features, padded_captions], y=labels,
                      epochs=epochs, batch_size=batch_size, verbose=1)
    return discriminator


# --------------------------------------------------------------------------- #
# 3. Adversarial refinement (SCST + GAN)                                       #
# --------------------------------------------------------------------------- #
def sample_caption_rl(tokenizer, image, img_id, generator, train_caps,
                      max_length: int = CONFIG.max_length,
                      num_words: int = CONFIG.num_words, greedy: bool = False):
    """Roll out one caption through ``ICEnv`` (thesis fig. 25).

    ``greedy=False`` samples from the policy (used for the SCST sample);
    ``greedy=True`` takes the arg-max (used for the self-critical baseline).
    """
    env = ICEnv(image=image, img_id=img_id, tokenizer=tokenizer,
                train_caps=train_caps, max_len=max_length, num_words=num_words)
    state = env.reset()
    done = False
    actions, probs_list, states = [], [], []

    while not done:
        pre_zero_seq = _pad_post_to_pre(env.state["words"], max_length)
        word_probs = generator.predict(
            [np.array(env.state["image"]).reshape((1, 2048)),
             np.array([pre_zero_seq])], verbose=0)
        if greedy:
            predicted_index = np.array([int(np.argmax(word_probs))])
        else:
            dist = tfp.distributions.Categorical(probs=word_probs, dtype=tf.float32)
            predicted_index = np.array(dist.sample())

        states.append(np.array(env.state["words"]))
        actions.append(int(predicted_index[0]))
        probs_list.append(word_probs)
        state, reward, done, _ = env.step(int(predicted_index[0]))

    return actions, probs_list, states, env.total_rewards


def scst_loss(action, reward, probs, greedy_reward):
    """Self-critical policy-gradient loss (thesis fig. 26 / eq. 10).

    ``loss = -log π(action) · (reward_sampled - reward_greedy)``
    """
    dist = tfp.distributions.Categorical(probs=probs, dtype=tf.float32)
    log_prob = dist.log_prob(action)
    return -log_prob * (reward - greedy_reward)


def compute_reward(disc_score: float, bleu: float,
                   lambda_val: float = CONFIG.lambda_val) -> float:
    """Mixed reward ``r = λ·D(x|I) + (1-λ)·BLEU`` (thesis fig. 24)."""
    return lambda_val * disc_score + (1 - lambda_val) * bleu


def train_gan(generator, discriminator, tokenizer, train_caps, photos,
              epochs: int = CONFIG.gan_epochs,
              mini_batch: int = CONFIG.mini_batch):
    """Alternating GAN loop (thesis fig. 23).

    Each epoch: (a) the generator produces a mini-batch of captions and is
    updated with SCST using the mixed reward; (b) the discriminator is refreshed
    on freshly generated real/fake/wrong samples. ``λ`` trades off the
    discriminator signal against BLEU.
    """
    optimizer = Adam(learning_rate=CONFIG.scst_lr, clipvalue=CONFIG.scst_clip)
    img_ids = list(train_caps.keys())

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        batch_ids = np.random.choice(img_ids, size=mini_batch, replace=False)

        # ----- generator (SCST) update -------------------------------------
        with tf.GradientTape() as tape:
            losses = []
            for img_id in batch_ids:
                image = photos[img_id_to_url(img_id)]
                actions, probs_list, _, reward_s = sample_caption_rl(
                    tokenizer, image, img_id, generator, train_caps, greedy=False)
                _, _, _, reward_g = sample_caption_rl(
                    tokenizer, image, img_id, generator, train_caps, greedy=True)

                disc_score = float(_score_caption(
                    discriminator, image, actions, tokenizer))
                reward = compute_reward(disc_score, reward_s)
                for action, probs in zip(actions, probs_list):
                    losses.append(scst_loss(action, reward, probs, reward_g))
            loss = tf.reduce_mean(losses)
        grads = tape.gradient(loss, generator.trainable_variables)
        optimizer.apply_gradients(zip(grads, generator.trainable_variables))

        # ----- discriminator update ----------------------------------------
        # (Generate real/fake/wrong samples and fit; see thesis §4-7.)
        # left as an extension point — requires the sample-generation helpers.
        print(f"  generator SCST loss: {float(loss):.4f}")

    return generator, discriminator


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def _pad_post_to_pre(words, max_length):
    """Convert a post-padded word sequence into the left-padded form the
    decoder expects (zeros moved to the front)."""
    tokens = [int(w) for w in words if w != 0]
    return pad_sequences([tokens], maxlen=max_length)[0]


def _score_caption(discriminator, image, actions, tokenizer,
                   max_length: int = CONFIG.max_length):
    """Discriminator confidence that ``actions`` is a real caption for ``image``."""
    padded = pad_sequences([actions], maxlen=max_length)
    features = np.array(image).reshape((1, 1, 2048))
    return discriminator.predict([features, padded], verbose=0)[0][0]
