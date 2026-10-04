"""Training pipeline: MLE pre-training, discriminator pre-training and the
adversarial (GAN + SCST) loop.

Ported from the notebook's "Decoder", "Final Training" and "Main Loop"
sections. The notebook relied on global variables; here they are passed
explicitly. Running this needs the cached features, the caption dictionaries
and a GPU. The released weights (``final_model_V4.h5``) are the MLE generator.

Data structures
---------------
features / t_features : ``{image_url: (2048,) np.ndarray}`` (train / test)
train_caps / test_caps : ``{image_id: [marked captions]}``
"""
from __future__ import annotations

import math
import random

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.preprocessing.sequence import pad_sequences

from .config import CONFIG
from .data import img_id_to_url, img_url_to_id
from .inference import generate_caption
from .metrics import compute_bleu
from .rl_env import ICEnv


# --------------------------------------------------------------------------- #
# 1. Generator (decoder) MLE pre-training                                      #
# --------------------------------------------------------------------------- #
def mle_data_generator(tokenizer, train_caps, photos,
                       batch_size: int = CONFIG.gen_batch_size,
                       max_length: int = CONFIG.max_length,
                       num_words: int = CONFIG.num_words):
    """Yield ``([image_features, partial_sequence], next_word)`` batches.

    Every prefix of every caption becomes one example whose target is the
    following token.
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
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq = tf.keras.utils.to_categorical(
                        [out_seq], num_classes=num_words)[0]
                    image_inp.append(photo)
                    sequence_inp.append(in_seq)
                    out_word.append(out_seq)
            if n == batch_size:
                yield ([np.array(image_inp), np.array(sequence_inp)],
                       np.array(out_word))
                image_inp, sequence_inp, out_word = [], [], []
                n = 0


def pretrain_generator(model, tokenizer, train_caps, features,
                       epochs: int = CONFIG.gen_epochs,
                       batch_size: int = CONFIG.gen_batch_size):
    """Categorical cross-entropy with Adam, one ``fit`` call per epoch."""
    model.compile(loss="categorical_crossentropy", optimizer="adam",
                  metrics=["accuracy"])
    steps = len(train_caps) // batch_size
    tf.config.run_functions_eagerly(True)
    for i in range(epochs):
        print(f"Epoch {i + 1}/{epochs}")
        generator = mle_data_generator(tokenizer, train_caps, features, batch_size)
        model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)
    return model


# --------------------------------------------------------------------------- #
# 2. Discriminator pre-training                                                #
# --------------------------------------------------------------------------- #
def generate_fake_captions(gmodel, tokenizer, features, n_samples=None,
                           max_length: int = CONFIG.max_length):
    """Greedy captions from the generator, ``{image_url: caption}``.

    With ``n_samples=None`` every image is captioned (used once, before
    pre-training the discriminator); otherwise a random subset is used.
    """
    urls = list(features.keys())
    if n_samples is not None:
        urls = random.sample(urls, n_samples)
    return {url: generate_caption(tokenizer, features[url].reshape((1, 2048)),
                                  gmodel, max_length)
            for url in urls}


def _different_image(img_id, all_ids):
    while True:
        second_id = random.choice(all_ids)
        if second_id != img_id:
            return second_id


def _to_disc_inputs(disc_features, disc_caps, disc_y, tokenizer, max_length):
    np_features = np.array(disc_features)
    sequences = tokenizer.texts_to_sequences(np.array(disc_caps))
    padded = pad_sequences(sequences, maxlen=max_length)
    return np_features, padded, disc_y


def generate_real_fake_wrong_samples(train_caps, features, fake_captions,
                                     tokenizer,
                                     max_length: int = CONFIG.max_length):
    """Pre-training set: 2 real captions per image (label 1), one generated
    caption per image (0) and one caption of a different image (0)."""
    fake_features, fake_caps = [], []
    for url, caption in fake_captions.items():
        fake_features.append(features[url])
        fake_caps.append(caption)

    real_features, real_caps = [], []
    for img_id in train_caps:
        feature = features[img_id_to_url(img_id)]
        real_cap1, real_cap2 = random.sample(train_caps[img_id], 2)
        real_features += [feature, feature]
        real_caps += [real_cap1, real_cap2]

    all_ids = list(train_caps.keys())
    wrong_features, wrong_caps = [], []
    for img_id in train_caps:
        wrong_features.append(features[img_id_to_url(img_id)])
        wrong_caps.append(random.choice(train_caps[_different_image(img_id, all_ids)]))

    disc_y = np.concatenate((np.ones(len(real_caps)), np.zeros(len(fake_caps)),
                             np.zeros(len(wrong_caps))), axis=0)
    return _to_disc_inputs(real_features + fake_features + wrong_features,
                           real_caps + fake_caps + wrong_caps,
                           disc_y, tokenizer, max_length)


def pretrain_discriminator(disc, np_features, padded, disc_y,
                           epochs: int = CONFIG.disc_epochs,
                           batch_size: int = CONFIG.disc_batch_size):
    """Binary cross-entropy with Adam."""
    n = len(disc_y)
    disc.compile(loss="binary_crossentropy", optimizer="adam",
                 metrics=["accuracy"])
    disc.fit(x=[np_features.reshape([n, 1, 2048]), padded],
             y=disc_y.reshape(n, 1), verbose=1, epochs=epochs,
             batch_size=batch_size)
    return disc


# --------------------------------------------------------------------------- #
# 3. Adversarial refinement (SCST + GAN)                                       #
# --------------------------------------------------------------------------- #
def post_zero_to_pre(array):
    """Move the zero padding of a word sequence to the front."""
    non_zero = [num for num in array if num != 0]
    pre_z = np.zeros(len(array) - len(non_zero), dtype=np.int32)
    return np.concatenate((pre_z, np.array(non_zero)), axis=0)


def sample_caption_rl(tokenizer, image, img_id, gmodel, train_caps,
                      max_length: int = CONFIG.max_length,
                      num_words: int = CONFIG.num_words):
    """Sample one caption from the generator's distribution through ``ICEnv``.

    Returns the final reward, the state before every action, the actions and
    the caption text.
    """
    env = ICEnv(image=image, img_id=img_id, tokenizer=tokenizer,
                train_caps=train_caps, max_len=max_length, num_words=num_words)
    env.reset()
    done = False
    reward = 0
    actions = []
    caption = "startseq"
    while not done:
        pre_zero_seq = post_zero_to_pre(env.state["words"])
        word_probs = gmodel.predict(
            [np.array(env.state["image"]).reshape((1, 2048)),
             np.array([pre_zero_seq])], verbose=0)

        dist = tfp.distributions.Categorical(probs=word_probs, dtype=tf.float32)
        predicted_index = np.array(dist.sample())

        new_word = tokenizer.sequences_to_texts([predicted_index])[0]
        caption += " " + new_word
        action = tokenizer.word_index[new_word]
        actions.append(action)
        _, reward, done, _ = env.step(action)

    startseq = tokenizer.word_index["startseq"]
    total_states = []
    for i in range(1, len(actions) + 1):
        prefix = [startseq] + actions[:i - 1]
        curr_state = np.concatenate(
            (prefix, np.zeros(max_length - len(prefix), dtype=np.int32)), axis=0)
        total_states.append({"image": env.state["image"], "words": curr_state})
    return reward, total_states, actions, caption


def create_fake_captions_batch(batch_size, gmodel, tokenizer, train_caps, features):
    """Sample ``batch_size`` captions for random training images."""
    fake_batch = {}
    total_actions, total_rewards, total_urls, total_states = [], [], [], []
    while len(fake_batch) < batch_size:
        img_id = random.choice(list(train_caps.keys()))
        img_url = img_id_to_url(img_id)
        feature = features[img_url].reshape((1, 2048))
        reward, states, actions, caption = sample_caption_rl(
            tokenizer, feature, img_id, gmodel, train_caps)
        total_actions.append(actions)
        total_rewards.append(reward)
        fake_batch[img_url] = caption
        total_urls.append(img_url)
        total_states.append(states)
    return fake_batch, total_actions, total_rewards, total_urls, total_states


def calculate_p(img_cap_pairs, disc, tokenizer, feature_lookup,
                max_length: int = CONFIG.max_length):
    """Discriminator score ``p`` for every (image, caption) pair."""
    urls = list(img_cap_pairs.keys())
    np_features = np.array([feature_lookup[u] for u in urls]).reshape(len(urls), 1, 2048)
    padded = pad_sequences(tokenizer.texts_to_sequences(list(img_cap_pairs.values())),
                           maxlen=max_length)
    return disc.predict([np_features, np.array(padded)]).flatten()


def calculate_s(img_cap_pairs, captions_dic):
    """Language score ``s`` (BLEU) for every (image, caption) pair."""
    return [compute_bleu(captions_dic[img_url_to_id(url)], cap)
            for url, cap in img_cap_pairs.items()]


def calculate_r(p_values, s_values, lambda_val: float = CONFIG.lambda_val):
    """Reward ``r = λ·p + (1 − λ)·s``."""
    return lambda_val * np.array(p_values) + (1 - lambda_val) * np.array(s_values)


def calculate_greedy_decoding_reward(img_url, tokenizer, gmodel, dmodel,
                                     captions_dic, feature_lookup,
                                     lambda_val: float = CONFIG.lambda_val,
                                     max_length: int = CONFIG.max_length):
    """Reward of the greedy (arg-max) caption: the SCST baseline."""
    picture = np.array(feature_lookup[img_url]).reshape((1, 2048))
    caption = generate_caption(tokenizer, picture, gmodel, max_length)
    bleu_score = compute_bleu(captions_dic[img_url_to_id(img_url)], caption)
    disc_score = calculate_p({img_url: caption}, dmodel, tokenizer,
                             feature_lookup, max_length)
    return bleu_score * (1 - lambda_val) + disc_score * lambda_val


def calculate_loss(action, reward, probs, greedy_reward):
    """Self-critical policy-gradient loss ``-log π(a) · (r − r_greedy)``."""
    dist = tfp.distributions.Categorical(probs=probs, dtype=tf.float32)
    log_prob = dist.log_prob(action)
    loss = -log_prob * (reward - greedy_reward)
    if math.isnan(loss):
        print("loss reached nan")
        print(np.min(probs), np.max(probs))
    return loss


def update_generator(r_values, actions, states, gmodel, urls, dmodel,
                     tokenizer, train_caps, features,
                     lambda_val: float = CONFIG.lambda_val):
    """One SCST pass over a sampled batch, one gradient step per word."""
    optimizer = tf.keras.optimizers.Adam(learning_rate=CONFIG.scst_lr,
                                         clipvalue=CONFIG.scst_clip)
    for current_states, reward, current_actions, img_url in zip(states, r_values, actions, urls):
        greedy_reward = calculate_greedy_decoding_reward(
            img_url, tokenizer, gmodel, dmodel, train_caps, features, lambda_val)

        for action, state in zip(current_actions, current_states):
            with tf.GradientTape() as tape:
                pre_zero_seq = post_zero_to_pre(state["words"])
                word_probs = gmodel([np.array(state["image"]).reshape((1, 2048)),
                                     np.array([pre_zero_seq])], training=True)
                if np.min(word_probs) == 0:
                    break
                if math.isnan(np.min(word_probs)):
                    print("model reached nan")
                    return gmodel
                loss = calculate_loss(action, reward, word_probs, greedy_reward)

            grads = tape.gradient(loss, gmodel.trainable_variables)
            optimizer.apply_gradients(zip(grads, gmodel.trainable_variables))
    return gmodel


def generate_real_fake_wrong_samples2(n_samples, gmodel, tokenizer, train_caps,
                                      features,
                                      max_length: int = CONFIG.max_length):
    """Fresh discriminator batch during GAN training: ``n/2`` real,
    ``n/4`` generated and ``n/4`` mismatched pairs."""
    all_ids = list(train_caps.keys())
    quarter = int(n_samples / 4)

    generated = generate_fake_captions(gmodel, tokenizer, features, quarter, max_length)
    fake_features = [features[url] for url in generated]
    fake_caps = list(generated.values())

    real_features, real_caps = [], []
    for img_id in random.sample(all_ids, quarter):
        feature = features[img_id_to_url(img_id)]
        real_cap1, real_cap2 = random.sample(train_caps[img_id], 2)
        real_features += [feature, feature]
        real_caps += [real_cap1, real_cap2]

    wrong_features, wrong_caps = [], []
    for img_id in random.sample(all_ids, quarter):
        wrong_features.append(features[img_id_to_url(img_id)])
        wrong_caps.append(random.choice(train_caps[_different_image(img_id, all_ids)]))

    disc_y = np.concatenate((np.ones(2 * quarter), np.zeros(quarter),
                             np.zeros(quarter)), axis=0)
    return _to_disc_inputs(real_features + fake_features + wrong_features,
                           real_caps + fake_caps + wrong_caps,
                           disc_y, tokenizer, max_length)


def train_gan(gen, disc, tokenizer, train_caps, features,
              n_epochs: int = CONFIG.gan_epochs,
              mini_batch: int = CONFIG.mini_batch,
              disc_data_batch: int = CONFIG.disc_data_batch,
              disc_training_batch: int = CONFIG.disc_training_batch,
              disc_epochs: int = CONFIG.disc_inner_epochs,
              lambda_val: float = CONFIG.lambda_val):
    """Alternate SCST generator updates and discriminator updates."""
    for i in range(n_epochs):
        print(f"Epoch {i + 1}/{n_epochs}")
        # generator
        print("generating samples for generator ...")
        fake_batch, actions, _, urls, states = create_fake_captions_batch(
            mini_batch, gen, tokenizer, train_caps, features)
        p_values = calculate_p(fake_batch, disc, tokenizer, features)
        s_values = calculate_s(fake_batch, train_caps)
        r_values = calculate_r(p_values, s_values, lambda_val)
        print("updating generator ...")
        gen = update_generator(r_values, actions, states, gen, urls, disc,
                               tokenizer, train_caps, features, lambda_val)

        # discriminator
        print("generating samples for discriminator ...")
        d_features, d_padded, d_y = generate_real_fake_wrong_samples2(
            disc_data_batch, gen, tokenizer, train_caps, features)
        print("updating discriminator ...")
        disc.fit(x=[d_features.reshape([disc_data_batch, 1, 2048]), d_padded],
                 y=d_y.reshape(disc_data_batch, 1), verbose=1,
                 epochs=disc_epochs, batch_size=disc_training_batch)
    return gen, disc


# --------------------------------------------------------------------------- #
# 4. Evaluation                                                                #
# --------------------------------------------------------------------------- #
def evaluate(gen, tokenizer, test_caps, t_features,
             max_length: int = CONFIG.max_length) -> float:
    """Mean BLEU of greedy captions over the test images."""
    scores = []
    for url, feature in t_features.items():
        caption = generate_caption(tokenizer, feature.reshape((1, 2048)), gen, max_length)
        scores.append(compute_bleu(test_caps[img_url_to_id(url)], caption))
    return float(np.mean(scores))
