# Method & architecture

This document summarises the model and training procedure. It implements
*Improving Image Captioning with Conditional Generative Adversarial Nets*
(Chen et al., AAAI 2019). Code references point at the `src/` package.

## 1. Data & feature caching

- **Dataset:** MS-COCO 2017, `val2017` split (~5 000 images, 5 captions each),
  split 80/20 into train/test ([`src/data.py`](../src/data.py)).
- **The key optimisation:** images are *not* kept in memory. Each image is passed
  **once** through InceptionV3 (ImageNet weights, classification head removed) to
  produce a **2048-d feature vector**, and the `{image_url: vector}` mapping is
  pickled ([`src/features.py`](../src/features.py)). Every later stage reads these
  cached vectors instead of raw pixels — decisive for fitting the pipeline into a
  free Colab GPU's RAM/time budget.

## 2. Vocabulary

- Captions are lower-cased and wrapped with `startseq` / `endseq`.
- A Keras `Tokenizer` keeps the most frequent `num_words` tokens (`OOV` for the
  rest). Embeddings are initialised from **GloVe-100d** and frozen
  ([`src/vocab.py`](../src/vocab.py)).
- `max_length = 16` tokens.

## 3. Generator (encoder–decoder)

A "merge" decoder ([`src/model.py`](../src/model.py)):

```
image (2048)  -> Dropout(0.5) -> Dense(512, relu) ─┐
                                                    add -> Dense(256, relu) -> Dense(V, softmax)
caption (16)  -> Embedding -> Dropout(0.5) -> LSTM(512) ─┘
```

Pre-trained by **maximum likelihood** (categorical cross-entropy, Adam): every
`n`-gram prefix of a caption predicts its next token
([`mle_data_generator`](../src/train.py)).

## 4. Discriminator

An RNN that scores `(image-features, caption)` pairs in `[0, 1]`:

```
image-features ─┐
                concat -> LSTM(512) -> Dense(512, leaky_relu) -> Dropout(0.4) -> Dense(1, sigmoid)
caption -> Embedding ─┘
```

Pre-trained with binary cross-entropy on three kinds of pairs:

| pair type | caption source                              | target |
| --------- | ------------------------------------------- | :----: |
| real      | a human caption of the image                |   1    |
| fake      | a caption produced by the generator         |   0    |
| wrong     | a real human caption of a *different* image |   0    |

## 5. Adversarial refinement with SCST

Caption generation is framed as a Markov decision process
([`src/rl_env.py`](../src/rl_env.py)):

- **state** `{image: 2048-d, words: token sequence}`
- **action** pick the next vocabulary index (`Discrete(num_words, start=1)`)
- **reward** sentence-level BLEU of the caption so far.

The generator (policy) is updated with **Self-Critical Sequence Training**, a
REINFORCE variant whose baseline is the reward of the model's own greedy
decoding:

```
∇θ L = - Σ_t ( r(x^s) - r(x^g) ) ∇θ log πθ(x^s_t | x^s_{1:t-1})
```

where `x^s` is sampled from the policy and `x^g` is the greedy rollout
([`scst_loss`](../src/train.py)). The reward mixes the discriminator score with
BLEU:

```
r = λ · D(caption | image) + (1 − λ) · BLEU      (λ = 0.2)
```

The GAN loop ([`train_gan`](../src/train.py)) alternates: generate a mini-batch,
update the generator via SCST, then refresh the discriminator on fresh
real/fake/wrong samples.

## 6. Hyper-parameters

| stage                 | setting                                                        |
| --------------------- | -------------------------------------------------------------- |
| Generator (MLE)       | Adam, categorical cross-entropy, 10 epochs, batch 3            |
| Discriminator (pre)   | Adam, binary cross-entropy, 15 epochs, batch 16 (~0.88 acc.)   |
| GAN                   | 25 epochs, mini-batch 8, λ = 0.2, 4 discriminator epochs/step  |
| SCST optimiser        | Adam, lr 5e-5, gradient clip 0.5                               |
| Vocabulary / sequence | `num_words = 2075`, `embedding_dim = 100`, `max_length = 16`   |

All values live in [`src/config.py`](../src/config.py).

## 7. Results

Average **BLEU ≈ 0.57** on the held-out test set — about a **10 %** relative
improvement over the plain encoder–decoder baseline. See
[`assets/demo/`](../assets/demo/README.md) for qualitative examples.
