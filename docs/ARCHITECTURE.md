# Method & architecture

This document summarises the model and training procedure. It implements
*Improving Image Captioning with Conditional Generative Adversarial Nets*
(Chen et al., AAAI 2019), simplified to fit a free Google Colab GPU (see
[§8](#8-relation-to-the-paper)). Code references point at the `src/` package and
mirror [`Image_Captioning.ipynb`](../Image_Captioning.ipynb).

## 1. Data & feature caching

- **Dataset:** MS-COCO 2017 `val2017` split (5 000 images, 5–6 captions each),
  split randomly 80/20 into 4 000 train / 1 000 test images
  ([`src/data.py`](../src/data.py)). Download links and folder layout are in the
  [README](../README.md#dataset).
- **The key optimisation:** images are *not* kept in memory. Each image passes
  **once** through InceptionV3 (ImageNet weights, classification head removed) to
  produce a **2048-d feature vector**, and the `{image_url: vector}` mapping is
  pickled ([`src/features.py`](../src/features.py)). Every later stage reads these
  cached vectors instead of raw pixels, which is what lets the pipeline fit into a
  free Colab GPU's RAM/time budget.

## 2. Vocabulary

- Captions are wrapped with `startseq` / `endseq` and grouped per image
  ([`src/vocab.py`](../src/vocab.py)).
- A Keras `Tokenizer` (lower-cases, strips punctuation) keeps the most frequent
  `num_words = 2075` tokens, with `OOV` for the rest.
- Word embeddings are **learned** by the decoder (`embedding_dim = 2048`); no
  pre-trained word vectors are used.
- `max_length = 16` tokens.

## 3. Generator (encoder–decoder)

A "merge" decoder ([`src/model.py`](../src/model.py)) on top of the frozen,
cached InceptionV3 features:

```
image (2048)  -> Dropout(0.5) -> Dense(512, relu) ─┐
                                                    add -> Dense(256, relu) -> Dense(V, softmax)
caption (16)  -> Embedding(2048) -> Dropout(0.5) -> LSTM(512) ─┘
```

Pre-trained by **maximum likelihood** (categorical cross-entropy, Adam): every
prefix of a caption predicts its next token
([`mle_data_generator`](../src/train.py)). At test time captions are decoded
greedily ([`src/inference.py`](../src/inference.py)).

## 4. Discriminator

An LSTM that scores `(image-features, caption)` pairs in `[0, 1]`. The image
vector is fed as the first time step and the caption words follow, which is why
the word embedding has the same size (2048) as the image features:

```
image-features (1 × 2048) ─┐
                           concat -> LSTM(512) -> Dense(512, leaky_relu) -> Dense(1, sigmoid)
caption -> Embedding ──────┘
```

The discriminator's embedding is copied from the pre-trained generator and
frozen. It is pre-trained with binary cross-entropy on 16 000 pairs:

| pair type | caption source                              | count  | target |
| --------- | ------------------------------------------- | -----: | :----: |
| real      | a human caption of the image (2 per image)  |  8 000 |   1    |
| fake      | the MLE generator's caption for the image   |  4 000 |   0    |
| wrong     | a human caption of a *different* image      |  4 000 |   0    |

## 5. Adversarial refinement with SCST

Caption generation is framed as a decision process
([`src/rl_env.py`](../src/rl_env.py)):

- **state** `{image: 2048-d, words: token sequence}`
- **action** the next vocabulary index (`Discrete(num_words, start=1)`)
- **reward** BLEU of the caption so far ([`src/metrics.py`](../src/metrics.py)).

The generator (policy) is updated with **Self-Critical Sequence Training**, a
REINFORCE variant whose baseline is the reward of the model's own greedy
decoding:

```
∇θ L = - Σ_t ( r(x^s) - r(x^g) ) ∇θ log πθ(x^s_t | x^s_{1:t-1})
```

where `x^s` is sampled from the policy and `x^g` is the greedy caption
([`calculate_loss`](../src/train.py)). Both rewards mix the discriminator score
`p` with the language score `s` (BLEU):

```
r = λ · p + (1 − λ) · s      (λ = 0.2)
```

The GAN loop ([`train_gan`](../src/train.py)) alternates two steps: sample 8
captions and update the generator via SCST (one gradient step per generated
word), then refresh the discriminator on 128 fresh real / fake / wrong pairs.

## 6. Hyper-parameters

| stage                 | setting                                                              |
| --------------------- | -------------------------------------------------------------------- |
| Generator (MLE)       | Adam, categorical cross-entropy, 10 epochs, batch of 3 images        |
| Discriminator (pre)   | Adam, binary cross-entropy, 15 epochs, batch 16 (~0.89 accuracy)     |
| GAN                   | 25 iterations, mini-batch 8, λ = 0.2, 128 discriminator samples/step, 4 discriminator epochs/step |
| SCST optimiser        | Adam, lr 5e-5, gradient clip 0.5                                     |
| Vocabulary / sequence | `num_words = 2075`, `embedding_dim = 2048`, `max_length = 16`        |

These are the settings of the thesis run. The last run saved in the notebook
uses 1 discriminator epoch per GAN step instead of 4. All values live in
[`src/config.py`](../src/config.py).

## 7. Results

The thesis reports an average **BLEU ≈ 0.57** on the test images, about 10 %
above the MLE-only encoder–decoder. The later run saved in the notebook (1
discriminator epoch per GAN step) printed 0.546 over the 1 000 test images. See
[`assets/demo/`](../assets/demo/README.md) for qualitative examples.

**Caveat:** `compute_bleu` hands raw strings to NLTK's `sentence_bleu`, which
then compares **characters** instead of words. On the notebook's own example,
character-level BLEU is 0.63 where word-level BLEU-4 is 0.51. The scores are
consistent within this project, which also uses them as the RL reward, but they
are not comparable to the word-level BLEU-4 reported in captioning papers.
`compute_bleu(..., word_level=True)` gives the conventional score.

## 8. Relation to the paper

This is an unofficial, simplified student re-implementation, not the authors'
code. It keeps the paper's core idea:

- a captioning **generator** plus a **discriminator** that separates human
  captions from generated and mismatched ones;
- a reward mixing the discriminator score with a language-evaluation score,
  `r = λ·p + (1 − λ)·s`;
- training the generator with **SCST** (policy gradient with a greedy-decoding
  baseline), alternating with discriminator updates.

What is different or left out:

| Aspect | This implementation |
| ------ | ------------------- |
| Discriminator | Only the **RNN-based** (LSTM) discriminator; no CNN-based discriminator. |
| Ensembles | **None**: a single generator/discriminator pair, no ensemble of CNN-GAN / RNN-GAN models. |
| Image encoder | InceptionV3 is **frozen**; its features are computed once and cached. RL updates **only the decoder** (embedding, LSTM and dense layers, including the image projection). |
| Generator | Plain merge LSTM decoder on one global image vector; **no attention** and no bottom-up / top-down region features. |
| Language evaluator | BLEU only (character-level, see §7); no CIDEr. |
| Data & scale | 4 000 training images from COCO `val2017` instead of the full COCO training set; 25 short GAN iterations. |

Because of these differences, the results are not comparable to the numbers in
the paper. Cite the original paper for the method; this repository can be
referenced as an educational implementation of it.
