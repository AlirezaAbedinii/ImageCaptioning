"""GAN-based image captioning — source package.

Modules
-------
config      Central paths and hyper-parameters.
data        COCO loading, train/test split and dataframe building.
features    InceptionV3 encoder that caches 2048-d image feature vectors.
vocab       Caption dictionaries (startseq/endseq markers) and the tokenizer.
model       Generator (LSTM decoder) and discriminator.
metrics     BLEU, used for evaluation and as the RL reward.
rl_env      Custom Gym environment used for SCST reinforcement learning.
train       MLE pre-training, discriminator pre-training and the GAN + SCST loop.
inference   Greedy caption generation from a single image.
"""
