"""GAN-based image captioning — source package.

Modules
-------
config      Central paths and hyper-parameters.
data        COCO loading, train/test split and dataframe building.
features    InceptionV3 encoder that caches 2048-d image feature vectors.
vocab       Keras tokenizer and GloVe embedding matrix.
model       Generator (CNN encoder + LSTM decoder) and discriminator.
rl_env      Custom Gym environment used for SCST reinforcement learning.
train       MLE pre-training and the adversarial (GAN + SCST) training loop.
inference   Greedy caption generation from a single image.
"""
