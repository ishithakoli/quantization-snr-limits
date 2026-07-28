"""
model_variants.py - shared CNN architecture spec for the capacity sweep
(recommendation #2). Each spec is a list of (filters, kernel_size) conv
blocks; every block but the last is followed by MaxPool2D, the last by
GlobalAveragePooling2D (matches the shape of the original train.py/
train_small.py architectures exactly, so "full" and "small" below reproduce
those two models).
"""
from tensorflow.keras import layers, models

BUILD_SPECS = {
    "full":   [(16, 3), (32, 3), (32, 3)],   # original train.py
    "medium": [(8, 3), (16, 3), (16, 3)],    # half filters, same depth
    "tiny":   [(8, 3), (16, 3)],             # 2 blocks, small.py's filter count
    "small":  [(4, 3), (8, 3)],              # original train_small.py
}


def build(spec, input_shape):
    inp = layers.Input(input_shape)
    x = inp
    for i, (filters, k) in enumerate(spec):
        x = layers.Conv2D(filters, k, activation="relu", padding="same")(x)
        if i < len(spec) - 1:
            x = layers.MaxPool2D()(x)
        else:
            x = layers.GlobalAveragePooling2D()(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return models.Model(inp, out)
