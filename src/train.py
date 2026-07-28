"""
train.py - small CNN classifier, Normal vs Fault.

Key design decision: train ONLY on high-SNR clips (the closest positions), where
the fault is unambiguous. Then we test across the full SNR range. This mirrors the
real scenario (model trained on clean examples, deployed where signal is faint)
and makes "sensitivity falls as SNR drops" a meaningful measurement rather than
something we trained away.
"""
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

SEED = 42
tf.random.set_seed(SEED); np.random.seed(SEED)
HIGH_SNR_TRAIN_PERCENTILE = 60  # train on clips above this SNR percentile


def build(input_shape):
    inp = layers.Input(input_shape)
    x = layers.Conv2D(16, 3, activation="relu", padding="same")(inp)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return models.Model(inp, out)


def main():
    d = np.load("data/processed/clips.npz", allow_pickle=True)
    X, y, snr, pos = d["X"], d["y"], d["snr_db"], d["position"]
    meta = json.load(open("data/processed/meta.json"))

    # Apply the SNR threshold to FAULT clips only — normal clips have no fault-band
    # signal by definition, so their SNR is always near 0 dB regardless of position.
    fault_snr = snr[y == 1]
    thr = np.percentile(fault_snr, HIGH_SNR_TRAIN_PERCENTILE)

    fault_idx = np.where((y == 1) & (snr >= thr))[0]

    # IMPORTANT: restrict normal clips to the same positions as the selected fault clips.
    # Drawing normals from all distances would let the model learn "far-room tone = normal"
    # as a shortcut, contaminating the SNR-sensitivity measurement we care about.
    close_positions = set(pos[fault_idx].tolist())
    normal_idx = np.where((y == 0) & np.isin(pos, list(close_positions)))[0]

    # balance: equal normal and fault counts
    rng = np.random.RandomState(SEED)
    n = min(len(fault_idx), len(normal_idx))
    fault_sel  = rng.choice(fault_idx,  n, replace=False)
    normal_sel = rng.choice(normal_idx, n, replace=False)
    idx = np.concatenate([fault_sel, normal_sel])
    rng.shuffle(idx)

    n_val = int(0.2 * len(idx))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    np.save("data/processed/train_idx.npy", tr_idx)

    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xval, yval = X[val_idx], y[val_idx]
    print(f"train on high-SNR fault + position-matched normal: n_train={len(tr_idx)} "
          f"n_val={len(val_idx)}  fault_SNR >= {thr:.1f} dB  "
          f"positions={sorted(close_positions)}  balance={int(ytr.mean()*100)}% fault")

    m = build(tuple(meta["shape"]))
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy",
              metrics=[tf.keras.metrics.AUC(name="auc")])
    cb = [tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                           patience=8, restore_best_weights=True)]
    m.fit(Xtr, ytr, validation_data=(Xval, yval),
          epochs=60, batch_size=32, callbacks=cb)
    m.save("models/fp32.keras")
    # Also export as TF SavedModel — required for TFLite conversion with Keras 3
    m.export("models/fp32_sm")
    print("saved models/fp32.keras + models/fp32_sm")


if __name__ == "__main__":
    import os; os.makedirs("models", exist_ok=True)
    main()
