"""
train_small.py - follow-up experiment: does a smaller CNN (less precision budget
to lose) show a real fp32/int8 sensitivity gap where the original model didn't?

Same data, same SNR-based train/eval split as train.py. Only the architecture
shrinks: 3 conv blocks (16/32/32 filters) -> 2 conv blocks (4/8 filters).
Everything else (loss, threshold logic downstream, SNR binning) is held fixed
so the comparison isolates model capacity as the variable.
"""
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

SEED = 42
tf.random.set_seed(SEED); np.random.seed(SEED)
HIGH_SNR_TRAIN_PERCENTILE = 60  # same threshold as train.py, for comparability


def build_small(input_shape):
    inp = layers.Input(input_shape)
    x = layers.Conv2D(4, 3, activation="relu", padding="same")(inp)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(8, 3, activation="relu", padding="same")(x)
    x = layers.GlobalAveragePooling2D()(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return models.Model(inp, out)


def main():
    d = np.load("data/processed/clips.npz", allow_pickle=True)
    X, y, snr, pos = d["X"], d["y"], d["snr_db"], d["position"]
    meta = json.load(open("data/processed/meta.json"))

    fault_snr = snr[y == 1]
    thr = np.percentile(fault_snr, HIGH_SNR_TRAIN_PERCENTILE)
    fault_idx = np.where((y == 1) & (snr >= thr))[0]

    close_positions = set(pos[fault_idx].tolist())
    normal_idx = np.where((y == 0) & np.isin(pos, list(close_positions)))[0]

    rng = np.random.RandomState(SEED)
    n = min(len(fault_idx), len(normal_idx))
    fault_sel  = rng.choice(fault_idx,  n, replace=False)
    normal_sel = rng.choice(normal_idx, n, replace=False)
    idx = np.concatenate([fault_sel, normal_sel])
    rng.shuffle(idx)

    n_val = int(0.2 * len(idx))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    np.save("data/processed/train_idx_small.npy", tr_idx)

    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xval, yval = X[val_idx], y[val_idx]
    print(f"[small] train on high-SNR fault + position-matched normal: n_train={len(tr_idx)} "
          f"n_val={len(val_idx)}  fault_SNR >= {thr:.1f} dB  "
          f"positions={sorted(close_positions)}  balance={int(ytr.mean()*100)}% fault")

    m = build_small(tuple(meta["shape"]))
    print(f"[small] param count: {m.count_params()}")
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy",
              metrics=[tf.keras.metrics.AUC(name="auc")])
    cb = [tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                           patience=8, restore_best_weights=True)]
    m.fit(Xtr, ytr, validation_data=(Xval, yval),
          epochs=60, batch_size=32, callbacks=cb)
    m.save("models/fp32_small.keras")
    m.export("models/fp32_small_sm")
    print("saved models/fp32_small.keras + models/fp32_small_sm")


if __name__ == "__main__":
    import os; os.makedirs("models", exist_ok=True)
    main()
