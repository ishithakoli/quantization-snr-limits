"""
sweep.py - capacity sweep (recommendation #2) + AUC-based evaluation
(recommendation #1), replacing the earlier full-vs-small-only comparison.

Trains "medium" and "tiny" variants (see model_variants.py) on the SAME
train/val split already used for "full" (train.py) and "small"
(train_small.py) — the split is deterministic given SEED=42 and depends only
on the data, not the model, so it's identical across all four (verified:
data/processed/train_idx.npy == train_idx_small.npy). This makes the four
variants directly comparable on the same held-out set.

Reuses the already-trained "full" and "small" TFLite models instead of
retraining them. Evaluation uses ROC AUC per SNR bin (and overall) instead of
a single fixed-threshold sensitivity number, because the small-model
follow-up showed fixed-threshold sensitivity is unstable for tiny,
near-degenerate models — AUC judges ranking quality independent of where the
threshold sits.
"""
import json, os, time
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from model_variants import BUILD_SPECS, build

SEED = 42
tf.random.set_seed(SEED); np.random.seed(SEED)
HIGH_SNR_TRAIN_PERCENTILE = 60
N_SNR_BINS = 8
TO_TRAIN = ["medium", "tiny"]          # "full" and "small" already exist
ALREADY_TRAINED = {
    "full":  ("models/fp32.tflite",       "models/int8.tflite"),
    "small": ("models/fp32_small.tflite", "models/int8_small.tflite"),
}


def get_split(meta_shape):
    d = np.load("data/processed/clips.npz", allow_pickle=True)
    X, y, snr, pos = d["X"], d["y"], d["snr_db"], d["position"]

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
    return X, y, snr, tr_idx, val_idx


def train_and_export(name, spec, X, y, tr_idx, val_idx, input_shape):
    m = build(spec, input_shape)
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy",
              metrics=[tf.keras.metrics.AUC(name="auc")])
    cb = [tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                           patience=8, restore_best_weights=True)]
    m.fit(X[tr_idx], y[tr_idx], validation_data=(X[val_idx], y[val_idx]),
          epochs=60, batch_size=32, callbacks=cb, verbose=0)
    val_auc = m.evaluate(X[val_idx], y[val_idx], verbose=0)[1]
    print(f"[{name}] params={m.count_params()}  val_auc={val_auc:.4f}")

    sm_path = f"models/fp32_{name}_sm"
    m.export(sm_path)

    conv = tf.lite.TFLiteConverter.from_saved_model(sm_path)
    fp32_bytes = conv.convert()
    fp32_path = f"models/fp32_{name}.tflite"
    open(fp32_path, "wb").write(fp32_bytes)

    tr_idx_arr = tr_idx
    sel = np.random.RandomState(0).choice(len(tr_idx_arr), min(200, len(tr_idx_arr)), replace=False)
    Xtr = X[tr_idx_arr]
    def gen():
        for i in sel:
            yield [Xtr[i:i+1].astype("float32")]
    conv2 = tf.lite.TFLiteConverter.from_saved_model(sm_path)
    conv2.optimizations = [tf.lite.Optimize.DEFAULT]
    conv2.representative_dataset = gen
    int8_bytes = conv2.convert()
    int8_path = f"models/int8_{name}.tflite"
    open(int8_path, "wb").write(int8_bytes)

    return fp32_path, int8_path, m.count_params()


def predict_tflite(path, X):
    interp = tf.lite.Interpreter(model_path=path); interp.allocate_tensors()
    i = interp.get_input_details()[0]; o = interp.get_output_details()[0]
    out = []
    for k in range(len(X)):
        xb = X[k:k+1].astype(i["dtype"])
        interp.set_tensor(i["index"], xb); interp.invoke()
        out.append(float(interp.get_tensor(o["index"]).ravel()[0]))
    return np.array(out)


def safe_auc(y, scores):
    if len(set(y.tolist())) < 2:
        return None
    return float(roc_auc_score(y, scores))


def evaluate_variant(name, fp32_path, int8_path, X, y, snr, tr_idx):
    keep = np.ones(len(y), dtype=bool)
    keep[tr_idx] = False
    Xe, ye, snre = X[keep], y[keep], snr[keep]

    fp32_scores = predict_tflite(fp32_path, Xe)
    int8_scores = predict_tflite(int8_path, Xe)

    overall_fp32 = safe_auc(ye, fp32_scores)
    overall_int8 = safe_auc(ye, int8_scores)

    edges = np.linspace(snre.min(), snre.max(), N_SNR_BINS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bins = []
    for b in range(N_SNR_BINS):
        m = (snre >= edges[b]) & (snre < edges[b + 1])
        if m.sum() == 0:
            continue
        auc_fp32 = safe_auc(ye[m], fp32_scores[m])
        auc_int8 = safe_auc(ye[m], int8_scores[m])
        bins.append(dict(snr_center=float(centers[b]), n=int(m.sum()),
                          auc_fp32=auc_fp32, auc_int8=auc_int8,
                          gap=None if (auc_fp32 is None or auc_int8 is None)
                              else float(auc_fp32 - auc_int8)))

    gaps = [b["gap"] for b in bins if b["gap"] is not None]
    return dict(name=name, overall_auc_fp32=overall_fp32, overall_auc_int8=overall_int8,
                overall_gap=None if (overall_fp32 is None or overall_int8 is None)
                    else float(overall_fp32 - overall_int8),
                max_abs_gap=float(max(abs(g) for g in gaps)) if gaps else None,
                mean_gap=float(np.mean(gaps)) if gaps else None, bins=bins)


def main():
    meta = json.load(open("data/processed/meta.json"))
    input_shape = tuple(meta["shape"])
    X, y, snr, tr_idx, val_idx = get_split(input_shape)

    os.makedirs("models", exist_ok=True)
    paths = dict(ALREADY_TRAINED)
    params = {"full": None, "small": None}
    for name in TO_TRAIN:
        fp32_p, int8_p, n_params = train_and_export(
            name, BUILD_SPECS[name], X, y, tr_idx, val_idx, input_shape)
        paths[name] = (fp32_p, int8_p)
        params[name] = n_params

    results = []
    for name in ["full", "medium", "tiny", "small"]:
        fp32_p, int8_p = paths[name]
        r = evaluate_variant(name, fp32_p, int8_p, X, y, snr, tr_idx)
        r["params"] = params.get(name)
        results.append(r)
        gap_str = f"{r['overall_gap']:+.4f}" if r['overall_gap'] is not None else "n/a"
        print(f"[{name}] overall AUC fp32={r['overall_auc_fp32']:.4f}  "
              f"int8={r['overall_auc_int8']:.4f}  gap={gap_str}  "
              f"max|gap|(per-bin)={r['max_abs_gap']:.4f}" if r['max_abs_gap'] is not None
              else f"[{name}] insufficient bins for per-bin AUC")

    json.dump(results, open("supplementary/exploratory_results/results_capacity_sweep.json", "w"), indent=2)
    print("\nwrote results_capacity_sweep.json")


if __name__ == "__main__":
    main()
