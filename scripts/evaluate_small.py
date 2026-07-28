"""
evaluate_small.py - sensitivity vs measured SNR, fp32 vs int8, for the small CNN.
Same procedure as evaluate.py (fixed ~5% FPR threshold on normals, SNR bins,
per-trial error bars), excluding clips used in train_small.py.
"""
import json
import numpy as np
import tensorflow as tf

N_SNR_BINS = 8


def predict_tflite(path, X):
    interp = tf.lite.Interpreter(model_path=path); interp.allocate_tensors()
    i = interp.get_input_details()[0]; o = interp.get_output_details()[0]
    out = []
    for k in range(len(X)):
        xb = X[k:k+1].astype(i["dtype"])
        interp.set_tensor(i["index"], xb); interp.invoke()
        out.append(float(interp.get_tensor(o["index"]).ravel()[0]))
    return np.array(out)


def sensitivity(scores, y, thr):
    fault = y == 1
    return float(np.mean(scores[fault] >= thr)) if fault.any() else np.nan


def main():
    d = np.load("data/processed/clips.npz", allow_pickle=True)
    X, y, snr, trial = d["X"], d["y"], d["snr_db"], d["trial"]
    tr_idx = set(np.load("data/processed/train_idx_small.npy").tolist())
    keep = np.array([k not in tr_idx for k in range(len(y))])
    X, y, snr, trial = X[keep], y[keep], snr[keep], trial[keep]

    fp32_all = predict_tflite("models/fp32_small.tflite", X)
    int8_all = predict_tflite("models/int8_small.tflite", X)
    normals = y == 0
    thr_fp32 = np.quantile(fp32_all[normals], 0.95)
    thr_int8 = np.quantile(int8_all[normals], 0.95)

    edges = np.linspace(snr.min(), snr.max(), N_SNR_BINS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    rows = []
    for b in range(N_SNR_BINS):
        m = (snr >= edges[b]) & (snr < edges[b+1])
        if m.sum() == 0:
            continue
        per_trial_fp32, per_trial_int8 = [], []
        for t in sorted(set(trial[m].tolist())):
            mt = m & (trial == t)
            if (y[mt] == 1).sum() == 0:
                continue
            per_trial_fp32.append(sensitivity(fp32_all[mt], y[mt], thr_fp32))
            per_trial_int8.append(sensitivity(int8_all[mt], y[mt], thr_int8))
        if not per_trial_fp32:
            continue
        rows.append(dict(
            snr_center=float(centers[b]),
            fp32_mean=float(np.mean(per_trial_fp32)),
            fp32_sd=float(np.std(per_trial_fp32)),
            int8_mean=float(np.mean(per_trial_int8)),
            int8_sd=float(np.std(per_trial_int8)),
            n_fault=int((y[m] == 1).sum()),
        ))

    json.dump({"threshold_fp32": float(thr_fp32),
               "threshold_int8": float(thr_int8),
               "bins": rows}, open("supplementary/exploratory_results/results_sensitivity_small.json", "w"), indent=2)
    print("[small] SNR(dB)  fp32_sens  int8_sens  gap")
    for r in rows:
        print(f"{r['snr_center']:6.1f}   {r['fp32_mean']:.3f}      "
              f"{r['int8_mean']:.3f}     {r['fp32_mean']-r['int8_mean']:+.3f}")
    print("\nwrote results_sensitivity_small.json")


if __name__ == "__main__":
    main()
