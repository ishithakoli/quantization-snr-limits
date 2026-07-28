"""
cross_fan_eval.py - reframed hypothesis: does int8 quantization degrade
cross-source generalization (a different, quieter fan, never seen in
training) more than fp32?

The fp32/int8 models were trained only on session 1 (the original loud fan,
high-SNR clips). Session 1's held-out clips are the in-domain test set
(already scored in results_sensitivity.json). Session 2 (quiet fan) was
never used in training or threshold calibration - it's a clean out-of-domain
test set for exactly this question. Reuses the operating thresholds already
calibrated in evaluate.py so in-domain and out-of-domain numbers are
comparable at the same decision boundary.
"""
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
import prepare

SESSION2_POSITIONS_CM = {"10": "10cm_s2", "25": "25cm_s2", "51": "51cm_s2",
                          "102": "102cm_s2", "203": "203cm_s2", "254": "254cm_s2",
                          "305": "305cm_s2", "381": "381cm_s2", "508": "508cm_s2"}


def load_session2():
    import glob, os, re
    amb_e = prepare.ambient_band_energy("2")
    X, y, snr, pos, trial = [], [], [], [], []
    for posdir in sorted(SESSION2_POSITIONS_CM.values()):
        pdir = os.path.join("data/raw", posdir)
        for label_name, label in [("normal", 0), ("fault", 1)]:
            audio_files = sorted(glob.glob(os.path.join(pdir, label_name, "*.m4a")) +
                                 glob.glob(os.path.join(pdir, label_name, "*.wav")))
            for wav in audio_files:
                t = int(re.search(r"trial(\d+)", wav).group(1))
                for seg in prepare.clips_of(prepare.load(wav)):
                    X.append(prepare.logmel(seg)); y.append(label)
                    snr.append(prepare.clip_snr_db(seg, amb_e))
                    pos.append(posdir); trial.append(t)
    X = np.stack(X)[..., None]
    return X, np.array(y), np.array(snr), np.array(pos), np.array(trial)


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
    thresholds = json.load(open("results/results_sensitivity.json"))
    thr_fp32, thr_int8 = thresholds["threshold_fp32"], thresholds["threshold_int8"]

    print("Loading session 2 (quiet fan, out-of-domain, never trained on)...")
    X2, y2, snr2, pos2, trial2 = load_session2()

    fp32_scores = predict_tflite("models/fp32.tflite", X2)
    int8_scores = predict_tflite("models/int8.tflite", X2)

    ood_sens_fp32 = sensitivity(fp32_scores, y2, thr_fp32)
    ood_sens_int8 = sensitivity(int8_scores, y2, thr_int8)
    ood_auc_fp32 = float(roc_auc_score(y2, fp32_scores))
    ood_auc_int8 = float(roc_auc_score(y2, int8_scores))

    # in-domain reference, SNR-matched to session 2's range: pooling over the
    # FULL in-domain SNR range (now including sessions 3/4's low-SNR clips)
    # would compare "full range" against "high-SNR-only" and call the
    # difference a domain-generalization gap, which it isn't - restrict to
    # the same SNR band session 2 actually occupies for a fair comparison.
    d = np.load("data/processed/clips.npz", allow_pickle=True)
    X1, y1, snr1 = d["X"], d["y"], d["snr_db"]
    tr_idx = set(np.load("data/processed/train_idx.npy").tolist())
    held_out = np.array([k not in tr_idx for k in range(len(y1))])
    # match on FAULT-clip SNR specifically, since sensitivity = recall on
    # fault clips - matching on the whole array (incl. normals) would let
    # session 2's much-lower normal-clip SNRs widen the band artificially.
    # 5th/95th percentile, not min/max: the raw min is dragged to -0.6 dB by
    # 8 clips from the known 305cm_s2/trial2 dead-air tail (documented
    # elsewhere) - a recording artifact, not a real low-SNR fault reading.
    fault_snr2 = snr2[y2 == 1]
    snr_lo, snr_hi = np.percentile(fault_snr2, [5, 95])
    matched = held_out & (snr1 >= snr_lo) & (snr1 <= snr_hi)
    X1m, y1m = X1[matched], y1[matched]

    fp32_id_scores = predict_tflite("models/fp32.tflite", X1m)
    int8_id_scores = predict_tflite("models/int8.tflite", X1m)
    id_sens_fp32 = sensitivity(fp32_id_scores, y1m, thr_fp32)
    id_sens_int8 = sensitivity(int8_id_scores, y1m, thr_int8)
    print(f"In-domain comparison restricted to SNR [{snr_lo:.1f}, {snr_hi:.1f}] dB "
          f"(session 2's range) - n={matched.sum()} held-out same-fan clips in that band")

    result = dict(
        in_domain_sensitivity=dict(fp32=id_sens_fp32, int8=id_sens_int8),
        out_of_domain_sensitivity=dict(fp32=ood_sens_fp32, int8=ood_sens_int8),
        out_of_domain_auc=dict(fp32=ood_auc_fp32, int8=ood_auc_int8),
        generalization_gap_sensitivity=dict(
            fp32=id_sens_fp32 - ood_sens_fp32,
            int8=id_sens_int8 - ood_sens_int8),
    )
    result["quantization_widens_gap_by"] = (
        result["generalization_gap_sensitivity"]["int8"] -
        result["generalization_gap_sensitivity"]["fp32"])

    # per-position (session 2) breakdown, sorted by nominal cm
    by_pos = []
    for p in sorted(set(pos2.tolist()), key=lambda s: int(s.split("cm")[0])):
        m = pos2 == p
        by_pos.append(dict(
            position=p, snr_mean=float(snr2[m & (y2 == 1)].mean()) if (m & (y2==1)).any() else None,
            fp32_sens=sensitivity(fp32_scores[m], y2[m], thr_fp32),
            int8_sens=sensitivity(int8_scores[m], y2[m], thr_int8)))
    result["by_position"] = by_pos

    json.dump(result, open("results/results_cross_fan.json", "w"), indent=2)

    print(f"\nIn-domain (session 1, same fan)   sensitivity: fp32={id_sens_fp32:.3f}  int8={id_sens_int8:.3f}")
    print(f"Out-of-domain (session 2, quiet fan) sensitivity: fp32={ood_sens_fp32:.3f}  int8={ood_sens_int8:.3f}")
    print(f"Out-of-domain AUC:                                fp32={ood_auc_fp32:.3f}  int8={ood_auc_int8:.3f}")
    print(f"\nGeneralization gap (in-domain - out-of-domain):")
    print(f"  fp32: {result['generalization_gap_sensitivity']['fp32']:+.3f}")
    print(f"  int8: {result['generalization_gap_sensitivity']['int8']:+.3f}")
    print(f"  int8 gap - fp32 gap: {result['quantization_widens_gap_by']:+.3f}  "
          f"({'int8 generalizes WORSE' if result['quantization_widens_gap_by'] > 0 else 'int8 generalizes AS WELL OR BETTER'})")
    print("\nPer-position (session 2, quiet fan):")
    print("position    SNR(dB)  fp32_sens  int8_sens")
    for r in by_pos:
        snr_str = f"{r['snr_mean']:7.1f}" if r['snr_mean'] is not None else "    n/a"
        print(f"{r['position']:10s} {snr_str}   {r['fp32_sens']:.3f}      {r['int8_sens']:.3f}")
    print("\nwrote results_cross_fan.json")


if __name__ == "__main__":
    main()
