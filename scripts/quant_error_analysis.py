"""
quant_error_analysis.py - recommendation #3: measure quantization error
directly in activation space, instead of inferring it indirectly through
classification metrics.

For each held-out clip, compares the float32 activation at the
GlobalAveragePooling2D layer (the pooled feature vector feeding the final
Dense layer) against the same activation reconstructed from the int8 TFLite
graph (dequantized using that tensor's stored scale/zero-point). Bins the
resulting error by measured SNR, alongside the norm of the fp32 activation
itself (a proxy for how much "signal" reaches that layer).

The question this answers directly: does quantization error grow relative to
signal as SNR drops (would support "quantization matters more when the
acoustic signal is faint"), or does it stay roughly proportional /  flat
(supports "acoustic noise, not quantization noise, is the binding
constraint" - the leading candidate explanation from FINDINGS.md).
"""
import json
import numpy as np
import tensorflow as tf

N_SNR_BINS = 8


def gap_layer_name(keras_model):
    return next(l.name for l in keras_model.layers
                if l.__class__.__name__ == "GlobalAveragePooling2D")


def gap_tensor_index(interp):
    for t in interp.get_tensor_details():
        if "global_average_pooling2d" in t["name"] and t["dtype"] == np.int8:
            return t["index"], t["quantization"]
    raise RuntimeError("GAP tensor not found in int8 graph")


def run(model_name, keras_path, tflite_int8_path, X, snr, tr_idx):
    keras_model = tf.keras.models.load_model(keras_path)
    gap_name = gap_layer_name(keras_model)
    fp32_intermediate = tf.keras.Model(
        keras_model.input, keras_model.get_layer(gap_name).output)

    # preserve_all_tensors: intermediate tensor buffers are normally reused/
    # freed once consumed, so the GAP tensor's data isn't retrievable after
    # invoke() unless this is set.
    interp = tf.lite.Interpreter(model_path=tflite_int8_path,
                                  experimental_preserve_all_tensors=True)
    interp.allocate_tensors()
    in_idx = interp.get_input_details()[0]["index"]
    gap_idx, (scale, zero_point) = gap_tensor_index(interp)

    keep = np.ones(len(snr), dtype=bool)
    keep[tr_idx] = False
    Xe, snre = X[keep], snr[keep]

    fp32_act = fp32_intermediate.predict(Xe, batch_size=64, verbose=0)

    int8_act = np.empty_like(fp32_act)
    for k in range(len(Xe)):
        interp.set_tensor(in_idx, Xe[k:k+1].astype(np.float32))
        interp.invoke()
        raw = interp.get_tensor(gap_idx).astype(np.float32)
        int8_act[k] = (raw - zero_point) * scale

    fp32_norm = np.linalg.norm(fp32_act, axis=1)
    quant_err = np.linalg.norm(fp32_act - int8_act, axis=1)
    rel_err = quant_err / (fp32_norm + 1e-8)

    edges = np.linspace(snre.min(), snre.max(), N_SNR_BINS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bins = []
    for b in range(N_SNR_BINS):
        m = (snre >= edges[b]) & (snre < edges[b + 1])
        if m.sum() == 0:
            continue
        bins.append(dict(
            snr_center=float(centers[b]), n=int(m.sum()),
            fp32_activation_norm_mean=float(fp32_norm[m].mean()),
            quant_error_norm_mean=float(quant_err[m].mean()),
            relative_quant_error_mean=float(rel_err[m].mean()),
        ))

    corr = float(np.corrcoef(snre, rel_err)[0, 1])
    result = dict(model=model_name, gap_layer=gap_name,
                  corr_snr_vs_relative_error=corr, bins=bins)
    print(f"\n[{model_name}] corr(SNR, relative quant error) = {corr:+.3f}")
    print("SNR(dB)   fp32_act_norm   quant_err_norm   rel_err")
    for r in bins:
        print(f"{r['snr_center']:7.1f}   {r['fp32_activation_norm_mean']:11.4f}   "
              f"{r['quant_error_norm_mean']:13.4f}   {r['relative_quant_error_mean']:.4f}")
    return result


def main():
    meta = json.load(open("data/processed/meta.json"))
    d = np.load("data/processed/clips.npz", allow_pickle=True)
    X, snr = d["X"], d["snr_db"]
    tr_idx = np.load("data/processed/train_idx.npy")

    results = [
        run("full",  "models/fp32.keras",       "models/int8.tflite",       X, snr, tr_idx),
        run("small", "models/fp32_small.keras", "models/int8_small.tflite", X, snr, tr_idx),
    ]
    json.dump(results, open("supplementary/exploratory_results/results_quant_error.json", "w"), indent=2)
    print("\nwrote results_quant_error.json")


if __name__ == "__main__":
    main()
