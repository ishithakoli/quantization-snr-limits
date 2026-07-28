"""
quantize.py - post-training int8 quantization + size/latency numbers.

The representative dataset uses HIGH-SNR training clips (same distribution the
model was trained on) - this is the realistic calibration a practitioner would do,
and it's the honest setup: we are NOT calibrating on faint clips, because in
deployment you wouldn't have them.
"""
import os, time, json
import numpy as np
import tensorflow as tf


def representative():
    d = np.load("data/processed/clips.npz", allow_pickle=True)
    tr_idx = np.load("data/processed/train_idx.npy")
    X = d["X"][tr_idx]
    sel = np.random.RandomState(0).choice(len(X), min(200, len(X)), replace=False)
    def gen():
        for i in sel:
            yield [X[i:i+1].astype("float32")]
    return gen


def main():
    # Use from_saved_model — avoids MLIR crash in TF 2.16 + Keras 3 on macOS
    conv = tf.lite.TFLiteConverter.from_saved_model("models/fp32_sm")
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = representative()
    tfl = conv.convert()
    open("models/int8.tflite", "wb").write(tfl)

    # plain fp32 tflite for fair latency comparison
    conv2 = tf.lite.TFLiteConverter.from_saved_model("models/fp32_sm")
    open("models/fp32.tflite", "wb").write(conv2.convert())

    size_int8 = os.path.getsize("models/int8.tflite") / 1024
    size_fp32 = os.path.getsize("models/fp32.tflite") / 1024
    print(f"int8: {size_int8:.1f} KB   fp32: {size_fp32:.1f} KB   "
          f"ratio: {size_fp32/size_int8:.1f}x")

    lat = {}
    for name in ["fp32.tflite", "int8.tflite"]:
        interp = tf.lite.Interpreter(model_path=f"models/{name}")
        interp.allocate_tensors()
        i = interp.get_input_details()[0]; o = interp.get_output_details()[0]
        x = np.random.rand(*i["shape"]).astype(i["dtype"])
        ts = []
        for _ in range(200):
            t = time.perf_counter()
            interp.set_tensor(i["index"], x); interp.invoke(); interp.get_tensor(o["index"])
            ts.append((time.perf_counter()-t)*1000)
        lat[name] = float(np.median(ts))
        print(f"{name}: median {lat[name]:.2f} ms")

    json.dump({"size_kb": {"int8": size_int8, "fp32": size_fp32},
               "latency_ms": lat}, open("results/results_edge.json", "w"), indent=2)


if __name__ == "__main__":
    main()
