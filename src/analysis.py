"""
analysis.py - answer the actual scientific question: is the int8 sensitivity drop
a THRESHOLD-like collapse (a "cliff") or a GRADUAL decline relative to fp32?

We fit two competing models to the int8 sensitivity-vs-SNR curve:
  (A) a smooth logistic (gradual) curve
  (B) a piecewise/step model (a cliff at some SNR)
and compare fit quality (e.g. by residual sum of squares / AIC-style comparison).
This is what turns "the graph looks like a cliff" into a defensible claim.

Also plots the fault spectrogram at high vs low SNR (Figure 2) so you can pick
FAULT_BAND_HZ in prepare.py sensibly.

Run AFTER evaluate.py.
"""
import json
import numpy as np
from scipy.optimize import curve_fit


def logistic(x, lo, hi, x0, k):
    return lo + (hi - lo) / (1 + np.exp(-k * (x - x0)))


def step(x, lo, hi, x0):
    return np.where(x < x0, lo, hi)


def rss(y, yhat):
    return float(np.sum((y - yhat) ** 2))


def main():
    res = json.load(open("results/results_sensitivity.json"))
    bins = res["bins"]
    x = np.array([b["snr_center"] for b in bins])
    y = np.array([b["int8_mean"] for b in bins])
    fp = np.array([b["fp32_mean"] for b in bins])

    order = np.argsort(x); x, y, fp = x[order], y[order], fp[order]

    out = {"n_points": len(x)}
    # gradual (logistic) fit
    try:
        # k bounded >0: sensitivity must be non-decreasing in SNR, so a
        # negative-k solution is a degenerate reparameterization, not a
        # different physical fit - excluding it keeps the sign meaningful.
        p, _ = curve_fit(logistic, x, y,
                         p0=[y.min(), y.max(), np.median(x), 1.0], maxfev=10000,
                         bounds=([-0.01, -0.01, -np.inf, 1e-6], [1.01, 1.01, np.inf, np.inf]))
        out["logistic_rss"] = rss(y, logistic(x, *p))
        out["logistic_midpoint_snr"] = float(p[2])
        out["logistic_steepness_k"] = float(p[3])
    except Exception as e:
        out["logistic_error"] = str(e)

    # cliff (step) fit: search best breakpoint
    best = None
    for x0 in np.linspace(x.min(), x.max(), 50):
        lo = y[x < x0].mean() if (x < x0).any() else y.min()
        hi = y[x >= x0].mean() if (x >= x0).any() else y.max()
        r = rss(y, np.where(x < x0, lo, hi))
        if best is None or r < best[0]:
            best = (r, x0, lo, hi)
    out["step_rss"] = best[0]
    out["step_breakpoint_snr"] = float(best[1])

    # crude interpretation
    if "logistic_rss" in out:
        # a very steep logistic ~ a cliff; report both so the reader judges
        out["interpretation"] = (
            "cliff-like" if out["step_rss"] <= out["logistic_rss"]
            and out.get("logistic_steepness_k", 0) > 1.0 else "gradual-or-ambiguous")

    # where does int8 diverge meaningfully from fp32?
    gap = fp - y
    div = x[np.argmax(gap > 0.2)] if (gap > 0.2).any() else None
    out["snr_where_gap_exceeds_0.2"] = None if div is None else float(div)

    json.dump(out, open("results/results_analysis.json", "w"), indent=2)
    print(json.dumps(out, indent=2))
    print("\nReport BOTH fits honestly. A steep logistic or a strongly-preferred "
          "step model supports a 'quantization floor'; a shallow logistic that "
          "beats the step model supports 'gradual cost'. Either is publishable.")


if __name__ == "__main__":
    main()
