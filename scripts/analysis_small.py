"""
analysis_small.py - gradual vs cliff fit for the small-CNN follow-up, same
procedure as analysis.py, plus the fp32/int8 gap check.
"""
import json
import numpy as np
from scipy.optimize import curve_fit


def logistic(x, lo, hi, x0, k):
    return lo + (hi - lo) / (1 + np.exp(-k * (x - x0)))


def rss(y, yhat):
    return float(np.sum((y - yhat) ** 2))


def main():
    res = json.load(open("supplementary/exploratory_results/results_sensitivity_small.json"))
    bins = res["bins"]
    x = np.array([b["snr_center"] for b in bins])
    y = np.array([b["int8_mean"] for b in bins])
    fp = np.array([b["fp32_mean"] for b in bins])

    order = np.argsort(x); x, y, fp = x[order], y[order], fp[order]

    out = {"n_points": len(x)}
    try:
        p, _ = curve_fit(logistic, x, y,
                         p0=[y.min(), y.max(), np.median(x), 1.0], maxfev=10000)
        out["logistic_rss"] = rss(y, logistic(x, *p))
        out["logistic_midpoint_snr"] = float(p[2])
        out["logistic_steepness_k"] = float(p[3])
    except Exception as e:
        out["logistic_error"] = str(e)

    best = None
    for x0 in np.linspace(x.min(), x.max(), 50):
        lo = y[x < x0].mean() if (x < x0).any() else y.min()
        hi = y[x >= x0].mean() if (x >= x0).any() else y.max()
        r = rss(y, np.where(x < x0, lo, hi))
        if best is None or r < best[0]:
            best = (r, x0, lo, hi)
    out["step_rss"] = best[0]
    out["step_breakpoint_snr"] = float(best[1])

    if "logistic_rss" in out:
        out["interpretation"] = (
            "cliff-like" if out["step_rss"] <= out["logistic_rss"]
            and out.get("logistic_steepness_k", 0) > 1.0 else "gradual-or-ambiguous")

    gap = fp - y
    div = x[np.argmax(gap > 0.2)] if (gap > 0.2).any() else None
    out["snr_where_gap_exceeds_0.2"] = None if div is None else float(div)
    out["max_gap"] = float(gap.max())
    out["max_gap_snr"] = float(x[np.argmax(gap)])

    json.dump(out, open("supplementary/exploratory_results/results_analysis_small.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
