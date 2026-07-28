"""
make_figures.py - generate the paper's figures from existing results files.
Run after evaluate.py/analysis.py so it picks up the latest data.

Fig 1  - high-SNR control/validity check (fp32 vs int8 near-perfect detection)
Fig 2  - spectrogram, high-SNR vs low-SNR fault clip (justifies FAULT_BAND_HZ)
Fig 3  - core result: sensitivity vs measured SNR, fp32 vs int8, with error bars
Fig 4  - gradual (logistic) vs threshold (step) fit over the sensitivity curve
Fig 5  - measured SNR vs. nominal distance, session 1 vs session 2: shows
         session 2's relocated mic/ambient raised SNR instead of lowering it
         (values are session snapshots - see FINDINGS.md "Session 2 data
         added" and the module docstring of cross_fan_eval.py for how the
         session 2 numbers were computed; recompute via
         cross_fan_eval.load_session2() if the raw data changes)

Color: blue #2a78d6 (fp32) / orange #eb6834 (int8) - validated colorblind-safe
categorical pair (dataviz skill). Distinct markers/linestyles too, so the
figures still read correctly in grayscale print.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FP32_COLOR = "#2a78d6"
INT8_COLOR = "#eb6834"
GRID_COLOR = "#d9d9d6"
OUTDIR = "figures"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#111111",
    "axes.grid": True,
    "grid.color": GRID_COLOR,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})


def fig1_control():
    res = json.load(open("results/results_sensitivity.json"))
    bins = sorted(res["bins"], key=lambda b: b["snr_center"])
    high = [b for b in bins if b["snr_center"] >= 20]
    labels = [f"{b['snr_center']:.0f} dB" for b in high]
    fp32 = [b["fp32_mean"] for b in high]
    int8 = [b["int8_mean"] for b in high]

    x = np.arange(len(labels))
    w = 0.32
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.bar(x - w/2, fp32, w, label="fp32", color=FP32_COLOR)
    ax.bar(x + w/2, int8, w, label="int8", color=INT8_COLOR, hatch="//", edgecolor="white")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Sensitivity (recall on fault clips)")
    ax.set_xlabel("Measured SNR bin")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_title("Fig. 1 — High-SNR control: both precisions detect\nthe fault near-perfectly")
    for xi, (a, b) in enumerate(zip(fp32, int8)):
        ax.text(xi - w/2, a + 0.03, f"{a:.2f}", ha="center", fontsize=8, color="#111")
        ax.text(xi + w/2, b + 0.03, f"{b:.2f}", ha="center", fontsize=8, color="#111")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x")
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/fig1_control.png")
    plt.close(fig)


def fig2_spectrograms():
    import librosa
    d = np.load("data/processed/clips.npz", allow_pickle=True)
    snr, y, X = d["snr_db"], d["y"], d["X"]
    fault = y == 1
    hi_idx = np.argmax(np.where(fault, snr, -np.inf))
    lo_idx = np.argmin(np.where(fault, snr, np.inf))

    meta = json.load(open("data/processed/meta.json"))
    fault_lo, fault_hi = meta["FAULT_BAND_HZ"]

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), sharey=True)
    for ax, idx, title in [(axes[0], hi_idx, f"High SNR ({snr[hi_idx]:.1f} dB)"),
                            (axes[1], lo_idx, f"Low SNR ({snr[lo_idx]:.1f} dB)")]:
        spec = X[idx, ..., 0]
        im = ax.imshow(spec, origin="lower", aspect="auto", cmap="Greys",
                        extent=[0, meta["CLIP_SEC"], 0, meta["N_MELS"]])
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Time (s)")
        n_mels = meta["N_MELS"]
        lo_mel = n_mels * fault_lo / 8000
        hi_mel = n_mels * fault_hi / 8000
        ax.axhspan(lo_mel, hi_mel, color=INT8_COLOR, alpha=0.15, edgecolor=INT8_COLOR, linewidth=1)
    axes[0].set_ylabel("Mel bin")
    fig.suptitle(f"Fig. 2 — Fault-clip log-mel spectrogram, high vs. low SNR\n"
                 f"(shaded band = FAULT_BAND_HZ, {fault_lo}–{fault_hi} Hz)", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/fig2_spectrograms.png")
    plt.close(fig)


def fig3_core_result():
    res = json.load(open("results/results_sensitivity.json"))
    bins = sorted(res["bins"], key=lambda b: b["snr_center"])
    x = np.array([b["snr_center"] for b in bins])
    fp32 = np.array([b["fp32_mean"] for b in bins])
    int8 = np.array([b["int8_mean"] for b in bins])
    fp32_sd = np.array([b["fp32_sd"] for b in bins])
    int8_sd = np.array([b["int8_sd"] for b in bins])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(x, fp32, yerr=fp32_sd, color=FP32_COLOR, marker="o", markersize=6,
                linewidth=2, capsize=3, label="fp32")
    ax.errorbar(x, int8, yerr=int8_sd, color=INT8_COLOR, marker="s", markersize=6,
                linewidth=2, linestyle="--", capsize=3, label="int8")
    ax.set_xlabel("Measured SNR (dB)")
    ax.set_ylabel("Sensitivity (recall on fault clips)")
    ax.set_ylim(-0.05, 1.15)
    ax.set_title("Fig. 3 — Detection sensitivity vs. measured SNR,\nfp32 vs. int8 (error bars: SD across trials)")
    ax.legend(frameon=False, loc="lower right")
    ax.axhline(0, color="#999", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/fig3_core_result.png")
    plt.close(fig)


def fig4_gradual_vs_cliff():
    res = json.load(open("results/results_sensitivity.json"))
    fit = json.load(open("results/results_analysis.json"))
    bins = sorted(res["bins"], key=lambda b: b["snr_center"])
    x = np.array([b["snr_center"] for b in bins])
    y = np.array([b["int8_mean"] for b in bins])

    def logistic(xx, lo, hi, x0, k):
        return lo + (hi - lo) / (1 + np.exp(-k * (xx - x0)))

    xx = np.linspace(x.min() - 2, x.max() + 2, 400)
    x0 = fit["step_breakpoint_snr"]
    lo_step = y[x < x0].mean() if (x < x0).any() else y.min()
    hi_step = y[x >= x0].mean() if (x >= x0).any() else y.max()
    step_y = np.where(xx < x0, lo_step, hi_step)
    logi_y = logistic(xx, y.min(), y.max(), fit["logistic_midpoint_snr"], fit["logistic_steepness_k"])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xx, step_y, color="#999999", linewidth=2, linestyle=":", label=f"step fit (RSS={fit['step_rss']:.4f})")
    ax.plot(xx, logi_y, color=INT8_COLOR, linewidth=2, label=f"logistic fit (RSS={fit['logistic_rss']:.4f})")
    ax.scatter(x, y, color="#111111", zorder=5, s=36, label="int8 (observed)")
    ax.set_xlabel("Measured SNR (dB)")
    ax.set_ylabel("Sensitivity")
    ax.set_ylim(-0.05, 1.15)
    ax.set_title(f"Fig. 4 — Gradual vs. threshold fit to the sensitivity curve\n"
                 f"({fit['interpretation']}, midpoint ≈ {fit['logistic_midpoint_snr']:.1f} dB)")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/fig4_gradual_vs_cliff.png")
    plt.close(fig)


def fig5_snr_vs_distance():
    # Session 1: open-air distance, mic/ambient co-located with the source.
    # Session 2: mic+ambient relocated to a quieter room along with the fan
    # (a different, quieter fan too - see FINDINGS.md). Values are fault-clip
    # SNR means per position, from data/processed/clips.npz (session 1) and
    # cross_fan_eval.load_session2() (session 2) - both session snapshots.
    s1_cm =  [25, 64, 127, 254, 508, 1016]
    s1_snr = [35.78, 33.64, 29.59, 21.64, 12.88, 3.05]
    s2_cm =  [10, 25, 51, 102, 203, 254, 305, 381, 508]
    s2_snr = [46.46, 47.20, 43.63, 34.36, 31.48, 31.69, 35.97, 42.01, 43.12]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(s1_cm, s1_snr, color=FP32_COLOR, marker="o", markersize=6,
            linewidth=2, label="Session 1 (open air, mic/ambient with source)")
    ax.plot(s2_cm, s2_snr, color=INT8_COLOR, marker="s", markersize=6,
            linewidth=2, linestyle="--",
            label="Session 2 (mic/ambient relocated with source, quieter fan)")
    ax.set_xscale("log")
    ax.set_xlabel("Nominal distance from mic to source (cm, log scale)")
    ax.set_ylabel("Measured fault-clip SNR (dB)")
    ax.set_title("Fig. 5 — Measured SNR vs. distance, Session 1 vs. Session 2\n"
                 "(relocating mic+ambient raised SNR instead of lowering it)")
    ax.legend(frameon=False, loc="lower left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/fig5_snr_vs_distance.png")
    plt.close(fig)


if __name__ == "__main__":
    import os
    os.makedirs(OUTDIR, exist_ok=True)
    fig1_control()
    fig2_spectrograms()
    fig3_core_result()
    fig4_gradual_vs_cliff()
    fig5_snr_vs_distance()
    print(f"wrote figures to {OUTDIR}/")
