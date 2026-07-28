"""
inspect_pilot.py - run this on your FIRST recordings before the full session.

Minimal pilot layout (6 files total):
  data/raw/ambient.wav                    (fan OFF, ~30 s)
  data/raw/10cm/normal/trial1.wav
  data/raw/10cm/fault/trial1.wav
  data/raw/400cm/normal/trial1.wav
  data/raw/400cm/fault/trial1.wav
  (optionally a mid position, e.g. 100cm, for a richer SNR sanity check)

Run:
  python inspect_pilot.py

Outputs (all in pilot_inspection/):
  spec_fault_<pos>.png       spectrogram of a fault clip with band markers
  spec_normal_<pos>.png      spectrogram of a normal clip
  fault_band_suggestion.png  fault/normal excess-energy ratio -> suggested band
  snr_vs_distance.png        per-position SNR; must drop clearly with distance
  agc_check.png              background RMS per position; must NOT rise

Terminal: four numbered checks, each PASS / WARN / FAIL, then GO / NO-GO.

Requires: pip install matplotlib  (everything else already in requirements.txt)
"""

import os
import glob
import json
import numpy as np
import librosa
import librosa.display          # explicit: not always loaded by bare `import librosa`
import matplotlib
matplotlib.use("Agg")           # no display needed — saves PNGs
import matplotlib.pyplot as plt

SR = 16000
N_FFT = 1024
HOP_LENGTH = 256
OUTDIR = "pilot_inspection"

# Starting guess; Check 1 will suggest an update from the actual recording.
FAULT_BAND_GUESS = (2000, 7000)

# Go / no-go thresholds
SNR_RANGE_MIN_DB     = 10.0   # near-minus-far must exceed this
AGC_RISE_LIMIT_DB    =  1.5   # background RMS may rise this much before warning
CLIP_MAX_PCT         =  0.1   # > 0.1 % clipped samples -> reduce gain
CLICKS_PER_MIN_MIN   = 100    # fewer -> zip-tie may not be contacting reliably


# ─── audio helpers ────────────────────────────────────────────────────────────

def load(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y


def stft_power(y):
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)) ** 2
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    return S, freqs


def band_energy(y, lo, hi):
    S, f = stft_power(y)
    return float(S[(f >= lo) & (f <= hi)].mean() + 1e-12)


def rms_db(vals_rms, ref_rms):
    """Return dB relative to ref_rms; positive = louder than reference."""
    return 20.0 * np.log10(np.maximum(vals_rms, 1e-12) / (ref_rms + 1e-12))


def clipping_pct(y, thr=0.985):
    return float(np.mean(np.abs(y) > thr)) * 100.0


def count_transients(y, lo, hi, thresh_sd=3.0):
    """Rough click count: fault-band frames above median + thresh_sd × MAD."""
    S, f = stft_power(y)
    e = S[(f >= lo) & (f <= hi)].mean(axis=0)
    med = np.median(e)
    mad = np.median(np.abs(e - med)) + 1e-12
    hot = e > (med + thresh_sd * mad)
    # count rising edges (start of each event)
    n_events = int(np.sum(hot[1:] & ~hot[:-1]))
    rate = n_events / max(len(y) / SR / 60.0, 1e-6)
    return n_events, float(rate)


def background_rms(y):
    """Median frame RMS — a stable estimate of the background level."""
    frame_len = N_FFT
    frames = librosa.util.frame(y, frame_length=frame_len, hop_length=HOP_LENGTH)
    return float(np.median(np.sqrt(np.mean(frames ** 2, axis=0))))


# ─── check 1: fault-band suggestion ──────────────────────────────────────────

def check_fault_band(pos, fault_y, normal_y, outdir):
    """
    Compare fault vs normal at the nearest position.
    Find the contiguous frequency band with the strongest fault excess.
    Saves spec_fault_<pos>.png, spec_normal_<pos>.png, fault_band_suggestion.png.
    Returns (lo, hi, ok, message).
    """
    # spectrograms
    for y, label in [(fault_y, "fault"), (normal_y, "normal")]:
        S, _ = stft_power(y)
        Sdb = librosa.power_to_db(S, ref=np.max)
        fig, ax = plt.subplots(figsize=(9, 4))
        img = librosa.display.specshow(Sdb, sr=SR, hop_length=HOP_LENGTH,
                                       x_axis="time", y_axis="hz",
                                       cmap="magma", ax=ax)
        ax.axhline(FAULT_BAND_GUESS[0], color="cyan", ls="--", lw=1,
                   label=f"current band {FAULT_BAND_GUESS}")
        ax.axhline(FAULT_BAND_GUESS[1], color="cyan", ls="--", lw=1)
        ax.set_title(f"{label.upper()} @ {pos}")
        ax.legend(fontsize=8)
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"spec_{label}_{pos}.png"), dpi=110)
        plt.close(fig)

    # fault/normal ratio per frequency bin
    Sf, f = stft_power(fault_y)
    Sn, _ = stft_power(normal_y)
    excess_db = 10.0 * np.log10(Sf.mean(axis=1) / (Sn.mean(axis=1) + 1e-12) + 1e-12)

    hot_bins = np.where((excess_db > 3.0) & (f > 200))[0]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(f, excess_db, color="darkviolet", lw=1.5, label="fault/normal excess (dB)")
    ax.axhline(3.0, color="red", ls="--", lw=1, label="+3 dB threshold")
    ax.axvspan(*FAULT_BAND_GUESS, alpha=0.10, color="orange",
               label=f"current guess {FAULT_BAND_GUESS}")

    if len(hot_bins) == 0:
        ax.set_title(f"Fault/normal excess @ {pos}  — NO clear band found")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "fault_band_suggestion.png"), dpi=110)
        plt.close(fig)
        return (None, None, False,
                "No frequency band shows >3 dB fault excess. "
                "Check zip-tie contact and mic position before proceeding.")

    lo_hz = float(f[hot_bins.min()])
    hi_hz = float(f[hot_bins.max()])
    # round to nearest 500 Hz for a clean FAULT_BAND_HZ value
    lo_round = max(200, int(lo_hz / 500) * 500)
    hi_round = min(int(SR / 2) - 500, int(hi_hz / 500 + 1) * 500)

    ax.axvspan(lo_hz, hi_hz, alpha=0.20, color="limegreen",
               label=f"suggested ({lo_round}, {hi_round})")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Fault / Normal (dB)")
    ax.set_title(f"Fault/normal excess @ {pos}  — suggested band shaded green")
    ax.set_xlim(0, SR / 2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fault_band_suggestion.png"), dpi=110)
    plt.close(fig)

    same = (lo_round == FAULT_BAND_GUESS[0] and hi_round == FAULT_BAND_GUESS[1])
    if same:
        msg = f"Suggested band ({lo_round}–{hi_round} Hz) matches current FAULT_BAND_HZ."
    else:
        msg = (f"Suggested FAULT_BAND_HZ = ({lo_round}, {hi_round}). "
               f"Current = {FAULT_BAND_GUESS}. "
               f"Update prepare.py and verify against fault_band_suggestion.png.")
    return lo_round, hi_round, True, msg


# ─── check 2: SNR vs distance ─────────────────────────────────────────────────

def check_snr_vs_distance(rows, fault_band, outdir):
    """
    rows: list of (position_name, fault_y) sorted nearest→farthest.
    Measures per-position fault-band SNR vs ambient, checks for a clear drop.
    Returns (dynamic_range_db, ok, message).
    """
    if not os.path.exists("data/raw/ambient.wav"):
        return 0.0, False, "data/raw/ambient.wav missing."
    amb_y = load("data/raw/ambient.wav")
    amb_e = band_energy(amb_y, *fault_band)

    snr_records = []
    for pos, fault_y in rows:
        snr = 10.0 * np.log10(band_energy(fault_y, *fault_band) / amb_e)
        snr_records.append((pos, snr))

    positions = [r[0] for r in snr_records]
    snrs      = [r[1] for r in snr_records]
    dyn = snrs[0] - snrs[-1] if len(snrs) >= 2 else 0.0

    # plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(len(snrs)), snrs, "o-", color="crimson", lw=1.8)
    ax.axhline(0, color="gray", ls="--", lw=1, label="0 dB (noise floor)")
    ax.set_xticks(range(len(positions)))
    ax.set_xticklabels(positions)
    ax.set_xlabel("Position (nearest → farthest)")
    ax.set_ylabel(f"Fault-band SNR  [{fault_band[0]}–{fault_band[1]} Hz]  (dB)")
    ax.set_title(f"SNR vs Distance   dynamic range = {dyn:.1f} dB   "
                 f"(need ≥ {SNR_RANGE_MIN_DB} dB)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "snr_vs_distance.png"), dpi=110)
    plt.close(fig)

    # print table
    print(f"\n  {'Position':>8}  {'SNR (dB)':>10}")
    for pos, snr in snr_records:
        print(f"  {pos:>8}  {snr:>10.1f}")
    print(f"  Dynamic range: {dyn:.1f} dB")

    if len(snrs) < 2:
        return dyn, False, "Need at least two positions."
    if dyn < SNR_RANGE_MIN_DB:
        return dyn, False, (
            f"Only {dyn:.1f} dB of dynamic range (need ≥ {SNR_RANGE_MIN_DB} dB). "
            "Increase max distance, lower fixed gain, or use a quieter background.")
    # allow 3 dB slack per step for monotonicity
    if not all(snrs[i] >= snrs[i+1] - 3.0 for i in range(len(snrs)-1)):
        return dyn, False, (
            f"SNR does not fall monotonically across positions (range = {dyn:.1f} dB). "
            "Likely cause: wrong FAULT_BAND_HZ, room reflections, or inconsistent gain.")
    return dyn, True, f"{dyn:.1f} dB dynamic range, monotonically decreasing."


# ─── check 3: AGC / background level ─────────────────────────────────────────

def check_agc(rows, outdir):
    """
    rows: list of (position_name, fault_y OR normal_y) sorted nearest→farthest.
    Measures background RMS per position. A rise with distance flags AGC.
    Returns (ok, message).
    """
    bg_records = []
    for pos, y in rows:
        bg_records.append((pos, background_rms(y)))

    positions = [r[0] for r in bg_records]
    bgs       = np.array([r[1] for r in bg_records])
    bgs_db    = rms_db(bgs, bgs[0])   # relative to nearest position (= 0 dB)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(len(bgs_db)), bgs_db, "s-", color="steelblue", lw=1.8)
    ax.axhline(0, color="gray", ls="--", lw=1)
    ax.axhline(AGC_RISE_LIMIT_DB, color="orange", ls="--", lw=1,
               label=f"tolerance +{AGC_RISE_LIMIT_DB} dB")
    ax.set_xticks(range(len(positions)))
    ax.set_xticklabels(positions)
    ax.set_xlabel("Position (nearest → farthest)")
    ax.set_ylabel("Background RMS (dB re nearest position)")
    ax.set_title("AGC check — background should stay flat or fall with distance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "agc_check.png"), dpi=110)
    plt.close(fig)

    max_rise = float(np.max(bgs_db[1:]))   # skip first (= 0 by definition)
    if max_rise > AGC_RISE_LIMIT_DB:
        return False, (
            f"Background RMS rises by {max_rise:.1f} dB at far positions. "
            "This is the AGC fingerprint: the mic/interface is turning the gain up "
            "as the source gets fainter, actively erasing the effect you're measuring. "
            "Find the real gain-lock setting and re-record the pilot.")
    return True, f"Background RMS within {max_rise:.1f} dB — gain appears locked."


# ─── check 4: clipping + click count ─────────────────────────────────────────

def check_hygiene(fault_rows, fault_band):
    """
    Returns (all_ok, list-of-row-dicts).
    """
    rows_out = []
    all_ok = True
    for pos, y in fault_rows:
        cp = clipping_pct(y)
        n_ev, rate = count_transients(y, *fault_band)
        clip_ok = cp <= CLIP_MAX_PCT
        rate_ok = rate >= CLICKS_PER_MIN_MIN
        if not clip_ok or not rate_ok:
            all_ok = False
        notes = []
        if not clip_ok:
            notes.append(f"CLIPPING {cp:.2f}% — lower gain")
        if not rate_ok:
            notes.append(f"only {rate:.0f} clicks/min — check zip-tie contact")
        rows_out.append(dict(
            position=pos,
            clip_pct=round(cp, 3),
            clicks_per_min=round(rate, 1),
            pass_bool=clip_ok and rate_ok,
            notes="; ".join(notes) if notes else "ok",
        ))
    return all_ok, rows_out


# ─── main ─────────────────────────────────────────────────────────────────────

def sort_key(name):
    import re
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else 0


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    if not os.path.exists("data/raw/ambient.wav"):
        print("ERROR: data/raw/ambient.wav not found. Record fan-OFF ambient first.")
        return

    pos_dirs = sorted(
        [d for d in glob.glob("data/raw/*/") if os.path.isdir(d)],
        key=lambda d: sort_key(os.path.basename(d.rstrip("/")))
    )
    positions = [os.path.basename(d.rstrip("/")) for d in pos_dirs]
    if not positions:
        print("No position folders found under data/raw/. Aborting.")
        return

    print(f"Positions found: {positions}")

    # load one fault file per position (first trial is enough for pilot checks)
    fault_rows = []
    normal_rows = []
    for pos in positions:
        ffiles = sorted(glob.glob(f"data/raw/{pos}/fault/*.wav"))
        nfiles = sorted(glob.glob(f"data/raw/{pos}/normal/*.wav"))
        if ffiles:
            fault_rows.append((pos, load(ffiles[0])))
        if nfiles:
            normal_rows.append((pos, load(nfiles[0])))

    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 64)
    print("PILOT INSPECTION — FOUR CHECKS")
    print("=" * 64)

    results = {}

    # ── Check 1: fault band ────────────────────────────────────────────────
    print("\n[1] FAULT BAND")
    near_pos = fault_rows[0][0] if fault_rows else None
    near_fault_y = fault_rows[0][1] if fault_rows else None
    near_normal_y = next((y for p, y in normal_rows if p == near_pos), None)

    fault_band = FAULT_BAND_GUESS   # fallback if check fails

    if near_fault_y is not None and near_normal_y is not None:
        lo, hi, band_ok, band_msg = check_fault_band(
            near_pos, near_fault_y, near_normal_y, OUTDIR)
        if band_ok:
            fault_band = (lo, hi)
        print(f"  {'PASS' if band_ok else 'WARN'}  {band_msg}")
        print(f"       (always verify fault_band_suggestion.png by eye — "
              f"room resonances can look like excess energy)")
    else:
        band_ok = False
        print(f"  SKIP  Need both fault and normal at {near_pos}.")
    results["check_1_fault_band"] = {
        "status": "pass" if band_ok else "warn",
        "fault_band": list(fault_band),
    }

    # ── Check 2: SNR vs distance ───────────────────────────────────────────
    print("\n[2] SNR vs DISTANCE")
    dyn, snr_ok, snr_msg = check_snr_vs_distance(fault_rows, fault_band, OUTDIR)
    print(f"  {'PASS' if snr_ok else 'FAIL'}  {snr_msg}")
    results["check_2_snr"] = {"status": "pass" if snr_ok else "fail",
                               "dynamic_range_db": round(dyn, 1)}

    # ── Check 3: AGC ──────────────────────────────────────────────────────
    print("\n[3] AGC CHECK")
    # prefer normal clips for background (no click contamination); fall back to fault
    bg_rows = normal_rows if len(normal_rows) >= 2 else fault_rows
    agc_ok, agc_msg = check_agc(bg_rows, OUTDIR)
    print(f"  {'PASS' if agc_ok else 'FAIL'}  {agc_msg}")
    results["check_3_agc"] = {"status": "pass" if agc_ok else "fail"}

    # ── Check 4: clipping + click count ───────────────────────────────────
    print("\n[4] CLIPPING + CLICK RATE")
    hygiene_ok, hygiene_rows = check_hygiene(fault_rows, fault_band)
    for r in hygiene_rows:
        print(f"  {'PASS' if r['pass_bool'] else 'FAIL'}  "
              f"{r['position']:>8}  clip={r['clip_pct']:.3f}%  "
              f"{r['clicks_per_min']:.0f} clicks/min  {r['notes']}")
    results["check_4_hygiene"] = {"status": "pass" if hygiene_ok else "fail",
                                   "rows": hygiene_rows}

    # ══════════════════════════════════════════════════════════════════════════
    # Overall verdict
    hard_pass = snr_ok and agc_ok and hygiene_ok
    print("\n" + "=" * 64)
    if hard_pass:
        print("OVERALL: GO")
        if not band_ok:
            print("  Update FAULT_BAND_HZ in prepare.py before running prepare.py.")
        elif fault_band != FAULT_BAND_GUESS:
            print(f"  Update FAULT_BAND_HZ = {fault_band} in prepare.py first.")
        else:
            print("  FAULT_BAND_HZ looks correct. Proceed to full recording run.")
    else:
        blockers = []
        if not snr_ok:
            blockers.append("SNR dynamic range insufficient or non-monotone")
        if not agc_ok:
            blockers.append("AGC detected — gain is not locked")
        if not hygiene_ok:
            blockers.append("Clipping or insufficient click rate")
        print("OVERALL: NO-GO")
        for b in blockers:
            print(f"  • {b}")
        print("  Fix the rig and re-run inspect_pilot.py before the full run.")
    print("=" * 64)

    results["overall"] = "go" if hard_pass else "no-go"
    out_json = os.path.join(OUTDIR, "pilot_summary.json")
    json.dump(results, open(out_json, "w"), indent=2)
    print(f"\nOutputs in {OUTDIR}/")
    print("  spec_fault_<pos>.png, spec_normal_<pos>.png")
    print("  fault_band_suggestion.png, snr_vs_distance.png, agc_check.png")
    print("  pilot_summary.json")


if __name__ == "__main__":
    main()
