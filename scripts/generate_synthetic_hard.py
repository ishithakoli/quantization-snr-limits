"""
generate_synthetic_hard.py - positive control: a synthetic dataset where
int8 *should* fail at low SNR, validating that the pipeline can register a
divergence when one exists.

What changed from generate_synthetic.py and why:
  - Background noise is now BROADBAND (flat up to Nyquist) rather than
    low-pass filtered at 800 Hz. This means noise energy fills the fault
    band (2-7 kHz), so a faint click genuinely competes with the noise floor
    instead of sitting in a band that is empty of noise.
  - The fault click is a BAND-LIMITED NOISE BURST (2-7 kHz bandpass filtered
    impulse) rather than clean pure tones. A broadband click with gradually
    falling amplitude eventually sinks into the broadband background; two
    pure tones in a silent band never do, regardless of amplitude.
  - Ambient recording uses the same broadband noise model so that the SNR
    denominator (ambient fault-band energy) is correctly non-trivial.

Expected outcome: at close positions (10/25/50cm) both fp32 and int8 should
detect the fault reliably. At far positions (200/400cm) the click is buried in
broadband noise; fp32 should still find it (it's the analogue) but int8's
coarser representation should start dropping sensitivity first. That divergence
is the positive control: if analysis.py reports snr_where_gap_exceeds_0.2 as
non-null, the pipeline can see a quantization-induced divergence.

Usage:
    python generate_synthetic_hard.py     # writes data/raw/ (replaces easy version)
    python prepare.py
    python src/train.py
    python src/quantize.py
    python evaluate.py
    python analysis.py

Alternatively, run_pipeline.sh will call this if you swap the generator name in
the script.
"""

import os
import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import butter, sosfilt

SR = 16000
DURATION_SEC = 90       # seconds per recording
TRIALS = 3
AMBIENT_SEC = 30
BLADE_HZ = 20.0         # clicks per second
TRANSIENT_DUR = 0.020   # 20 ms click burst

# Fault band used in prepare.py — must match FAULT_BAND_HZ there.
FAULT_BAND_LO = 2000
FAULT_BAND_HI = 7000

# Background noise level: broadband, same order of magnitude as fault band.
# This is the key change: the fault has to compete with real noise in its band.
BACKGROUND_AMP = 0.04   # rms ~ 0.04 * sqrt(1) ≈ 40 mV normalised

# Fault transient amplitudes by position.
# The close positions are loud enough that the click clears the noise floor
# comfortably; the far positions are designed to be near or below it so that
# int8 quantisation costs detection before fp32 does.
# Rough guideline: band-limited noise burst amplitude vs BACKGROUND_AMP * sqrt(BW_ratio)
# where BW_ratio = (7000-2000)/(SR/2) ≈ 0.625, so effective in-band noise rms ~ 0.032.
# Burst peak amplitudes below ~0.05 are genuinely difficult; below ~0.02 is near noise.
POSITIONS = {
    "10cm":  0.50,   # ~24 dB above in-band noise floor → easy for both models
    "25cm":  0.25,   # ~18 dB → still reliable
    "50cm":  0.12,   # ~12 dB → reliable but less margin
    "100cm": 0.055,  # ~5 dB → marginal; int8 may start dropping here
    "200cm": 0.028,  # ~1 dB → near noise; int8 expected to struggle
    "400cm": 0.014,  # ~-3 dB → below noise floor; both should struggle, int8 worse
}

RNG = np.random.RandomState(42)


def broadband_noise(n_samples, amp):
    """Flat-spectrum (white-ish) noise — fills the fault band unlike the easy version."""
    return (RNG.randn(n_samples) * amp).astype("float32")


def fan_hum_component(n_samples, amp=0.015):
    """Low-frequency mechanical rumble component (fan rotation harmonics < 400 Hz).
    Added on top of broadband so the overall shape is realistic: strong lows,
    but with the broadband floor that actually matters for the SNR measurement."""
    sos = butter(4, 400 / (SR / 2), btype="low", output="sos")
    noise = RNG.randn(n_samples).astype("float32") * amp
    return sosfilt(sos, noise).astype("float32")


def fault_click(n_samples, transient_amp):
    """Periodic band-limited noise burst in the fault band.

    A noise burst (bandpass-filtered impulse) rather than pure tones means:
    - The energy is spread across the fault band (more realistic)
    - Amplitude reduction genuinely changes band-limited SNR
    - INT8's coarser per-tensor scale quantises this spread-spectrum click
      worse than fp32, especially when the burst is near the noise floor.
    """
    period = int(SR / BLADE_HZ)
    burst_len = int(TRANSIENT_DUR * SR)

    # Build one click: bandpass-filtered impulse with exponential taper.
    # The bandpass puts energy squarely inside FAULT_BAND_HZ.
    impulse = RNG.randn(burst_len).astype("float64")
    lo = FAULT_BAND_LO / (SR / 2)
    hi = FAULT_BAND_HI / (SR / 2)
    sos = butter(4, [lo, hi], btype="band", output="sos")
    filtered = sosfilt(sos, impulse)
    # Exponential taper so it sounds like a click, not a sustained tone.
    t = np.arange(burst_len) / SR
    taper = np.exp(-t * 60)   # 60/s decay — ~16ms half-energy
    click = (filtered * taper).astype("float32")
    # Normalise click peak to 1.0 so `transient_amp` is the actual peak amplitude.
    peak = np.abs(click).max()
    if peak > 0:
        click /= peak

    sig = np.zeros(n_samples, dtype="float32")
    for start in range(0, n_samples - burst_len, period):
        sig[start:start + burst_len] += click * transient_amp

    return sig


def to_int16(sig):
    sig = np.clip(sig, -1.0, 1.0)
    return (sig * 32767).astype(np.int16)


def write(path, sig):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    wav.write(path, SR, to_int16(sig))


def main():
    base = "data/raw"
    n_samples = int(DURATION_SEC * SR)

    # Ambient: broadband noise with no fault — this sets the SNR denominator.
    # Must be broadband so the ambient fault-band energy is non-trivial and the
    # SNR values at far positions are genuinely low (not just large positive numbers).
    ambient_n = int(AMBIENT_SEC * SR)
    ambient = (broadband_noise(ambient_n, BACKGROUND_AMP) +
               fan_hum_component(ambient_n))
    write(f"{base}/ambient.wav", ambient)
    print(f"wrote {base}/ambient.wav  (broadband; fault-band floor is real now)")

    for pos, amp in POSITIONS.items():
        for trial in range(1, TRIALS + 1):
            # Background: broadband + low-freq rumble (same in both normal and fault)
            bg = (broadband_noise(n_samples, BACKGROUND_AMP) +
                  fan_hum_component(n_samples))

            # Normal: background only — no click
            normal_sig = bg.copy()
            write(f"{base}/{pos}/normal/trial{trial}.wav", normal_sig)

            # Fault: background + periodic click burst
            fault_sig = bg + fault_click(n_samples, amp)
            write(f"{base}/{pos}/fault/trial{trial}.wav", fault_sig)

        # Rough in-band SNR estimate for this position (informational only)
        # In-band noise rms ≈ BACKGROUND_AMP * sqrt(BW/Nyquist)
        bw_ratio = (FAULT_BAND_HI - FAULT_BAND_LO) / (SR / 2)
        noise_rms = BACKGROUND_AMP * np.sqrt(bw_ratio)
        # Click peak power approximation (burst_len * amp^2 / 2, averaged over period)
        burst_len = int(TRANSIENT_DUR * SR)
        period = int(SR / BLADE_HZ)
        click_rms_in_period = amp * np.sqrt(burst_len / (2 * period))
        snr_approx = 20 * np.log10(click_rms_in_period / (noise_rms + 1e-12))
        print(f"  {pos}: amp={amp:.3f}  approx_band_SNR~{snr_approx:.1f}dB")

    print(f"\nDone.  Key difference from generate_synthetic.py:")
    print("  - Background is now broadband (not 800Hz low-pass)")
    print("  - Click is a bandpass noise burst (not pure tones in a silent band)")
    print("  - Far positions have genuinely sub-zero band SNR — int8 should struggle")
    print("\nExpected pipeline outcome:")
    print("  close positions (10–50cm):  both fp32 and int8 sensitivity ~1.0")
    print("  mid positions  (100cm):     small gap may appear")
    print("  far positions  (200–400cm): visible fp32 > int8 divergence")
    print("  analysis.py:                snr_where_gap_exceeds_0.2 should be non-null")
    print("\nIf the gap does NOT appear, there is a measurement bug to find before")
    print("recording real audio. Either way, you learn something essential.")
    print("\nNext: python prepare.py")


if __name__ == "__main__":
    main()
