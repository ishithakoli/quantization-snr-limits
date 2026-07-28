"""
generate_synthetic.py - create synthetic audio data so the full pipeline
(prepare → train → quantize → evaluate → analysis) can run end-to-end
without real recordings.

Synthetic model:
  - Normal: band-limited Gaussian noise (fan hum)
  - Fault:  same noise + periodic transient (brief high-freq burst at blade rate)
            amplitude scales with SNR tier so the experiment's x-axis (measured SNR)
            varies naturally across positions.

Positions: 10cm, 25cm, 50cm, 100cm, 200cm, 400cm
  - Closer positions → louder transient → higher SNR (matches the physical rig)
  - 3 independent trials per position (for error bars in evaluate.py)

Run once before prepare.py:
    python generate_synthetic.py
"""

import os
import numpy as np
import scipy.io.wavfile as wav

SR = 16000
DURATION_SEC = 90       # seconds per recording (enough clips for stats)
TRIALS = 3
AMBIENT_SEC = 30
BLADE_HZ = 20.0         # clicks per second — more repetitions per clip = clearer feature
TRANSIENT_DUR = 0.025   # 25 ms burst — fills enough mel frames to survive normalization

# Relative fault transient amplitudes by position (closer = louder)
POSITIONS = {
    "10cm":  2.00,
    "25cm":  1.10,
    "50cm":  0.55,
    "100cm": 0.22,
    "200cm": 0.08,
    "400cm": 0.025,
}

RNG = np.random.RandomState(42)


def fan_hum(n_samples, amp=0.03):
    """Band-limited noise simulating fan background (energy mostly < 1 kHz)."""
    from scipy.signal import butter, sosfilt
    noise = RNG.randn(n_samples).astype("float32") * amp
    sos = butter(4, 800 / (SR / 2), btype="low", output="sos")
    return sosfilt(sos, noise).astype("float32")


def fault_transient(n_samples, transient_amp):
    """Periodic 4–6 kHz burst at BLADE_HZ — sits cleanly in FAULT_BAND_HZ."""
    sig = np.zeros(n_samples, dtype="float32")
    period = int(SR / BLADE_HZ)
    burst_len = int(TRANSIENT_DUR * SR)
    t_burst = np.arange(burst_len) / SR
    # multi-tone burst: 3 kHz + 5 kHz, fast decay — prominent on log-mel
    burst = (np.sin(2 * np.pi * 3000 * t_burst) +
             np.sin(2 * np.pi * 5000 * t_burst)) * np.exp(-t_burst * 80)
    burst = burst.astype("float32")
    for start in range(0, n_samples - burst_len, period):
        sig[start:start + burst_len] += burst * transient_amp
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

    # Ambient recording (fan off — pure room noise)
    ambient = RNG.randn(int(AMBIENT_SEC * SR)).astype("float32") * 0.008
    write(f"{base}/ambient.wav", ambient)
    print(f"wrote {base}/ambient.wav")

    for pos, amp in POSITIONS.items():
        for trial in range(1, TRIALS + 1):
            hum = fan_hum(n_samples)

            # Normal: fan hum only
            normal_sig = hum + RNG.randn(n_samples).astype("float32") * 0.003
            path = f"{base}/{pos}/normal/trial{trial}.wav"
            write(path, normal_sig)

            # Fault: fan hum + periodic click transient
            fault_sig = hum + fault_transient(n_samples, amp) \
                       + RNG.randn(n_samples).astype("float32") * 0.003
            path = f"{base}/{pos}/fault/trial{trial}.wav"
            write(path, fault_sig)

        print(f"  {pos}: transient_amp={amp:.3f}  (3 trials normal + fault)")

    print(f"\nDone. Directory structure:")
    for pos in POSITIONS:
        print(f"  data/raw/{pos}/normal/trial{{1,2,3}}.wav")
        print(f"  data/raw/{pos}/fault/trial{{1,2,3}}.wav")
    print("  data/raw/ambient.wav")
    print("\nNext: python prepare.py")


if __name__ == "__main__":
    main()
