"""
prepare.py - segment recordings into clips, compute log-mel features, AND measure
the signal-to-noise ratio of each clip. SNR is the experiment's real x-axis.

Expected raw layout (you record this):
  data/raw/<position>/normal/*.wav       e.g. data/raw/100cm/normal/...
  data/raw/<position>/fault/*.wav
  data/raw/ambient.wav                    (room noise, fan off) for SNR reference

Multiple recording SESSIONS (different day/room/mic-gain/setup) are supported
via a "_s<N>" suffix on the position folder, e.g. data/raw/51cm_s2/. Each
non-default session needs its OWN ambient reference at data/raw/ambient_s<N>.*
- sessions are never pooled onto the same ambient denominator, since gain/room
differences would bias the SNR computation. The default (un-suffixed) session
uses data/raw/ambient.*. Position folders across sessions are kept distinct
even at coincidentally-matching nominal distances, so trials from different
sessions are never silently treated as repeats of the same measurement.

Outputs:
  data/processed/clips.npz  -> X (features), y (0=normal,1=fault),
                               snr_db (per clip), position (per clip), trial
  data/processed/meta.json
"""
import os, glob, json, re
import numpy as np
import librosa

# Only folders matching "<N>cm" or "<N>cm_s<M>" are real position data.
# Allow-listing by pattern (rather than blacklisting known-bad folders like
# "pilot") means a future stray/ad-hoc recording folder fails loud instead of
# silently entering the training set.
POSITION_RE = re.compile(r"^\d+cm(?:_s(\d+))?$")

# Session 2 (newtestrun-0721) used a different fan AND a relocated mic/ambient
# reference relative to session 1 - two confounds the core hypothesis test
# can't tolerate (see FINDINGS.md). Excluded from the core pipeline until a
# same-fan, correctly-isolated (mic/ambient stay put, only the source moves)
# low-SNR session exists. Files remain on disk for a future cross-fan study.
EXCLUDED_SESSIONS = {"2"}

SR = 16000
CLIP_SEC = 1.0
HOP_SEC = 0.5
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256
# Fault band: the clicking transient is broadband but energy-rich in highs.
# Tune this to YOUR fault after inspecting a spectrogram (see analysis.py).
FAULT_BAND_HZ = (2000, 7000)
SEED = 42


def load(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y


def band_energy(seg, lo, hi):
    S = np.abs(librosa.stft(seg, n_fft=N_FFT, hop_length=HOP_LENGTH)) ** 2
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    mask = (freqs >= lo) & (freqs <= hi)
    return float(S[mask].mean() + 1e-12)


def clip_snr_db(seg, ambient_band_energy):
    """SNR of this clip's fault-band energy over the ambient fault-band energy."""
    e = band_energy(seg, *FAULT_BAND_HZ)
    return 10.0 * np.log10(e / ambient_band_energy)


def logmel(seg):
    m = librosa.feature.melspectrogram(y=seg, sr=SR, n_fft=N_FFT,
                                        hop_length=HOP_LENGTH, n_mels=N_MELS)
    logm = librosa.power_to_db(m, ref=np.max)
    return ((logm - logm.min()) / (logm.max() - logm.min() + 1e-8)).astype("float32")


def clips_of(y):
    clip_len = int(CLIP_SEC * SR); hop = int(HOP_SEC * SR)
    return [y[s:s+clip_len] for s in range(0, max(1, len(y)-clip_len+1), hop)
            if len(y[s:s+clip_len]) == clip_len]


def ambient_band_energy(session):
    suffix = "" if session == "1" else f"_s{session}"
    candidates = (glob.glob(f"data/raw/ambient{suffix}.wav") +
                  glob.glob(f"data/raw/ambient{suffix}.m4a"))
    if not candidates:
        raise FileNotFoundError(
            f"No ambient{suffix}.wav or ambient{suffix}.m4a found in data/raw/ "
            f"(required as the SNR reference for session {session})")
    return band_energy(load(candidates[0]), *FAULT_BAND_HZ)


def main():
    os.makedirs("data/processed", exist_ok=True)

    X, y, snr, pos, trial = [], [], [], [], []
    skipped = []
    amb_e_by_session = {}
    for pdir in sorted(glob.glob("data/raw/*/")):
        position = os.path.basename(pdir.rstrip("/"))
        m = POSITION_RE.match(position)
        if not m:
            skipped.append(position)
            continue
        session = m.group(1) or "1"
        if session in EXCLUDED_SESSIONS:
            skipped.append(position)
            continue
        if session not in amb_e_by_session:
            amb_e_by_session[session] = ambient_band_energy(session)
        amb_e = amb_e_by_session[session]

        for label_name, label in [("normal", 0), ("fault", 1)]:
            audio_files = sorted(glob.glob(os.path.join(pdir, label_name, "*.wav")) +
                                 glob.glob(os.path.join(pdir, label_name, "*.m4a")))
            for wav in audio_files:
                tm = re.search(r"trial(\d+)", wav)
                if tm is None:
                    raise ValueError(f"{wav}: expected 'trial<N>' in filename "
                                      f"(protocol requires >=3 independent trials per position)")
                t = int(tm.group(1))
                for seg in clips_of(load(wav)):
                    X.append(logmel(seg)); y.append(label)
                    snr.append(clip_snr_db(seg, amb_e))
                    pos.append(position); trial.append(t)

    X = np.stack(X)[..., None]
    np.savez_compressed("data/processed/clips.npz",
                        X=X, y=np.array(y), snr_db=np.array(snr),
                        position=np.array(pos), trial=np.array(trial))
    meta = dict(SR=SR, CLIP_SEC=CLIP_SEC, N_MELS=N_MELS, N_FFT=N_FFT,
                HOP_LENGTH=HOP_LENGTH, FAULT_BAND_HZ=FAULT_BAND_HZ,
                shape=list(X.shape[1:]), n_clips=int(len(y)))
    json.dump(meta, open("data/processed/meta.json", "w"), indent=2)
    print(f"clips={len(y)}  fault={int(sum(y))}  positions={sorted(set(pos))}")
    print(f"sessions with ambient reference: {sorted(amb_e_by_session)}")
    if skipped:
        print(f"skipped non-position folders (not matching '<N>cm[_s<M>]'): {sorted(skipped)}")
    print("wrote data/processed/clips.npz and meta.json")


if __name__ == "__main__":
    main()
