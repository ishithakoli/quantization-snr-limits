# Quantization vs. Signal Attenuation: When Does INT8 Lose Faint Acoustic Faults?

Tests whether an int8-quantized acoustic fault detector loses detection
sensitivity faster than its fp32 counterpart as the measured signal-to-noise
ratio (SNR) of the fault signature decreases. A small CNN classifies 1-second
audio clips from a mechanical fan as NORMAL or FAULT (a repeatable mechanical
click), trained only on high-SNR clips and evaluated across the full measured
SNR range in both fp32 and int8.

## Installation

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.9+ recommended.

## Data and trained models

Raw audio recordings and trained models are not included in this repository
(large binary files). Download them from:

https://drive.google.com/drive/folders/1-azwaC9SZ_SQqKuJ5b4clr4gEpJMIQmD?usp=sharing

Place the downloaded contents under `data/raw/` and `models/` at the repo
root. Expected layout (`.wav` and `.m4a` both supported):

```
data/raw/<N>cm/normal/trial<N>.wav       # session 1 (open air)
data/raw/<N>cm/fault/trial<N>.wav
data/raw/<N>cm_s3/normal/trial<N>.m4a    # session 3 (barrier-attenuated)
data/raw/<N>cm_s3/fault/trial<N>.m4a
data/raw/<N>cm_s4/normal/trial<N>.m4a    # session 4 (barrier-attenuated)
data/raw/<N>cm_s4/fault/trial<N>.m4a
data/raw/ambient.wav                     # session 1 SNR reference
data/raw/ambient_s3.m4a                  # session 3 SNR reference
data/raw/ambient_s4.m4a                  # session 4 SNR reference
```

No real audio on hand? `python3 scripts/generate_synthetic.py` creates a
synthetic dataset so the pipeline can be smoke-tested end-to-end — useful
for verifying the code runs, but it does not reproduce the paper's actual
numbers.

## Repository structure

```
src/        core pipeline (.py) - reproduces every result reported in the paper
scripts/    shell entry points + follow-up/exploratory analyses (.py) not
            part of the paper's reported results, but informing the
            investigation
results/    JSON outputs corresponding to the paper's reported results
figures/    the paper's figures, as rendered by src/make_figures.py
```

## Running the pipeline

From the repo root, with the virtual environment active:

```
python3 src/prepare.py    # segment clips, compute log-mel features + per-clip SNR
python3 src/train.py      # train fp32 CNN on high-SNR clips
python3 src/quantize.py   # int8 TFLite export + size/latency benchmark
python3 src/evaluate.py   # sensitivity vs. measured SNR, fp32 vs. int8
python3 src/analysis.py   # gradual-vs-cliff fit to the sensitivity curve
python3 src/cross_fan_eval.py  # cross-source generalization probe
python3 src/make_figures.py    # renders all paper figures to figures/
```

Or run the first five as one step: `./scripts/run_pipeline.sh --real` (omit
`--real` to use synthetic data instead).

## Which script produces which paper result

| Script | Output | Paper reference |
|---|---|---|
| `src/evaluate.py` | `results/results_sensitivity.json` | Table S1, Fig. 3 |
| `src/analysis.py` | `results/results_analysis.json` | Fig. 4 |
| `src/quantize.py` | `results/results_edge.json` | Table 1 (size/latency) |
| `src/cross_fan_eval.py` | `results/results_cross_fan.json` | Table S2, §3.5 |
| `src/make_figures.py` | `figures/fig1_control.png` … `fig5_snr_vs_distance.png` | Fig. 1–5 |

`scripts/` contains additional analyses (a model-capacity sweep, an
activation-level quantization-error analysis, and a smaller model variant)
that informed the investigation but are not part of the paper's reported
results: `model_variants.py`, `sweep.py`, `quant_error_analysis.py`,
`train_small.py`, `quantize_small.py`, `evaluate_small.py`,
`analysis_small.py`. Also included: `inspect_pilot.py` (pre-recording rig
validation) and `generate_synthetic.py`/`generate_synthetic_hard.py`
(synthetic-data smoke testing).

## Honest limitations

Single fault type, single primary acoustic source, single microphone, one
quantization scheme (post-training int8). Generalization to other faults,
sources, or model architectures untested. Not a safety-critical system.
