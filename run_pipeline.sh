#!/usr/bin/env bash
# run_pipeline.sh - run the core experiment end-to-end
# Usage:
#   ./run_pipeline.sh           # uses synthetic data (generates it first) - smoke test only
#   ./run_pipeline.sh --real    # skips generate_synthetic.py (use your real recordings)
#
# Real recordings expected layout (see README.md for full details):
#   data/raw/<N>cm[_s3|_s4]/normal/trial<N>.{wav,m4a}
#   data/raw/<N>cm[_s3|_s4]/fault/trial<N>.{wav,m4a}
#   data/raw/ambient.wav, ambient_s3.m4a, ambient_s4.m4a
#
# Does NOT run src/cross_fan_eval.py or src/make_figures.py - run those
# separately after this completes (see README.md's "which script produces
# which result" table).

set -e
VENV=".venv/bin/python"

if [ "$1" != "--real" ]; then
    echo "==> Generating synthetic data (smoke test only, not paper-reproducing)..."
    $VENV scripts/generate_synthetic.py
fi

echo ""
echo "==> Step 1/5: src/prepare.py — segment clips, compute log-mel + SNR"
$VENV src/prepare.py

echo ""
echo "==> Step 2/5: src/train.py — train fp32 CNN on high-SNR clips"
$VENV src/train.py

echo ""
echo "==> Step 3/5: src/quantize.py — int8 TFLite export + size/latency"
$VENV src/quantize.py

echo ""
echo "==> Step 4/5: src/evaluate.py — sensitivity vs SNR, fp32 vs int8"
$VENV src/evaluate.py

echo ""
echo "==> Step 5/5: src/analysis.py — cliff vs gradual fit"
$VENV src/analysis.py

echo ""
echo "=== Pipeline complete ==="
echo "Results (see README.md for how each maps to the paper):"
echo "  results/results_sensitivity.json"
echo "  results/results_analysis.json"
echo "  results/results_edge.json"
echo ""
echo "Run separately: python3 src/cross_fan_eval.py, then python3 src/make_figures.py"
