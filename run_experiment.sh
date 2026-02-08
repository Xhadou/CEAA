#!/bin/bash
# Quick start script for CAAA experiment
set -e

echo "CAAA: Context-Aware Anomaly Attribution"
echo "========================================="

python3 --version

# Install dependencies
echo "Installing dependencies..."
pip install -q -r requirements.txt

# Run the main pipeline
echo ""
echo "Running CAAA pipeline..."
python -m src.main \
    --n-fault 50 \
    --n-load 50 \
    --epochs 50 \
    --model caaa \
    --output outputs/results

echo ""
echo "Experiment complete! Check outputs/results/ for results."
