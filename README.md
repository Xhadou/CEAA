# CAAA: Context-Aware Anomaly Attribution

> A Novel Framework for False Positive Reduction in Cloud Microservice Anomaly Detection

## Overview

CAAA (Context-Aware Anomaly Attribution) is a research framework that classifies microservice anomalies as **FAULT**, **EXPECTED_LOAD**, or **UNKNOWN**. It addresses the critical problem of false positive alerts in cloud monitoring by integrating contextual information (scheduled events, deployments, time-of-day patterns) into the anomaly classification pipeline. The key innovation is a **Context Consistency Loss** that penalizes predictions contradicting available context signals, achieving significant false positive reduction while maintaining high fault recall.

## Architecture

The CAAA pipeline has an optional two-stage design:

1. **Stage 1 (Optional): Anomaly Detection** — An LSTM autoencoder trained on normal
   (expected-load) metrics identifies which time windows are anomalous via reconstruction
   error. Only anomalous windows proceed to Stage 2.
2. **Stage 2: Anomaly Attribution** — The CAAA model classifies detected anomalies as
   FAULT, EXPECTED_LOAD, or UNKNOWN using:
   - **Temporal Encoder**: MLP-based encoder → 64-dim representation
   - **Context Integration Module**: Attention + confidence gating over 5 context features
   - **Classification Head**: 2-class logits + post-hoc UNKNOWN via confidence threshold

The **Context Consistency Loss** combines:
- Standard cross-entropy classification loss
- Context consistency penalty (contradicting context signals)
- Confidence calibration loss (entropy regularization guided by context confidence)

## Key Results

| Metric | Target | Description |
|--------|--------|-------------|
| FP Reduction | >40% | Reduction in false positive rate vs naive baseline |
| Fault Recall | >90% | Proportion of actual faults correctly identified |
| Accuracy | >80% | Overall classification accuracy |

*Run `python scripts/ablation.py` to generate full results.*

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:** Python 3.9+, PyTorch 2.0+, scikit-learn 1.3+, NumPy, Pandas, SciPy, Matplotlib, Seaborn, PyYAML.

## Data

CAAA supports two data modes:

1. **Synthetic data** (default): Generated on-the-fly via `generate_combined_dataset()` or `generate_research_dataset()`. No downloads required.
2. **RCAEval benchmark data**: Real-world microservice failure traces from Zenodo. Download with `download_rcaeval_dataset()` and load with `RCAEvalLoader`.

## Quick Start

```bash
# Demo (small dataset, fast)
python scripts/demo.py --n-fault 20 --n-load 20 --epochs 30

# Full training with baseline comparison
python scripts/train.py --n-fault 100 --n-load 100 --epochs 50 --baseline

# Training with config file
python scripts/train.py --config configs/config.yaml --baseline

# Ablation study (systematic evaluation of model variants)
python scripts/ablation.py --n-fault 50 --n-load 50 --epochs 30 --n-runs 5

# Full pipeline (supports both CAAA and sklearn models)
python -m src.main --n-fault 50 --n-load 50 --model caaa
python -m src.main --n-fault 50 --n-load 50 --model random_forest

# Download RCAEval dataset (requires network)
python -c "from src.data_loader.download_data import download_rcaeval_dataset; download_rcaeval_dataset('RE1', ['online-boutique'])"

# Run tests
python -m pytest tests/ -v
```

### Using RCAEval Real-World Data

```bash
# Download RCAEval dataset (one-time, requires network)
python -m src.main --download-data --dataset RE1 --system online-boutique

# Train with real fault data + synthetic expected-load cases
python -m src.main --data rcaeval --dataset RE1 --system online-boutique --model caaa

# Full pipeline with anomaly detection pre-stage
python -m src.main --data rcaeval --dataset RE1 --system online-boutique \
    --model caaa --anomaly-detector --ad-epochs 50

# Ablation on real data
python scripts/ablation.py --data rcaeval --dataset RE1 --system online-boutique \
    --epochs 50 --n-runs 10
```

## Project Structure

```
├── configs/config.yaml               # Training and model configuration
├── src/
│   ├── main.py                       # Unified pipeline entry point
│   ├── data_loader/
│   │   ├── data_types.py             # ServiceMetrics, AnomalyCase dataclasses
│   │   ├── synthetic_generator.py    # Normal & load-spike metric generation
│   │   ├── fault_generator.py        # Fault injection (11 types)
│   │   ├── dataset.py               # Combined & research dataset generation
│   │   ├── download_data.py         # RCAEval dataset downloader
│   │   └── rcaeval_loader.py        # RCAEval dataset parser
│   ├── features/
│   │   └── extractors.py            # 36-dimensional feature extraction
│   ├── models/
│   │   ├── temporal_encoder.py      # MLP-based temporal encoder
│   │   ├── context_module.py        # Context integration with attention & gating
│   │   ├── caaa_model.py            # Full CAAA model (novel)
│   │   ├── anomaly_detector.py     # LSTM autoencoder anomaly detector
│   │   ├── classifier.py            # Multi-backend sklearn classifier
│   │   └── baseline.py             # RandomForest & Naive baselines
│   ├── training/
│   │   ├── losses.py               # Context Consistency Loss (novel)
│   │   └── trainer.py              # PyTorch training harness
│   └── evaluation/
│       ├── metrics.py              # Evaluation metrics & FP reduction
│       └── visualization.py        # Plotting utilities
├── scripts/
│   ├── demo.py                     # Quick demonstration
│   ├── train.py                    # Full training pipeline
│   └── ablation.py                 # Ablation study framework
├── tests/
│   ├── test_data_loader.py         # Data generation tests
│   ├── test_features.py            # Feature extraction tests
│   ├── test_models.py              # Model component tests
│   ├── test_integration.py         # End-to-end pipeline tests
│   └── test_plan_modules.py        # Sklearn classifier tests
├── requirements.txt
├── .gitignore
└── README.md
```

## Features

The 36-dimensional feature vector is organized into 5 groups:

| Group | Count | Description |
|-------|-------|-------------|
| Workload (0-5) | 6 | Global load ratio, CPU-request correlation, cross-service sync, error rate delta, latency-CPU correlation, memory trend uniformity |
| Behavioral (6-11) | 6 | Onset gradient, peak duration, cascade score, recovery indicator, affected service ratio, variance change ratio |
| Context (12-16) | 5 | Event active, event expected impact, time seasonality, recent deployment, context confidence |
| Statistical (17-29) | 13 | Mean and std of 6 metrics + max error rate |
| Service-Level (30-35) | 6 | Service count, max CPU/error ratios, CPU/error/latency spread |

## Citation

```bibtex
@article{caaa2025,
  title={CAAA: Context-Aware Anomaly Attribution for False Positive Reduction in Cloud Microservice Monitoring},
  author={},
  year={2025},
  institution={Shiv Nadar University}
}
```

## Authors

- Shiv Nadar University
