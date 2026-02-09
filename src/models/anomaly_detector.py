"""Anomaly detector using LSTM Autoencoder.

This module detects *that* something is anomalous (reconstruction-error
based), not *why*.  The attribution (FAULT vs EXPECTED_LOAD) is handled
by :mod:`src.models.caaa_model` or :mod:`src.models.classifier`.
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


class TimeSeriesDataset(Dataset):
    """Sliding-window dataset for time-series sequences."""

    def __init__(self, data: np.ndarray, seq_length: int = 30) -> None:
        self.data = data
        self.seq_length = seq_length

    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_length)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.FloatTensor(self.data[idx: idx + self.seq_length])


class LSTMAutoencoder(nn.Module):
    """LSTM-based Autoencoder for time-series anomaly detection."""

    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 64,
        latent_dim: int = 16,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.encoder_fc = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.output_layer = nn.Linear(hidden_dim, n_features)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.encoder_lstm(x)
        return self.encoder_fc(hidden[-1])

    def decode(self, latent: torch.Tensor, seq_len: int) -> torch.Tensor:
        hidden = self.decoder_fc(latent).unsqueeze(1).repeat(1, seq_len, 1)
        output, _ = self.decoder_lstm(hidden)
        return self.output_layer(output)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(x)
        return self.decode(latent, x.size(1)), latent


class AnomalyDetector:
    """High-level anomaly detector wrapping :class:`LSTMAutoencoder`.

    Trains on *normal* data and flags inputs whose reconstruction error
    exceeds a learned percentile threshold.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        latent_dim: int = 16,
        num_layers: int = 2,
        seq_length: int = 30,
        threshold_percentile: float = 95,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.threshold_percentile = threshold_percentile

        self.model: Optional[LSTMAutoencoder] = None
        self.scaler = StandardScaler()
        self.threshold: Optional[float] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Pre-processing
    # ------------------------------------------------------------------

    def _preprocess(self, metrics: pd.DataFrame, fit_scaler: bool = False) -> np.ndarray:
        numeric = metrics.select_dtypes(include=[np.number])
        data = np.nan_to_num(numeric.values, nan=0.0)
        if fit_scaler:
            data = self.scaler.fit_transform(data)
        else:
            data = self.scaler.transform(data)
        return data

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        train_metrics: List[pd.DataFrame],
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        calibration_split: float = 0.2,
    ) -> None:
        """Train on normal-operation metrics.

        Args:
            train_metrics: List of normal metrics DataFrames.
            epochs: Training epochs.
            batch_size: Mini-batch size.
            learning_rate: Adam learning rate.
            calibration_split: Fraction of data held out for threshold
                calibration (avoids data leakage).
        """
        # Fit scaler on concatenated data for consistent normalisation
        raw_parts = []
        for m in train_metrics:
            numeric = m.select_dtypes(include=[np.number])
            raw_parts.append(np.nan_to_num(numeric.values, nan=0.0))
        combined_raw = np.vstack(raw_parts)
        self.scaler.fit(combined_raw)

        combined = self.scaler.transform(combined_raw)
        n_features = combined.shape[1]

        # Split into train and calibration sets
        n_cal = max(1, int(len(combined) * calibration_split))
        n_train = len(combined) - n_cal
        indices = np.random.permutation(len(combined))
        train_data = combined[indices[:n_train]]
        cal_data = combined[indices[n_train:]]

        self.model = LSTMAutoencoder(
            n_features=n_features,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            num_layers=self.num_layers,
        ).to(self.device)

        dataset = TimeSeriesDataset(train_data, self.seq_length)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch in loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                reconstruction, _ = self.model(batch)
                loss = criterion(reconstruction, batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / max(len(loader), 1):.6f}")

        # Compute threshold on HELD-OUT calibration data (not training data)
        self._compute_threshold(cal_data)

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def _compute_threshold(self, data: np.ndarray) -> None:
        errors = self.compute_reconstruction_errors(data)
        self.threshold = float(np.percentile(errors, self.threshold_percentile))
        print(f"Anomaly threshold set to: {self.threshold:.6f}")

    def compute_reconstruction_errors(self, data: np.ndarray) -> np.ndarray:
        """Compute per-window reconstruction error."""
        assert self.model is not None
        self.model.eval()
        errors: List[float] = []
        with torch.no_grad():
            for i in range(len(data) - self.seq_length):
                seq = torch.FloatTensor(data[i: i + self.seq_length]).unsqueeze(0).to(self.device)
                reconstruction, _ = self.model(seq)
                errors.append(torch.mean((reconstruction - seq) ** 2).item())
        return np.array(errors)

    def detect(self, metrics: pd.DataFrame) -> Tuple[np.ndarray, float]:
        """Return anomaly scores and max score for *metrics*."""
        data = self._preprocess(metrics)
        errors = self.compute_reconstruction_errors(data)
        scores = errors / (self.threshold + 1e-10)
        return scores, float(np.max(scores)) if len(scores) > 0 else 0.0

    def is_anomalous(self, metrics: pd.DataFrame) -> bool:
        _, max_score = self.detect(metrics)
        return max_score > 1.0

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        assert self.model is not None
        torch.save({
            "model_state": self.model.state_dict(),
            "scaler": self.scaler,
            "threshold": self.threshold,
            "config": {
                "hidden_dim": self.hidden_dim,
                "latent_dim": self.latent_dim,
                "num_layers": self.num_layers,
                "seq_length": self.seq_length,
            },
        }, path)

    def load(self, path: str, n_features: int) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        cfg = checkpoint["config"]
        self.model = LSTMAutoencoder(
            n_features=n_features,
            hidden_dim=cfg["hidden_dim"],
            latent_dim=cfg["latent_dim"],
            num_layers=cfg["num_layers"],
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.scaler = checkpoint["scaler"]
        self.threshold = checkpoint["threshold"]
