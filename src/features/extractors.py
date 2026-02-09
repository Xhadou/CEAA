"""Feature extraction for anomaly cases.

Extracts a fixed-size feature vector (36 features) from an AnomalyCase,
organized into workload, behavioral, context, statistical, and service-level
feature groups.

The ``onset_gradient`` feature uses PELT change point detection (Killick 2012)
via the ``ruptures`` library, inspired by BARO (FSE 2024) which showed that
Bayesian change point detection before RCA improves results by 58-189%.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from src.data_loader.data_types import AnomalyCase, ServiceMetrics
from src.features.feature_schema import (
    WORKLOAD_NAMES as _WORKLOAD_NAMES,
    BEHAVIORAL_NAMES as _BEHAVIORAL_NAMES,
    CONTEXT_NAMES as _CONTEXT_NAMES,
    STAT_METRIC_COLS as _STAT_METRIC_COLS,
    STATISTICAL_NAMES as _STATISTICAL_NAMES,
    SERVICE_LEVEL_NAMES as _SERVICE_LEVEL_NAMES,
    N_FEATURES,
)

logger = logging.getLogger(__name__)


def _safe_pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation, returning 0.0 for constant arrays."""
    if len(x) < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0
    try:
        r, _ = pearsonr(x, y)
        return float(r) if np.isfinite(r) else 0.0
    except Exception:
        return 0.0


def _linear_slope(arr: np.ndarray) -> float:
    """Compute slope of a linear fit to *arr*."""
    n = len(arr)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    with np.errstate(invalid="ignore"):
        coeffs = np.polyfit(x, arr, 1)
    return float(coeffs[0]) if np.isfinite(coeffs[0]) else 0.0


def _detect_change_point(
    series: np.ndarray, penalty: float = 10,
) -> Tuple[int, float, float]:
    """Detect the most significant change point in a time series.

    Uses the PELT algorithm (Killick 2012) for O(n) change point detection
    with an RBF cost model, inspired by BARO (FSE 2024) which showed that
    change point detection before RCA improves results significantly.

    The RBF (Radial Basis Function) kernel is chosen because it can detect
    changes in both mean and variance simultaneously — important for fault
    signals that may shift the distribution shape, not just the mean.

    Args:
        series: 1-D array of metric values.
        penalty: Penalty value for PELT.  Higher values produce fewer change
            points.

    Returns:
        Tuple of ``(change_idx, magnitude, abruptness)`` where:
            - *change_idx*: index of the most significant change point, or
              ``-1`` if none found.
            - *magnitude*: ``|mean_after - mean_before| / std`` — measures
              the size of the regime change.
            - *abruptness*: ``|gradient[cp]| / std`` — measures how sudden
              the change is.
    """
    import ruptures as rpt

    if len(series) < 10:
        return -1, 0.0, 0.0

    algo = rpt.Pelt(model="rbf").fit(series.reshape(-1, 1))
    change_points = algo.predict(pen=penalty)
    # ruptures includes len(series) as the last element
    change_points = [cp for cp in change_points if cp < len(series)]

    if not change_points:
        return -1, 0.0, 0.0

    # Find the most significant change point.  We skip change points within
    # 3 indices of the edges to ensure stable mean/gradient estimates on
    # both sides of the split.
    best_cp, best_magnitude = -1, 0.0
    for cp in change_points:
        if cp < 3 or cp > len(series) - 3:
            continue
        mean_before = np.mean(series[:cp])
        mean_after = np.mean(series[cp:])
        std = np.std(series) + 1e-10
        magnitude = abs(mean_after - mean_before) / std
        if magnitude > best_magnitude:
            best_cp, best_magnitude = cp, magnitude

    if best_cp == -1:
        return -1, 0.0, 0.0

    # Compute abruptness (gradient at change point)
    grad = np.gradient(series)
    abruptness = abs(grad[best_cp]) / (np.std(series) + 1e-10)

    return best_cp, best_magnitude, abruptness


class FeatureExtractor:
    """Extracts a 36-dimensional feature vector from an ``AnomalyCase``.

    Feature groups:
        - Workload features (6)
        - Behavioral features (6)
        - Context features (5)
        - Statistical features (13)
        - Service-level features (6)
    """

    def __init__(self) -> None:
        self._names: List[str] = (
            _WORKLOAD_NAMES
            + _BEHAVIORAL_NAMES
            + _CONTEXT_NAMES
            + _STATISTICAL_NAMES
            + _SERVICE_LEVEL_NAMES
        )
        assert len(self._names) == N_FEATURES

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, case: AnomalyCase) -> np.ndarray:
        """Extract a feature vector from a single anomaly case.

        Args:
            case: An ``AnomalyCase`` instance.

        Returns:
            A 1-D numpy array of shape ``(36,)``.
        """
        feats = np.concatenate([
            self._workload_features(case.services),
            self._behavioral_features(case.services),
            self._context_features(case.context, case.services, case.label),
            self._statistical_features(case.services),
            self._service_level_features(case.services),
        ])
        assert feats.shape == (N_FEATURES,), f"Expected {N_FEATURES}, got {feats.shape}"
        return feats

    def extract_batch(self, cases: List[AnomalyCase]) -> np.ndarray:
        """Extract features for multiple cases.

        Args:
            cases: List of ``AnomalyCase`` instances.

        Returns:
            A 2-D numpy array of shape ``(len(cases), 36)``.
        """
        return np.vstack([self.extract(c) for c in cases])

    def feature_names(self) -> List[str]:
        """Return ordered list of 36 feature names."""
        return list(self._names)

    # ------------------------------------------------------------------
    # Workload features (6)
    # ------------------------------------------------------------------

    def _workload_features(self, services: List[ServiceMetrics]) -> np.ndarray:
        n = len(services)
        if n == 0:
            return np.zeros(6)

        # 1. global_load_ratio
        increased = 0
        for svc in services:
            cpu = svc.metrics["cpu_usage"].values
            mid = len(cpu) // 2
            if mid == 0:
                continue
            if np.mean(cpu[mid:]) > np.mean(cpu[:mid]) * 1.10:
                increased += 1
        global_load_ratio = increased / n

        # 2. cpu_request_correlation
        corrs = []
        for svc in services:
            cpu = svc.metrics["cpu_usage"].values
            req = svc.metrics["request_rate"].values
            corrs.append(_safe_pearsonr(cpu, req))
        cpu_request_correlation = float(np.mean(corrs))

        # 3. cross_service_sync
        if n < 2:
            cross_service_sync = 0.0
        else:
            cpu_series = [svc.metrics["cpu_usage"].values for svc in services]
            pair_corrs = []
            for i in range(n):
                for j in range(i + 1, n):
                    min_len = min(len(cpu_series[i]), len(cpu_series[j]))
                    pair_corrs.append(
                        _safe_pearsonr(cpu_series[i][:min_len], cpu_series[j][:min_len])
                    )
            cross_service_sync = float(np.mean(pair_corrs)) if pair_corrs else 0.0

        # 4. error_rate_delta
        deltas = []
        for svc in services:
            err = svc.metrics["error_rate"].values
            mid = len(err) // 2
            if mid == 0:
                continue
            deltas.append(float(np.mean(err[mid:]) - np.mean(err[:mid])))
        error_rate_delta = float(np.mean(deltas)) if deltas else 0.0

        # 5. latency_cpu_correlation
        lat_corrs = []
        for svc in services:
            lat = svc.metrics["latency"].values
            cpu = svc.metrics["cpu_usage"].values
            lat_corrs.append(_safe_pearsonr(lat, cpu))
        latency_cpu_correlation = float(np.mean(lat_corrs))

        # 6. change_point_magnitude (replaces memory_trend_uniformity)
        magnitudes = []
        for svc in services:
            cpu = svc.metrics["cpu_usage"].values
            _, mag, _ = _detect_change_point(cpu)
            magnitudes.append(mag)
        change_point_magnitude = float(np.mean(magnitudes))

        return np.array([
            global_load_ratio,
            cpu_request_correlation,
            cross_service_sync,
            error_rate_delta,
            latency_cpu_correlation,
            change_point_magnitude,
        ])

    # ------------------------------------------------------------------
    # Behavioral features (6)
    # ------------------------------------------------------------------

    def _behavioral_features(self, services: List[ServiceMetrics]) -> np.ndarray:
        n = len(services)
        if n == 0:
            return np.zeros(6)

        # 7. onset_gradient (change-point-based abruptness via PELT)
        abruptness_values = []
        for svc in services:
            cpu = svc.metrics["cpu_usage"].values
            _, _, abruptness = _detect_change_point(cpu)
            abruptness_values.append(abruptness)
        onset_gradient = float(np.mean(abruptness_values))

        # 8. peak_duration
        durations = []
        for svc in services:
            cpu = svc.metrics["cpu_usage"].values
            seq_len = len(cpu)
            if seq_len == 0:
                durations.append(0.0)
                continue
            std = np.std(cpu)
            if std == 0.0:
                durations.append(0.0)
                continue
            z = (cpu - np.mean(cpu)) / std
            anom_count = int(np.sum(z > 2.0))
            durations.append(anom_count / seq_len)
        peak_duration = float(np.mean(durations))

        # 9. cascade_score
        onset_times: List[float] = []
        for svc in services:
            cpu = svc.metrics["cpu_usage"].values
            std = np.std(cpu)
            if std == 0.0 or len(cpu) < 2:
                continue
            z = (cpu - np.mean(cpu)) / std
            anom_idx = np.where(z > 2.0)[0]
            if len(anom_idx) > 0:
                onset_times.append(float(anom_idx[0]))
        if len(onset_times) < 2:
            cascade_score = 0.0
        else:
            seq_len = max(len(svc.metrics.index) for svc in services)
            cascade_score = float(np.std(onset_times) / (seq_len + 1e-10))

        # 10. recovery_indicator
        recoveries = []
        for svc in services:
            cpu = svc.metrics["cpu_usage"].values
            if len(cpu) < 4:
                recoveries.append(0.0)
                continue
            peak = np.max(cpu)
            baseline = np.mean(cpu)
            if peak <= baseline:
                recoveries.append(0.0)
                continue
            tail = np.mean(cpu[-max(1, len(cpu) // 5):])
            recoveries.append(float((peak - tail) / (peak - baseline + 1e-10)))
        recovery_indicator = float(np.clip(np.mean(recoveries), 0.0, 1.0))

        # 11. affected_service_ratio
        affected = 0
        for svc in services:
            cpu = svc.metrics["cpu_usage"].values
            std = np.std(cpu)
            if std == 0.0:
                continue
            z = (cpu - np.mean(cpu)) / std
            if np.any(z > 2.0):
                affected += 1
        affected_service_ratio = affected / n

        # 12. variance_change_ratio
        var_ratios = []
        for svc in services:
            cpu = svc.metrics["cpu_usage"].values
            mid = len(cpu) // 2
            if mid == 0:
                var_ratios.append(0.0)
                continue
            var1 = np.var(cpu[:mid]) + 1e-10
            var2 = np.var(cpu[mid:]) + 1e-10
            var_ratios.append(float(np.log(var2 / var1)))
        variance_change_ratio = float(np.mean(var_ratios)) if var_ratios else 0.0

        return np.array([
            onset_gradient,
            peak_duration,
            cascade_score,
            recovery_indicator,
            affected_service_ratio,
            variance_change_ratio,
        ])

    # ------------------------------------------------------------------
    # Context features (5)
    #
    # NOTE: Context features are intentionally noisy to prevent label
    # leakage.  Without noise, ``event_active`` would be a perfect proxy
    # for the label (1.0 for all EXPECTED_LOAD, 0.0 for all FAULT),
    # allowing the model to achieve near-perfect accuracy without
    # learning from metric patterns.  Gaussian noise on
    # ``context_confidence`` and ``event_expected_impact``, plus a
    # label-independent ``recent_deployment`` base rate, force the model
    # to rely on workload / behavioral / statistical features.
    # ------------------------------------------------------------------

    def _context_features(
        self,
        context: Optional[Dict],
        services: Optional[List[ServiceMetrics]] = None,
        label: Optional[str] = None,
    ) -> np.ndarray:
        """Extract context features from an anomaly case.

        Context features are intentionally noisy to prevent label leakage
        and force the model to learn from metric patterns rather than
        relying on a deterministic context signal.

        Noise applied:
            - ``context_confidence``: Gaussian noise (std=0.1)
            - ``event_expected_impact``: Gaussian noise (std=0.05)
            - ``recent_deployment``: sampled from a base rate of 0.15
              for all cases, independent of the label
        """
        ctx = context or {}

        # 13. event_active
        event_active = 1.0 if "event_type" in ctx else 0.0

        # 14. event_expected_impact (with Gaussian noise, std=0.05)
        if "load_multiplier" in ctx:
            event_expected_impact = min(float(ctx["load_multiplier"]) / 5.0, 1.0)
        else:
            event_expected_impact = 0.0
        event_expected_impact = float(
            np.clip(event_expected_impact + np.random.normal(0, 0.05), 0.0, 1.0)
        )

        # 15. time_seasonality – derive from mean service timestamp
        if services:
            mean_ts = np.mean(
                [svc.metrics["timestamp"].mean() for svc in services]
            )
            hour = mean_ts % 24
            if 9 <= hour <= 20:
                time_seasonality = 0.7 + 0.3 * (hour - 9) / 11.0
            else:
                h = hour if hour < 9 else hour - 20
                time_seasonality = 0.1 + 0.3 * h / 8.0
            time_seasonality = float(
                np.clip(time_seasonality + np.random.uniform(-0.05, 0.05), 0.0, 1.0)
            )
        else:
            time_seasonality = 0.5

        # 16. recent_deployment – label-independent base rate of 0.15
        recent_deployment = 0.3 * np.random.random() if np.random.random() < 0.15 else 0.0

        # 17. context_confidence (with Gaussian noise, std=0.1)
        conf = 0.0
        if "event_type" in ctx:
            conf += 0.3
        if "load_multiplier" in ctx:
            conf += 0.2
        if "event_name" in ctx:
            conf += 0.1
        context_confidence = float(np.clip(min(conf, 1.0) + np.random.normal(0, 0.1), 0.0, 1.0))

        return np.array([
            event_active,
            event_expected_impact,
            time_seasonality,
            recent_deployment,
            context_confidence,
        ])

    # ------------------------------------------------------------------
    # Statistical features (13)
    # ------------------------------------------------------------------

    def _statistical_features(self, services: List[ServiceMetrics]) -> np.ndarray:
        if not services:
            return np.zeros(13)

        all_frames = pd.concat(
            [svc.metrics[_STAT_METRIC_COLS] for svc in services], ignore_index=True
        )

        means = all_frames.mean().values.astype(float)  # 6
        stds = all_frames.std().values.astype(float)  # 6
        stds = np.nan_to_num(stds, nan=0.0)

        # max error_rate across all services
        max_error = float(
            max(svc.metrics["error_rate"].max() for svc in services)
        )

        return np.concatenate([means, stds, [max_error]])  # 13

    # ------------------------------------------------------------------
    # Service-level features (6)
    # ------------------------------------------------------------------

    def _service_level_features(self, services: List[ServiceMetrics]) -> np.ndarray:
        n = len(services)
        if n == 0:
            return np.zeros(6)

        cpu_means = np.array([svc.metrics["cpu_usage"].mean() for svc in services])
        err_means = np.array([svc.metrics["error_rate"].mean() for svc in services])
        lat_means = np.array([svc.metrics["latency"].mean() for svc in services])

        overall_cpu = float(np.mean(cpu_means))
        overall_err = float(np.mean(err_means))

        n_services = float(n)
        max_cpu_service_ratio = float(np.max(cpu_means) / (overall_cpu + 1e-10))
        max_error_service_ratio = float(np.max(err_means) / (overall_err + 1e-10))
        cpu_spread = float(np.std(cpu_means))
        error_spread = float(np.std(err_means))
        latency_spread = float(np.std(lat_means))

        return np.array([
            n_services,
            max_cpu_service_ratio,
            max_error_service_ratio,
            cpu_spread,
            error_spread,
            latency_spread,
        ])
