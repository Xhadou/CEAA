"""Single source of truth for feature vector layout.

All feature indices, names, and group boundaries are defined here.
Every module that needs to know about feature positions imports from this
module rather than using hardcoded integers.

Feature vector layout (36 features total)::

    [0:6]   Workload features
    [6:12]  Behavioral features
    [12:17] Context features
    [17:30] Statistical features
    [30:36] Service-level features

Usage::

    from src.features.feature_schema import CONTEXT_START, CONTEXT_END, N_FEATURES
    context = x[:, CONTEXT_START:CONTEXT_END]
"""

from typing import Dict, List, Tuple

# ── Feature group names (ordered) ────────────────────────────────────

WORKLOAD_NAMES: List[str] = [
    "global_load_ratio",
    "cpu_request_correlation",
    "cross_service_sync",
    "error_rate_delta",
    "latency_cpu_correlation",
    "change_point_magnitude",
]

BEHAVIORAL_NAMES: List[str] = [
    "onset_gradient",
    "peak_duration",
    "cascade_score",
    "recovery_indicator",
    "affected_service_ratio",
    "variance_change_ratio",
]

CONTEXT_NAMES: List[str] = [
    "event_active",
    "event_expected_impact",
    "time_seasonality",
    "recent_deployment",
    "context_confidence",
]

STAT_METRIC_COLS: List[str] = [
    "cpu_usage",
    "memory_usage",
    "request_rate",
    "error_rate",
    "latency",
    "network_in",
]

STATISTICAL_NAMES: List[str] = (
    [f"mean_{c}" for c in STAT_METRIC_COLS]
    + [f"std_{c}" for c in STAT_METRIC_COLS]
    + ["max_error_rate"]
)

SERVICE_LEVEL_NAMES: List[str] = [
    "n_services",
    "max_cpu_service_ratio",
    "max_error_service_ratio",
    "cpu_spread",
    "error_spread",
    "latency_spread",
]

# ── Derived constants ─────────────────────────────────────────────────

N_WORKLOAD: int = len(WORKLOAD_NAMES)          # 6
N_BEHAVIORAL: int = len(BEHAVIORAL_NAMES)      # 6
N_CONTEXT: int = len(CONTEXT_NAMES)            # 5
N_STATISTICAL: int = len(STATISTICAL_NAMES)    # 13
N_SERVICE_LEVEL: int = len(SERVICE_LEVEL_NAMES)  # 6

N_FEATURES: int = (
    N_WORKLOAD + N_BEHAVIORAL + N_CONTEXT + N_STATISTICAL + N_SERVICE_LEVEL
)
assert N_FEATURES == 36, f"Feature count mismatch: expected 36, got {N_FEATURES}"

# ── Index ranges (start, end) — use as x[:, start:end] ───────────────

WORKLOAD_RANGE: Tuple[int, int] = (0, N_WORKLOAD)
BEHAVIORAL_RANGE: Tuple[int, int] = (
    WORKLOAD_RANGE[1],
    WORKLOAD_RANGE[1] + N_BEHAVIORAL,
)
CONTEXT_RANGE: Tuple[int, int] = (
    BEHAVIORAL_RANGE[1],
    BEHAVIORAL_RANGE[1] + N_CONTEXT,
)
STATISTICAL_RANGE: Tuple[int, int] = (
    CONTEXT_RANGE[1],
    CONTEXT_RANGE[1] + N_STATISTICAL,
)
SERVICE_LEVEL_RANGE: Tuple[int, int] = (
    STATISTICAL_RANGE[1],
    STATISTICAL_RANGE[1] + N_SERVICE_LEVEL,
)

# Convenience aliases used across the codebase
CONTEXT_START: int = CONTEXT_RANGE[0]  # 12
CONTEXT_END: int = CONTEXT_RANGE[1]    # 17

# ── All feature names in canonical order ──────────────────────────────

ALL_FEATURE_NAMES: List[str] = (
    WORKLOAD_NAMES
    + BEHAVIORAL_NAMES
    + CONTEXT_NAMES
    + STATISTICAL_NAMES
    + SERVICE_LEVEL_NAMES
)
assert len(ALL_FEATURE_NAMES) == N_FEATURES

# ── Feature group index mapping (for visualization / analysis) ────────

FEATURE_GROUPS: Dict[str, List[int]] = {
    "workload": list(range(*WORKLOAD_RANGE)),
    "behavioral": list(range(*BEHAVIORAL_RANGE)),
    "context": list(range(*CONTEXT_RANGE)),
    "statistical": list(range(*STATISTICAL_RANGE)),
    "service-level": list(range(*SERVICE_LEVEL_RANGE)),
}

# ── Feature group color mapping (for visualization) ──────────────────

FEATURE_GROUP_COLORS: Dict[str, str] = {
    "workload": "#3498db",
    "behavioral": "#e74c3c",
    "context": "#2ecc71",
    "statistical": "#f39c12",
    "service-level": "#9b59b6",
}
