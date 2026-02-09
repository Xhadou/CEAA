"""Shared utility functions for the CAAA pipeline."""

from typing import Dict

# ── Label constants ───────────────────────────────────────────────────

LABEL_FAULT = "FAULT"
LABEL_EXPECTED_LOAD = "EXPECTED_LOAD"
LABEL_UNKNOWN = "UNKNOWN"

LABEL_TO_INT: Dict[str, int] = {
    LABEL_FAULT: 0,
    LABEL_EXPECTED_LOAD: 1,
    LABEL_UNKNOWN: 2,
}

INT_TO_LABEL: Dict[int, str] = {v: k for k, v in LABEL_TO_INT.items()}


def labels_to_int(labels, label_map=None):
    """Convert string labels to integer encoding.

    Args:
        labels: Iterable of string labels ("FAULT", "EXPECTED_LOAD").
        label_map: Optional custom mapping. Defaults to LABEL_TO_INT.

    Returns:
        List of integer labels.
    """
    if label_map is None:
        label_map = LABEL_TO_INT
    return [label_map[label] for label in labels]


def int_to_labels(ints, label_map=None):
    """Convert integer labels back to strings.

    Args:
        ints: Iterable of integer labels (0, 1, 2).
        label_map: Optional custom mapping. Defaults to INT_TO_LABEL.

    Returns:
        List of string labels.
    """
    if label_map is None:
        label_map = INT_TO_LABEL
    return [label_map[int_label] for int_label in ints]
