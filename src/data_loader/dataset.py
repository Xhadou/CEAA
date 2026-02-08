"""Combined dataset generation for CAAA anomaly attribution."""

import logging
from typing import List, Optional, Tuple

import numpy as np

from src.data_loader.data_types import AnomalyCase
from src.data_loader.fault_generator import FaultGenerator
from src.data_loader.synthetic_generator import SyntheticMetricsGenerator

logger = logging.getLogger(__name__)


def generate_combined_dataset(
    n_fault: int = 50,
    n_load: int = 50,
    systems: Optional[List[str]] = None,
    seed: int = 42,
) -> Tuple[List[AnomalyCase], List[AnomalyCase]]:
    """Generate a combined dataset of FAULT and EXPECTED_LOAD cases.

    Args:
        n_fault: Number of fault cases to generate.
        n_load: Number of expected-load cases to generate.
        systems: List of system names to cycle through. Defaults to
            ["online-boutique"].
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (fault_cases, load_cases).
    """
    if systems is None:
        systems = ["online-boutique"]

    fault_gen = FaultGenerator(seed=seed)
    load_gen = SyntheticMetricsGenerator(seed=seed + 1)

    fault_cases: List[AnomalyCase] = []
    load_cases: List[AnomalyCase] = []

    # Generate fault cases
    logger.info("Generating %d fault cases", n_fault)
    for i in range(n_fault):
        system = systems[i % len(systems)]
        services, fault_service, fault_type = fault_gen.generate_fault_metrics(
            system=system
        )
        fault_context = {}
        if np.random.random() < 0.3:
            fault_context["recent_deployment"] = True
        fault_cases.append(
            AnomalyCase(
                case_id=f"fault_{i:04d}",
                system=system,
                label="FAULT",
                services=services,
                context=fault_context,
                fault_service=fault_service,
                fault_type=fault_type,
            )
        )

    # Generate expected-load cases
    logger.info("Generating %d expected-load cases", n_load)
    for i in range(n_load):
        system = systems[i % len(systems)]
        services, context = load_gen.generate_load_spike_metrics(system=system)
        load_cases.append(
            AnomalyCase(
                case_id=f"load_{i:04d}",
                system=system,
                label="EXPECTED_LOAD",
                services=services,
                context=context,
            )
        )

    logger.info(
        "Dataset complete: %d fault cases, %d load cases",
        len(fault_cases),
        len(load_cases),
    )
    return fault_cases, load_cases
