"""Combined dataset generation for CAAA anomaly attribution."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.data_loader.data_types import AnomalyCase, ServiceMetrics
from src.data_loader.fault_generator import FaultGenerator
from src.data_loader.synthetic_generator import SyntheticMetricsGenerator, EVENT_TYPES

logger = logging.getLogger(__name__)

# Systems matching RCAEval benchmark.
RESEARCH_SYSTEMS: List[str] = ["online-boutique", "sock-shop", "train-ticket"]


def generate_combined_dataset(
    n_fault: int = 50,
    n_load: int = 50,
    systems: Optional[List[str]] = None,
    seed: int = 42,
    include_hard: bool = False,
) -> Tuple[List[AnomalyCase], List[AnomalyCase]]:
    """Generate a combined dataset of FAULT and EXPECTED_LOAD cases.

    Args:
        n_fault: Number of fault cases to generate.
        n_load: Number of expected-load cases to generate.
        systems: List of system names to cycle through. Defaults to
            ["online-boutique"].
        seed: Random seed for reproducibility.
        include_hard: When True, replace 20% of cases (5% each type)
            with hard/adversarial scenarios from :func:`generate_hard_dataset`.

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
        # 10% of fault cases get a fake context with event_type to prevent
        # event_active from being a perfect proxy for the label.
        if np.random.random() < 0.10:
            fake_event = str(np.random.choice([
                "flash_sale", "marketing_campaign", "scheduled_batch",
            ]))
            fault_context["event_type"] = fake_event
            fault_context["event_name"] = f"{fake_event}_event"
            fault_context["load_multiplier"] = float(
                np.random.uniform(1.2, 2.5)
            )
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
        # 15% of load cases get empty context (simulating unscheduled load
        # spikes with no calendar entry) to prevent label leakage.
        if np.random.random() < 0.15:
            context = {}
        load_cases.append(
            AnomalyCase(
                case_id=f"load_{i:04d}",
                system=system,
                label="EXPECTED_LOAD",
                services=services,
                context=context,
            )
        )

    # Optionally replace 20% of cases with hard/adversarial scenarios
    if include_hard:
        n_hard_fault = max(1, int(n_fault * 0.10))  # 10% of faults per hard fault type
        n_hard_load = max(1, int(n_load * 0.10))    # 10% of loads per hard load type
        hard_fault, hard_load = generate_hard_dataset(
            n_per_type=max(1, max(n_hard_fault, n_hard_load) // 2),
            systems=systems,
            seed=seed + 100,
        )
        # Replace tail of standard cases with hard cases
        if hard_fault:
            fault_cases = fault_cases[: max(0, len(fault_cases) - len(hard_fault))] + hard_fault
        if hard_load:
            load_cases = load_cases[: max(0, len(load_cases) - len(hard_load))] + hard_load

    logger.info(
        "Dataset complete: %d fault cases, %d load cases",
        len(fault_cases),
        len(load_cases),
    )
    return fault_cases, load_cases


def generate_research_dataset(
    seed: int = 42,
) -> Dict[str, List[AnomalyCase]]:
    """Generate the full research dataset matching the specification.

    Produces 735 FAULT + 600 EXPECTED_LOAD = 1335 total cases across
    3 systems (online-boutique, sock-shop, train-ticket), split into
    train / val / test partitions:

    +---------+-------+---------------+-------+
    | Split   | FAULT | EXPECTED_LOAD | Total |
    +---------+-------+---------------+-------+
    | train   |   500 |           400 |   900 |
    | val     |   100 |           100 |   200 |
    | test    |   135 |           100 |   235 |
    +---------+-------+---------------+-------+
    | Total   |   735 |           600 | 1,335 |
    +---------+-------+---------------+-------+

    Args:
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with keys ``"train"``, ``"val"``, ``"test"``, each
        mapping to a list of ``AnomalyCase`` objects.
    """
    systems = RESEARCH_SYSTEMS

    fault_cases, load_cases = generate_combined_dataset(
        n_fault=735, n_load=600, systems=systems, seed=seed,
    )

    rng = np.random.RandomState(seed)

    # Shuffle within each class
    fault_idx = rng.permutation(len(fault_cases)).tolist()
    load_idx = rng.permutation(len(load_cases)).tolist()

    # Split FAULT: 500 train, 100 val, 135 test
    fault_train = [fault_cases[i] for i in fault_idx[:500]]
    fault_val = [fault_cases[i] for i in fault_idx[500:600]]
    fault_test = [fault_cases[i] for i in fault_idx[600:735]]

    # Split EXPECTED_LOAD: 400 train, 100 val, 100 test
    load_train = [load_cases[i] for i in load_idx[:400]]
    load_val = [load_cases[i] for i in load_idx[400:500]]
    load_test = [load_cases[i] for i in load_idx[500:600]]

    train = fault_train + load_train
    val = fault_val + load_val
    test = fault_test + load_test

    # Shuffle each split
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    logger.info(
        "Research dataset: train=%d, val=%d, test=%d (total=%d)",
        len(train), len(val), len(test),
        len(train) + len(val) + len(test),
    )

    return {"train": train, "val": val, "test": test}


def generate_rcaeval_dataset(
    dataset: str = "RE1",
    system: str = "online-boutique",
    n_load_per_fault: int = 1,
    data_dir: str = "data/raw",
    seed: int = 42,
) -> Tuple[List[AnomalyCase], List[AnomalyCase]]:
    """Load FAULT cases from RCAEval and generate matching EXPECTED_LOAD cases.

    This is the intended research pipeline: real fault data from the RCAEval
    benchmark paired with synthetic expected-load cases for attribution training.

    Args:
        dataset: RCAEval dataset identifier (``"RE1"`` or ``"RE2"``).
        system: Microservice system (``"online-boutique"``, ``"sock-shop"``,
            or ``"train-ticket"``).
        n_load_per_fault: Number of synthetic load cases per fault case.
        data_dir: Path to downloaded RCAEval data.
        seed: Random seed.

    Returns:
        Tuple of (fault_cases, load_cases) as AnomalyCase lists.

    Raises:
        FileNotFoundError: If the RCAEval data has not been downloaded.
            Call :func:`src.data_loader.download_data.download_rcaeval_dataset`
            first.
    """
    from src.data_loader.rcaeval_loader import RCAEvalLoader

    loader = RCAEvalLoader(data_dir=data_dir)
    fault_cases = loader.load_dataset(dataset=dataset, system=system)

    if not fault_cases:
        raise FileNotFoundError(
            f"No RCAEval data found at {data_dir}/{dataset}/{system}. "
            f"Run: python -c \"from src.data_loader.download_data import "
            f"download_rcaeval_dataset; download_rcaeval_dataset('{dataset}', "
            f"['{system}'])\""
        )

    logger.info("Loaded %d fault cases from RCAEval %s/%s", len(fault_cases), dataset, system)

    # Generate synthetic EXPECTED_LOAD cases
    load_gen = SyntheticMetricsGenerator(seed=seed)
    load_cases: List[AnomalyCase] = []

    for i, fault_case in enumerate(fault_cases):
        for j in range(n_load_per_fault):
            services, context = load_gen.generate_load_spike_metrics(
                system=fault_case.system,
            )
            load_cases.append(
                AnomalyCase(
                    case_id=f"load_rcaeval_{i:04d}_{j:02d}",
                    system=fault_case.system,
                    label="EXPECTED_LOAD",
                    services=services,
                    context=context,
                )
            )

    logger.info(
        "RCAEval dataset: %d real faults + %d synthetic loads = %d total",
        len(fault_cases), len(load_cases), len(fault_cases) + len(load_cases),
    )
    return fault_cases, load_cases


# ── Hard / adversarial scenario types ────────────────────────────────

HARD_SCENARIO_TYPES: List[str] = [
    "fault_during_event",
    "capacity_exceeded_load",
    "gradual_fault",
    "partial_load",
]


def generate_hard_dataset(
    n_per_type: int = 5,
    systems: Optional[List[str]] = None,
    seed: int = 42,
) -> Tuple[List[AnomalyCase], List[AnomalyCase]]:
    """Generate hard/adversarial evaluation cases.

    Produces four scenario types where the distinction between FAULT and
    EXPECTED_LOAD is intentionally ambiguous:

    - **fault_during_event** (FAULT): A real fault occurs on top of a
      legitimate load spike. Metrics show both load and fault signatures.
    - **capacity_exceeded_load** (EXPECTED_LOAD): An extreme load spike
      (5-10×) causes error-rate increases and latency degradation purely
      from capacity exhaustion, not from a fault.
    - **gradual_fault** (FAULT): A fault that ramps up slowly (like a
      memory leak) across the full window, mimicking gradual load.
    - **partial_load** (EXPECTED_LOAD): A load spike that affects only
      3-4 services, overlapping with the localised-impact pattern of
      faults.

    Args:
        n_per_type: Number of cases to generate per scenario type.
        systems: System names to cycle through.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (fault_cases, load_cases).
    """
    if systems is None:
        systems = ["online-boutique"]

    np.random.seed(seed)
    fault_gen = FaultGenerator(seed=seed)
    load_gen = SyntheticMetricsGenerator(seed=seed + 1)

    fault_cases: List[AnomalyCase] = []
    load_cases: List[AnomalyCase] = []

    # --- (a) FAULT_DURING_EVENT: fault injected on top of load spike ---
    for i in range(n_per_type):
        system = systems[i % len(systems)]
        # Generate a load spike first
        services, context = load_gen.generate_load_spike_metrics(system=system)
        # Pick a random service (not loadgenerator) to inject a fault
        eligible = [s for s in services if s.service_name != "loadgenerator"]
        fault_svc = eligible[np.random.randint(len(eligible))]
        fault_type = str(np.random.choice(fault_gen.FAULT_TYPES))
        # Inject fault at a random point during the spike
        n_ts = len(fault_svc.metrics)
        fault_start = int(np.random.randint(n_ts // 4, 3 * n_ts // 4))
        fault_svc.metrics = fault_gen._inject_fault(
            fault_svc.metrics, fault_type, fault_start,
        )
        fault_cases.append(
            AnomalyCase(
                case_id=f"hard_fault_during_event_{i:04d}",
                system=system,
                label="FAULT",
                services=services,
                context=context,  # event IS happening
                fault_service=fault_svc.service_name,
                fault_type=fault_type,
            )
        )

    # --- (b) CAPACITY_EXCEEDED_LOAD: extreme load causes errors ---
    for i in range(n_per_type):
        system = systems[i % len(systems)]
        high_mult = float(np.random.uniform(5.0, 10.0))
        services, context = load_gen.generate_load_spike_metrics(
            system=system, load_multiplier=high_mult,
        )
        # Add load-correlated error increases (0.05-0.15) during the spike
        spike_start = context.get("spike_start", 0)
        spike_end = context.get("spike_end", len(services[0].metrics))
        for svc in services:
            df = svc.metrics
            n_ts = len(df)
            actual_end = min(spike_end, n_ts)
            spike_sl = slice(spike_start, actual_end)
            # Moderate error increase correlated with load envelope
            actual_len = len(df.loc[spike_sl, "error_rate"])
            if actual_len > 0:
                err_increase = np.random.uniform(0.05, 0.15) * np.ones(actual_len)
                err_increase *= np.linspace(0.3, 1.0, actual_len)  # ramp with load
                df.loc[spike_sl, "error_rate"] = np.clip(
                    df.loc[spike_sl, "error_rate"].values + err_increase, 0, 1,
                )
        load_cases.append(
            AnomalyCase(
                case_id=f"hard_capacity_exceeded_{i:04d}",
                system=system,
                label="EXPECTED_LOAD",
                services=services,
                context=context,
            )
        )

    # --- (c) GRADUAL_FAULT: slow ramp-up over full window ---
    for i in range(n_per_type):
        system = systems[i % len(systems)]
        services, fault_service, fault_type = fault_gen.generate_fault_metrics(
            system=system,
        )
        # Replace the sudden fault with a gradual ramp over the full window
        for svc in services:
            if svc.service_name == fault_service:
                df = svc.metrics
                n_ts = len(df)
                ramp = np.linspace(0, 1, n_ts)
                # Gradually increase error_rate
                df["error_rate"] = np.clip(
                    df["error_rate"].values + ramp * np.random.uniform(0.1, 0.4),
                    0, 1,
                )
                # Gradually increase latency
                df["latency"] = np.clip(
                    df["latency"].values + ramp * np.random.uniform(50, 300),
                    0, None,
                )
                # Gradually increase CPU
                df["cpu_usage"] = np.clip(
                    df["cpu_usage"].values + ramp * np.random.uniform(10, 40),
                    0, 100,
                )
        fault_cases.append(
            AnomalyCase(
                case_id=f"hard_gradual_fault_{i:04d}",
                system=system,
                label="FAULT",
                services=services,
                context={},
                fault_service=fault_service,
                fault_type=fault_type,
            )
        )

    # --- (d) PARTIAL_LOAD: spike affects only 3-4 services ---
    for i in range(n_per_type):
        system = systems[i % len(systems)]
        services, context = load_gen.generate_load_spike_metrics(system=system)
        # Reset most services to baseline (keep only 3-4 affected)
        n_affected = np.random.randint(
            min(3, len(services)), min(5, len(services)) + 1,
        )
        affected_indices = set(
            np.random.choice(len(services), size=n_affected, replace=False)
        )
        for idx, svc in enumerate(services):
            if idx not in affected_indices:
                # Regenerate as normal baseline metrics
                svc.metrics = load_gen._base_metrics(svc.service_name)
        load_cases.append(
            AnomalyCase(
                case_id=f"hard_partial_load_{i:04d}",
                system=system,
                label="EXPECTED_LOAD",
                services=services,
                context=context,
            )
        )

    logger.info(
        "Hard dataset: %d fault cases, %d load cases (n_per_type=%d)",
        len(fault_cases), len(load_cases), n_per_type,
    )
    return fault_cases, load_cases
