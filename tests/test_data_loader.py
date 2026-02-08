"""Tests for the data loading and generation pipeline."""

import pytest
import numpy as np
import pandas as pd

from src.data_loader.data_types import ServiceMetrics, AnomalyCase
from src.data_loader.synthetic_generator import (
    SyntheticMetricsGenerator,
    SERVICE_NAMES,
    EVENT_TYPES,
    EVENT_TYPE_CONFIG,
)
from src.data_loader.fault_generator import FaultGenerator, FAULT_TYPES
from src.data_loader.dataset import generate_combined_dataset, generate_research_dataset

EXPECTED_COLUMNS = [
    "timestamp", "cpu_usage", "memory_usage", "request_rate",
    "error_rate", "latency", "network_in", "network_out",
]


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def sample_dataframe():
    """A small metrics DataFrame for unit tests."""
    return pd.DataFrame({
        "timestamp": np.arange(10),
        "cpu_usage": np.random.uniform(10, 30, 10),
        "memory_usage": np.random.uniform(20, 40, 10),
        "request_rate": np.random.uniform(50, 200, 10),
        "error_rate": np.random.uniform(0.001, 0.01, 10),
        "latency": np.random.uniform(10, 100, 10),
        "network_in": np.random.uniform(1000, 5000, 10),
        "network_out": np.random.uniform(1000, 5000, 10),
    })


@pytest.fixture
def synthetic_gen():
    return SyntheticMetricsGenerator(n_services=12, sequence_length=60, seed=42)


@pytest.fixture
def fault_gen():
    return FaultGenerator(n_services=12, sequence_length=60, seed=42)


# ── Data types ────────────────────────────────────────────────────────

class TestServiceMetrics:
    def test_service_metrics_creation(self, sample_dataframe):
        sm = ServiceMetrics(service_name="frontend", metrics=sample_dataframe)
        assert sm.service_name == "frontend"
        assert isinstance(sm.metrics, pd.DataFrame)
        assert len(sm.metrics) == 10
        assert list(sm.metrics.columns) == EXPECTED_COLUMNS


class TestAnomalyCase:
    def test_anomaly_case_creation(self, sample_dataframe):
        sm = ServiceMetrics(service_name="frontend", metrics=sample_dataframe)
        case = AnomalyCase(
            case_id="test_001",
            system="online-boutique",
            label="FAULT",
            services=[sm],
            context={"event_type": "flash_sale"},
            fault_service="frontend",
            fault_type="cpu_hog",
        )
        assert case.case_id == "test_001"
        assert case.system == "online-boutique"
        assert case.label == "FAULT"
        assert len(case.services) == 1
        assert case.context == {"event_type": "flash_sale"}
        assert case.fault_service == "frontend"
        assert case.fault_type == "cpu_hog"

    def test_anomaly_case_defaults(self, sample_dataframe):
        sm = ServiceMetrics(service_name="cart", metrics=sample_dataframe)
        case = AnomalyCase(
            case_id="test_002",
            system="online-boutique",
            label="EXPECTED_LOAD",
            services=[sm],
        )
        assert case.context == {}
        assert case.fault_service is None
        assert case.fault_type is None


# ── SyntheticMetricsGenerator ─────────────────────────────────────────

class TestSyntheticGeneratorNormal:
    def test_synthetic_generator_normal(self, synthetic_gen):
        metrics = synthetic_gen.generate_normal_metrics()
        assert isinstance(metrics, list)
        assert len(metrics) == 12

        for sm in metrics:
            assert isinstance(sm, ServiceMetrics)
            assert list(sm.metrics.columns) == EXPECTED_COLUMNS
            assert len(sm.metrics) == 60

            cpu = sm.metrics["cpu_usage"].values
            assert np.all(cpu >= 0) and np.all(cpu <= 100)

            err = sm.metrics["error_rate"].values
            assert np.all(err >= 0) and np.all(err < 0.1)

            mem = sm.metrics["memory_usage"].values
            assert np.all(mem >= 0) and np.all(mem <= 100)


class TestSyntheticGeneratorLoadSpike:
    def test_synthetic_generator_load_spike(self, synthetic_gen):
        result = synthetic_gen.generate_load_spike_metrics()
        assert isinstance(result, tuple) and len(result) == 2

        services, context = result
        assert isinstance(services, list) and len(services) == 12
        assert isinstance(context, dict)
        assert "event_type" in context
        assert "load_multiplier" in context

        # Error rates stay stable (clipped to [0,1])
        for sm in services:
            err = sm.metrics["error_rate"].values
            assert np.all(err >= 0) and np.all(err <= 1)

    def test_load_spike_cpu_increase(self):
        gen = SyntheticMetricsGenerator(n_services=4, sequence_length=60, seed=99)
        services, ctx = gen.generate_load_spike_metrics(load_multiplier=4.0)
        # During a big spike the mean CPU across services should be higher
        # than normal baseline (~20). With multiplier 4, at least some
        # services should have max CPU > 30.
        max_cpus = [sm.metrics["cpu_usage"].max() for sm in services]
        assert max(max_cpus) > 30

    def test_all_five_event_types_supported(self):
        """Verify all 5 event types can be generated."""
        assert len(EVENT_TYPES) == 5
        expected = {
            "flash_sale", "marketing_campaign", "scheduled_batch",
            "viral_content", "seasonal_peak",
        }
        assert set(EVENT_TYPES) == expected

        gen = SyntheticMetricsGenerator(n_services=4, sequence_length=60, seed=1)
        for etype in EVENT_TYPES:
            services, ctx = gen.generate_load_spike_metrics(event_type=etype)
            assert ctx["event_type"] == etype
            assert len(services) == 4
            # Error rate should stay low (stable)
            for sm in services:
                err = sm.metrics["error_rate"].values
                assert np.all(err >= 0) and np.all(err <= 1)

    def test_event_type_multiplier_ranges(self):
        """Verify per-event-type multiplier ranges are respected."""
        rng = np.random.RandomState(42)
        for etype, cfg in EVENT_TYPE_CONFIG.items():
            lo, hi = cfg["multiplier_range"]
            gen = SyntheticMetricsGenerator(
                n_services=2, sequence_length=60,
                seed=int(rng.randint(10000)),
            )
            _, ctx = gen.generate_load_spike_metrics(event_type=etype)
            mult = ctx["load_multiplier"]
            assert lo <= mult <= hi, (
                f"{etype}: multiplier {mult} outside [{lo}, {hi}]"
            )


# ── FaultGenerator ────────────────────────────────────────────────────

class TestFaultGenerator:
    def test_fault_generator(self, fault_gen):
        result = fault_gen.generate_fault_metrics()
        assert isinstance(result, tuple) and len(result) == 3

        services, fault_service, fault_type = result
        assert isinstance(services, list)
        assert len(services) == 12
        assert fault_service in SERVICE_NAMES
        assert fault_type in FAULT_TYPES

        # The faulty service should have higher mean error rate
        err_by_service = {
            sm.service_name: sm.metrics["error_rate"].mean()
            for sm in services
        }
        other_errors = [v for k, v in err_by_service.items() if k != fault_service]
        assert err_by_service[fault_service] > np.mean(other_errors)

    def test_all_eleven_fault_types_supported(self):
        """Verify all 11 fault types can be generated."""
        assert len(FAULT_TYPES) == 11

        for i, ft in enumerate(FAULT_TYPES):
            gen = FaultGenerator(n_services=4, sequence_length=60, seed=100 + i)
            services, fault_svc, returned_ft = gen.generate_fault_metrics(
                fault_type=ft
            )
            assert returned_ft == ft
            assert len(services) == 4
            # Faulty service should have elevated error rate
            err_by_svc = {
                sm.service_name: sm.metrics["error_rate"].mean()
                for sm in services
            }
            other_errors = [v for k, v in err_by_svc.items() if k != fault_svc]
            assert err_by_svc[fault_svc] > np.mean(other_errors), (
                f"Fault type {ft}: faulty service error rate not elevated"
            )


# ── generate_combined_dataset ─────────────────────────────────────────

class TestCombinedDataset:
    def test_generate_combined_dataset(self):
        fault_cases, load_cases = generate_combined_dataset(
            n_fault=5, n_load=5, seed=42
        )
        assert len(fault_cases) == 5
        assert len(load_cases) == 5
        assert all(c.label == "FAULT" for c in fault_cases)
        assert all(c.label == "EXPECTED_LOAD" for c in load_cases)

    def test_generate_combined_dataset_multiple_systems(self):
        systems = ["online-boutique", "sock-shop"]
        fault_cases, load_cases = generate_combined_dataset(
            n_fault=4, n_load=4, systems=systems, seed=7
        )
        assert len(fault_cases) == 4
        assert len(load_cases) == 4
        # Systems cycle: 0->online-boutique, 1->sock-shop, ...
        assert fault_cases[0].system == "online-boutique"
        assert fault_cases[1].system == "sock-shop"

    def test_empty_dataset(self):
        fault_cases, load_cases = generate_combined_dataset(
            n_fault=0, n_load=0, seed=1
        )
        assert fault_cases == []
        assert load_cases == []

    def test_single_case(self):
        fault_cases, load_cases = generate_combined_dataset(
            n_fault=1, n_load=1, seed=2
        )
        assert len(fault_cases) == 1
        assert len(load_cases) == 1
        assert fault_cases[0].label == "FAULT"
        assert load_cases[0].label == "EXPECTED_LOAD"


# ── generate_research_dataset ────────────────────────────────────────

class TestResearchDataset:
    def test_research_dataset_sizes(self):
        """Verify 735 FAULT + 600 EXPECTED_LOAD split correctly."""
        splits = generate_research_dataset(seed=42)

        assert set(splits.keys()) == {"train", "val", "test"}

        # Count labels per split
        train_fault = sum(1 for c in splits["train"] if c.label == "FAULT")
        train_load = sum(1 for c in splits["train"] if c.label == "EXPECTED_LOAD")
        val_fault = sum(1 for c in splits["val"] if c.label == "FAULT")
        val_load = sum(1 for c in splits["val"] if c.label == "EXPECTED_LOAD")
        test_fault = sum(1 for c in splits["test"] if c.label == "FAULT")
        test_load = sum(1 for c in splits["test"] if c.label == "EXPECTED_LOAD")

        # Train: 500 FAULT + 400 EXPECTED_LOAD = 900
        assert train_fault == 500
        assert train_load == 400
        assert len(splits["train"]) == 900

        # Val: 100 FAULT + 100 EXPECTED_LOAD = 200
        assert val_fault == 100
        assert val_load == 100
        assert len(splits["val"]) == 200

        # Test: 135 FAULT + 100 EXPECTED_LOAD = 235
        assert test_fault == 135
        assert test_load == 100
        assert len(splits["test"]) == 235

        # Total: 735 + 600 = 1335
        total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
        assert total == 1335

    def test_research_dataset_three_systems(self):
        """Verify cases are distributed across all 3 systems."""
        splits = generate_research_dataset(seed=42)
        all_cases = splits["train"] + splits["val"] + splits["test"]
        systems_seen = {c.system for c in all_cases}
        assert systems_seen == {"online-boutique", "sock-shop", "train-ticket"}

    def test_research_dataset_all_fault_types(self):
        """Verify all 11 fault types appear in the dataset."""
        splits = generate_research_dataset(seed=42)
        all_cases = splits["train"] + splits["val"] + splits["test"]
        fault_types_seen = {
            c.fault_type for c in all_cases
            if c.label == "FAULT" and c.fault_type is not None
        }
        assert fault_types_seen == set(FAULT_TYPES)

    def test_research_dataset_all_event_types(self):
        """Verify all 5 event types appear in the dataset."""
        splits = generate_research_dataset(seed=42)
        all_cases = splits["train"] + splits["val"] + splits["test"]
        event_types_seen = {
            c.context.get("event_type")
            for c in all_cases
            if c.label == "EXPECTED_LOAD"
        }
        assert event_types_seen == set(EVENT_TYPES)
