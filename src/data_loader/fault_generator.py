"""Fault scenario metrics generator for microservice anomaly cases."""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data_loader.data_types import ServiceMetrics
from src.data_loader.synthetic_generator import SERVICE_NAMES

logger = logging.getLogger(__name__)

# All 11 fault types matching RCAEval benchmark specification.
FAULT_TYPES: List[str] = [
    "cpu_hog",
    "memory_leak",
    "network_delay",
    "packet_loss",
    "disk_io",
    "pod_failure",
    "dns_failure",
    "connection_pool_exhaustion",
    "thread_leak",
    "config_error",
    "dependency_failure",
]


class FaultGenerator:
    """Generates metrics that simulate fault injection in a microservice system.

    Supports all 11 fault types from the RCAEval benchmark:
        cpu_hog, memory_leak, network_delay, packet_loss, disk_io,
        pod_failure, dns_failure, connection_pool_exhaustion, thread_leak,
        config_error, dependency_failure.

    Attributes:
        n_services: Number of services to generate metrics for.
        sequence_length: Number of time steps per service.
    """

    SERVICE_NAMES = SERVICE_NAMES
    FAULT_TYPES = FAULT_TYPES

    def __init__(
        self,
        n_services: int = 12,
        sequence_length: int = 60,
        seed: int = 42,
    ) -> None:
        """Initialize the fault generator.

        Args:
            n_services: Number of microservices.
            sequence_length: Number of timesteps in each sequence.
            seed: Random seed for reproducibility.
        """
        self.n_services = n_services
        self.sequence_length = sequence_length
        np.random.seed(seed)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _base_metrics(self, service_name: str) -> pd.DataFrame:
        """Generate a DataFrame of normal-operation metrics for one service.

        Args:
            service_name: Name of the service.

        Returns:
            DataFrame with normal baseline metrics.
        """
        n = self.sequence_length
        noise = lambda scale: np.random.normal(0, scale, n)

        cpu = np.random.uniform(10, 30) + noise(2)
        mem = np.random.uniform(20, 40) + noise(1.5)
        req = np.random.uniform(50, 200) + noise(5)
        err = np.random.uniform(0.001, 0.01) + noise(0.001)
        lat = np.random.uniform(10, 100) + noise(3)
        net_in = np.random.uniform(1000, 5000) + noise(50)
        net_out = np.random.uniform(1000, 5000) + noise(50)

        cpu = np.clip(cpu, 0, 100)
        mem = np.clip(mem, 0, 100)
        req = np.clip(req, 0, None)
        err = np.clip(err, 0, 1)
        lat = np.clip(lat, 0, None)
        net_in = np.clip(net_in, 0, None)
        net_out = np.clip(net_out, 0, None)

        return pd.DataFrame(
            {
                "timestamp": np.arange(n),
                "cpu_usage": cpu,
                "memory_usage": mem,
                "request_rate": req,
                "error_rate": err,
                "latency": lat,
                "network_in": net_in,
                "network_out": net_out,
            }
        )

    def _inject_fault(
        self, df: pd.DataFrame, fault_type: str, fault_start: int
    ) -> pd.DataFrame:
        """Inject a fault into the metrics starting at fault_start.

        Args:
            df: Baseline metrics DataFrame.
            fault_type: One of the 11 supported fault types.
            fault_start: Index at which the fault begins.

        Returns:
            Modified DataFrame with the fault injected.
        """
        df = df.copy()
        n = len(df)
        fault_len = n - fault_start
        fault_slice = slice(fault_start, n)

        if fault_type == "cpu_hog":
            df.loc[fault_slice, "cpu_usage"] = np.clip(
                df.loc[fault_slice, "cpu_usage"].values
                + np.random.uniform(30, 60, fault_len),
                0, 100,
            )
            df.loc[fault_slice, "error_rate"] = np.clip(
                df.loc[fault_slice, "error_rate"].values
                + np.random.uniform(0.1, 0.5),
                0, 1,
            )

        elif fault_type == "memory_leak":
            # Gradual memory increase (leak pattern)
            leak_ramp = np.linspace(0, np.random.uniform(30, 55), fault_len)
            df.loc[fault_slice, "memory_usage"] = np.clip(
                df.loc[fault_slice, "memory_usage"].values + leak_ramp,
                0, 100,
            )
            df.loc[fault_slice, "error_rate"] = np.clip(
                df.loc[fault_slice, "error_rate"].values
                + np.random.uniform(0.1, 0.5),
                0, 1,
            )

        elif fault_type == "network_delay":
            df.loc[fault_slice, "latency"] = np.clip(
                df.loc[fault_slice, "latency"].values
                + np.random.uniform(200, 800, fault_len),
                0, None,
            )
            df.loc[fault_slice, "error_rate"] = np.clip(
                df.loc[fault_slice, "error_rate"].values
                + np.random.uniform(0.05, 0.3),
                0, 1,
            )

        elif fault_type == "packet_loss":
            df.loc[fault_slice, "network_in"] = np.clip(
                df.loc[fault_slice, "network_in"].values * 0.1, 0, None,
            )
            df.loc[fault_slice, "network_out"] = np.clip(
                df.loc[fault_slice, "network_out"].values * 0.1, 0, None,
            )
            df.loc[fault_slice, "error_rate"] = np.clip(
                df.loc[fault_slice, "error_rate"].values
                + np.random.uniform(0.1, 0.5),
                0, 1,
            )

        elif fault_type == "disk_io":
            df.loc[fault_slice, "latency"] = np.clip(
                df.loc[fault_slice, "latency"].values
                + np.random.uniform(100, 500, fault_len),
                0, None,
            )
            df.loc[fault_slice, "cpu_usage"] = np.clip(
                df.loc[fault_slice, "cpu_usage"].values
                + np.random.uniform(10, 30, fault_len),
                0, 100,
            )
            df.loc[fault_slice, "error_rate"] = np.clip(
                df.loc[fault_slice, "error_rate"].values
                + np.random.uniform(0.05, 0.2),
                0, 1,
            )

        elif fault_type == "pod_failure":
            # Service goes completely down — requests drop, errors spike
            df.loc[fault_slice, "request_rate"] = np.clip(
                df.loc[fault_slice, "request_rate"].values * 0.05, 0, None,
            )
            df.loc[fault_slice, "error_rate"] = np.clip(
                df.loc[fault_slice, "error_rate"].values
                + np.random.uniform(0.4, 0.8),
                0, 1,
            )
            df.loc[fault_slice, "cpu_usage"] = np.clip(
                df.loc[fault_slice, "cpu_usage"].values * 0.1, 0, 100,
            )

        elif fault_type == "dns_failure":
            df.loc[fault_slice, "latency"] = np.clip(
                df.loc[fault_slice, "latency"].values
                + np.random.uniform(500, 2000, fault_len),
                0, None,
            )
            df.loc[fault_slice, "error_rate"] = np.clip(
                df.loc[fault_slice, "error_rate"].values
                + np.random.uniform(0.2, 0.6),
                0, 1,
            )
            df.loc[fault_slice, "network_in"] = np.clip(
                df.loc[fault_slice, "network_in"].values * 0.3, 0, None,
            )

        elif fault_type == "connection_pool_exhaustion":
            df.loc[fault_slice, "latency"] = np.clip(
                df.loc[fault_slice, "latency"].values
                + np.random.uniform(300, 1000, fault_len),
                0, None,
            )
            df.loc[fault_slice, "error_rate"] = np.clip(
                df.loc[fault_slice, "error_rate"].values
                + np.random.uniform(0.15, 0.5),
                0, 1,
            )
            # Requests back up
            df.loc[fault_slice, "request_rate"] = np.clip(
                df.loc[fault_slice, "request_rate"].values * 0.4, 0, None,
            )

        elif fault_type == "thread_leak":
            # CPU climbs gradually, latency grows
            leak_ramp = np.linspace(0, np.random.uniform(20, 50), fault_len)
            df.loc[fault_slice, "cpu_usage"] = np.clip(
                df.loc[fault_slice, "cpu_usage"].values + leak_ramp,
                0, 100,
            )
            df.loc[fault_slice, "latency"] = np.clip(
                df.loc[fault_slice, "latency"].values
                + np.random.uniform(50, 300, fault_len),
                0, None,
            )
            df.loc[fault_slice, "error_rate"] = np.clip(
                df.loc[fault_slice, "error_rate"].values
                + np.random.uniform(0.05, 0.3),
                0, 1,
            )

        elif fault_type == "config_error":
            # Immediate error spike, some latency increase
            df.loc[fault_slice, "error_rate"] = np.clip(
                df.loc[fault_slice, "error_rate"].values
                + np.random.uniform(0.2, 0.7),
                0, 1,
            )
            df.loc[fault_slice, "latency"] = np.clip(
                df.loc[fault_slice, "latency"].values
                + np.random.uniform(50, 200, fault_len),
                0, None,
            )

        elif fault_type == "dependency_failure":
            # Downstream dependency dies — errors spike, latency goes up
            df.loc[fault_slice, "error_rate"] = np.clip(
                df.loc[fault_slice, "error_rate"].values
                + np.random.uniform(0.2, 0.6),
                0, 1,
            )
            df.loc[fault_slice, "latency"] = np.clip(
                df.loc[fault_slice, "latency"].values
                + np.random.uniform(200, 600, fault_len),
                0, None,
            )
            df.loc[fault_slice, "request_rate"] = np.clip(
                df.loc[fault_slice, "request_rate"].values * 0.5, 0, None,
            )

        return df

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_fault_metrics(
        self,
        system: str = "online-boutique",
        fault_type: Optional[str] = None,
        fault_service: Optional[str] = None,
    ) -> Tuple[List[ServiceMetrics], str, str]:
        """Generate metrics that simulate a fault in one microservice.

        Characteristics:
            - Error rate increases on the faulty service.
            - Sudden onset (step change) at a random point in the middle third.
            - Localised to a single service; others remain normal.
            - Fault type determines which metrics are affected.

        Args:
            system: Name of the system.
            fault_type: Type of fault to inject. Random from the 11 supported
                types if not given.
            fault_service: Service to inject the fault into. Random (excluding
                loadgenerator) if not given.

        Returns:
            Tuple of (list of ServiceMetrics, fault_service name, fault_type).
        """
        eligible_services = [s for s in self.SERVICE_NAMES[: self.n_services] if s != "loadgenerator"]

        if fault_type is None:
            fault_type = str(np.random.choice(self.FAULT_TYPES))
        if fault_service is None:
            fault_service = str(np.random.choice(eligible_services))

        # Fault starts at a random point in the middle third of the sequence
        mid_start = self.sequence_length // 3
        mid_end = 2 * self.sequence_length // 3
        fault_start = int(np.random.randint(mid_start, mid_end))

        logger.info(
            "Generating fault metrics: system=%s service=%s type=%s start=%d",
            system,
            fault_service,
            fault_type,
            fault_start,
        )

        results: List[ServiceMetrics] = []
        for name in self.SERVICE_NAMES[: self.n_services]:
            df = self._base_metrics(name)
            if name == fault_service:
                df = self._inject_fault(df, fault_type, fault_start)
            results.append(ServiceMetrics(service_name=name, metrics=df))

        return results, fault_service, fault_type
