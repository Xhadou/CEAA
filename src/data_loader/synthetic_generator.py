"""Synthetic metrics generator for normal and load-spike scenarios."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data_loader.data_types import ServiceMetrics

logger = logging.getLogger(__name__)

SERVICE_NAMES: List[str] = [
    "frontend",
    "cart",
    "checkout",
    "payment",
    "shipping",
    "email",
    "currency",
    "productcatalog",
    "recommendation",
    "ad",
    "redis-cart",
    "loadgenerator",
]


class SyntheticMetricsGenerator:
    """Generates realistic synthetic microservice metrics.

    Attributes:
        n_services: Number of services to generate metrics for.
        sequence_length: Number of time steps per service.
    """

    SERVICE_NAMES = SERVICE_NAMES

    def __init__(
        self,
        n_services: int = 12,
        sequence_length: int = 60,
        seed: int = 42,
    ) -> None:
        """Initialize the generator.

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

        # Clamp to sensible ranges
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_normal_metrics(
        self, system: str = "online-boutique"
    ) -> List[ServiceMetrics]:
        """Generate normal-operation metrics for all services.

        Args:
            system: Name of the system.

        Returns:
            List of ServiceMetrics, one per service.
        """
        logger.info("Generating normal metrics for system=%s", system)
        results: List[ServiceMetrics] = []
        for name in self.SERVICE_NAMES[: self.n_services]:
            df = self._base_metrics(name)
            results.append(ServiceMetrics(service_name=name, metrics=df))
        return results

    def generate_load_spike_metrics(
        self,
        system: str = "online-boutique",
        load_multiplier: Optional[float] = None,
        event_type: Optional[str] = None,
    ) -> Tuple[List[ServiceMetrics], Dict]:
        """Generate metrics showing a legitimate load spike.

        Characteristics:
            - Error rate stays stable (multiplied by 0.8-1.1 only).
            - Gradual ramp-up (10-20 % of sequence) and ramp-down.
            - All services scale together (high cross-service correlation).
            - CPU, request_rate, latency, network all increase proportionally.

        Args:
            system: Name of the system.
            load_multiplier: Peak multiplier for load metrics. Random 1.5-5.0
                if not provided.
            event_type: Type of event causing the spike. Random if not provided.

        Returns:
            Tuple of (list of ServiceMetrics, context dict).
        """
        event_types = [
            "flash_sale",
            "marketing_campaign",
            "scheduled_batch",
            "viral_content",
        ]
        if load_multiplier is None:
            load_multiplier = float(np.random.uniform(1.5, 5.0))
        if event_type is None:
            event_type = str(np.random.choice(event_types))

        logger.info(
            "Generating load-spike metrics: system=%s multiplier=%.2f event=%s",
            system,
            load_multiplier,
            event_type,
        )

        n = self.sequence_length
        ramp_frac = np.random.uniform(0.10, 0.20)
        ramp_len = max(1, int(n * ramp_frac))

        # Build a shared envelope: 0 → 1 ramp-up, plateau, 1 → 0 ramp-down
        spike_start = np.random.randint(int(n * 0.15), int(n * 0.35))
        spike_end = min(n, spike_start + np.random.randint(int(n * 0.3), int(n * 0.5)))
        ramp_down_start = max(spike_start + ramp_len, spike_end - ramp_len)

        envelope = np.zeros(n)
        # ramp-up
        up_end = min(spike_start + ramp_len, n)
        envelope[spike_start:up_end] = np.linspace(0, 1, up_end - spike_start)
        # plateau
        envelope[up_end:ramp_down_start] = 1.0
        # ramp-down
        if ramp_down_start < spike_end:
            envelope[ramp_down_start:spike_end] = np.linspace(
                1, 0, spike_end - ramp_down_start
            )

        results: List[ServiceMetrics] = []
        for name in self.SERVICE_NAMES[: self.n_services]:
            df = self._base_metrics(name)

            mult = 1.0 + (load_multiplier - 1.0) * envelope
            df["cpu_usage"] = np.clip(df["cpu_usage"] * mult, 0, 100)
            df["request_rate"] = np.clip(df["request_rate"] * mult, 0, None)
            df["latency"] = np.clip(df["latency"] * mult, 0, None)
            df["network_in"] = np.clip(df["network_in"] * mult, 0, None)
            df["network_out"] = np.clip(df["network_out"] * mult, 0, None)

            # Error rate stays stable
            err_mult = np.random.uniform(0.8, 1.1)
            df["error_rate"] = np.clip(df["error_rate"] * err_mult, 0, 1)

            results.append(ServiceMetrics(service_name=name, metrics=df))

        context = {
            "event_type": event_type,
            "load_multiplier": load_multiplier,
            "spike_start": int(spike_start),
            "spike_end": int(spike_end),
            "ramp_length": int(ramp_len),
            "event_name": f"{event_type}_event",
        }
        return results, context
