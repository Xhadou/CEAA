"""Extract context features from external event information.

This is the NOVEL component — integrating business context
(event calendars, deployments, time-of-day) into anomaly classification.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional


@dataclass
class ContextFeatures:
    """Container for context-related features."""

    event_active: float
    event_expected_impact: float
    time_seasonality: float
    recent_deployment: float
    context_confidence: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "event_active": self.event_active,
            "event_expected_impact": self.event_expected_impact,
            "time_seasonality": self.time_seasonality,
            "recent_deployment": self.recent_deployment,
            "context_confidence": self.context_confidence,
        }


class EventCalendar:
    """Manages business events that could cause expected load spikes.

    In a production system this would integrate with marketing-calendar
    APIs, CI/CD deployment pipelines, and incident-management systems.
    """

    def __init__(self) -> None:
        self.events: List[Dict] = []

    def add_event(
        self,
        name: str,
        event_type: str,
        start_time: datetime,
        end_time: datetime,
        expected_impact: float,
        affected_services: Optional[List[str]] = None,
    ) -> None:
        self.events.append({
            "name": name,
            "type": event_type,
            "start": start_time,
            "end": end_time,
            "impact": expected_impact,
            "services": affected_services or [],
        })

    def get_active_events(self, timestamp: datetime) -> List[Dict]:
        return [e for e in self.events if e["start"] <= timestamp <= e["end"]]

    def get_max_expected_impact(self, timestamp: datetime) -> float:
        active = self.get_active_events(timestamp)
        return max((e["impact"] for e in active), default=1.0)

    @classmethod
    def create_synthetic_calendar(cls, base_time: Optional[datetime] = None) -> "EventCalendar":
        """Create a synthetic event calendar for testing."""
        cal = cls()
        base = base_time or datetime.now()
        for name, etype, h_start, h_end, impact in [
            ("Flash Sale", "sale", 0, 2, 3.5),
            ("Marketing Campaign", "marketing", 4, 8, 2.0),
            ("Batch Processing", "batch", 12, 14, 1.8),
            ("Holiday Traffic", "seasonal", 24, 72, 2.5),
        ]:
            cal.add_event(
                name=name,
                event_type=etype,
                start_time=base + timedelta(hours=h_start),
                end_time=base + timedelta(hours=h_end),
                expected_impact=impact,
            )
        return cal


class ContextFeatureExtractor:
    """Extract features based on external context."""

    HIGH_TRAFFIC_HOURS = list(range(9, 21))  # 9 AM – 9 PM

    def __init__(self, event_calendar: Optional[EventCalendar] = None) -> None:
        self.calendar = event_calendar or EventCalendar()

    def extract_event_active(
        self, timestamp: Optional[datetime] = None, context: Optional[Dict] = None,
    ) -> float:
        if context and "event_type" in context:
            return 1.0
        if timestamp:
            return 1.0 if self.calendar.get_active_events(timestamp) else 0.0
        return 0.0

    def extract_event_expected_impact(
        self, timestamp: Optional[datetime] = None, context: Optional[Dict] = None,
    ) -> float:
        if context and "load_multiplier" in context:
            return min(1.0, float(context["load_multiplier"]) / 5.0)
        if timestamp:
            return min(1.0, self.calendar.get_max_expected_impact(timestamp) / 5.0)
        return 0.0

    def extract_time_seasonality(self, timestamp: Optional[datetime] = None) -> float:
        if timestamp is None:
            return 0.5
        hour = timestamp.hour
        weekend = 0.8 if timestamp.weekday() >= 5 else 1.0
        hour_factor = 1.0 if hour in self.HIGH_TRAFFIC_HOURS else 0.3
        return hour_factor * weekend

    def extract_recent_deployment(
        self,
        timestamp: Optional[datetime] = None,
        deployment_history: Optional[List[datetime]] = None,
        window_minutes: int = 30,
    ) -> float:
        if timestamp is None or not deployment_history:
            return 0.0
        window = timedelta(minutes=window_minutes).total_seconds()
        return 1.0 if any(
            abs((timestamp - d).total_seconds()) < window for d in deployment_history
        ) else 0.0

    def extract_context_confidence(
        self, context: Optional[Dict] = None, has_calendar: bool = True,
    ) -> float:
        confidence = 0.0
        if context:
            if "event_type" in context:
                confidence += 0.3
            if "load_multiplier" in context:
                confidence += 0.2
            if "event_name" in context:
                confidence += 0.1
            if "expected_error_rate" in context:
                confidence += 0.1
        if has_calendar and self.calendar.events:
            confidence += 0.3
        return min(1.0, confidence)

    def extract_all(
        self,
        timestamp: Optional[datetime] = None,
        context: Optional[Dict] = None,
        deployment_history: Optional[List[datetime]] = None,
    ) -> ContextFeatures:
        """Extract all context features."""
        return ContextFeatures(
            event_active=self.extract_event_active(timestamp, context),
            event_expected_impact=self.extract_event_expected_impact(timestamp, context),
            time_seasonality=self.extract_time_seasonality(timestamp),
            recent_deployment=self.extract_recent_deployment(timestamp, deployment_history),
            context_confidence=self.extract_context_confidence(context),
        )


if __name__ == "__main__":
    cal = EventCalendar.create_synthetic_calendar(datetime.now())
    ext = ContextFeatureExtractor(cal)

    print("Current context:", ext.extract_all(timestamp=datetime.now()).to_dict())

    ctx = {"event_type": "flash_sale", "load_multiplier": 3.5, "event_name": "Black Friday"}
    print("Event context:", ext.extract_all(context=ctx).to_dict())
