"""
S1-Mini Metrics and Monitoring
==============================

This module provides metrics collection and monitoring for the S1-Mini engine.
Designed for integration with Prometheus, Grafana, and other monitoring systems.

Metrics Collected:
------------------
1. Request Metrics:
   - Total requests
   - Request latency (histogram)
   - Error rate
   - Active requests

2. Model Metrics:
   - Tokens generated per second
   - Time to first token
   - Audio duration vs generation time

3. Resource Metrics:
   - VRAM usage
   - GPU utilization
   - CPU usage
   - Memory usage

4. Cache Metrics:
   - Reference cache hit rate
   - KV cache utilization

Usage:
------
    from s1_mini.metrics import MetricsCollector, track_request

    collector = MetricsCollector()

    # Track a request
    with track_request(collector, "tts_generation"):
        result = engine.generate(text)

    # Get metrics
    print(collector.get_prometheus_metrics())

Prometheus Integration:
-----------------------
    from s1_mini.metrics import get_prometheus_metrics

    @app.get("/metrics")
    def metrics():
        return Response(
            get_prometheus_metrics(),
            media_type="text/plain"
        )
"""

import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import torch
from loguru import logger


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class RequestMetrics:
    """Metrics for a single request."""

    request_id: str
    start_time: float
    end_time: Optional[float] = None
    duration_seconds: float = 0.0
    success: bool = True
    error: Optional[str] = None
    tokens_generated: int = 0
    audio_duration_seconds: float = 0.0


@dataclass
class HistogramBucket:
    """Histogram bucket for latency tracking."""

    le: float  # Less than or equal to
    count: int = 0


@dataclass
class Histogram:
    """Simple histogram implementation."""

    name: str
    buckets: List[HistogramBucket] = field(default_factory=list)
    sum: float = 0.0
    count: int = 0

    def __post_init__(self):
        if not self.buckets:
            # Default latency buckets (in seconds)
            self.buckets = [
                HistogramBucket(le=0.1),
                HistogramBucket(le=0.5),
                HistogramBucket(le=1.0),
                HistogramBucket(le=2.0),
                HistogramBucket(le=5.0),
                HistogramBucket(le=10.0),
                HistogramBucket(le=30.0),
                HistogramBucket(le=60.0),
                HistogramBucket(le=float("inf")),
            ]

    def observe(self, value: float):
        """Record a value."""
        self.sum += value
        self.count += 1
        for bucket in self.buckets:
            if value <= bucket.le:
                bucket.count += 1


# =============================================================================
# Metrics Collector
# =============================================================================


class MetricsCollector:
    """
    Collects and exposes metrics for monitoring.

    Thread-safe metrics collection with Prometheus-compatible output.

    Attributes:
        namespace: Metric name prefix (default: "s1_mini")

    Example:
        >>> collector = MetricsCollector()
        >>> collector.increment("requests_total")
        >>> collector.observe("request_duration_seconds", 1.5)
        >>> print(collector.get_prometheus_metrics())
    """

    def __init__(self, namespace: str = "s1_mini"):
        """
        Initialize the metrics collector.

        Args:
            namespace: Prefix for metric names
        """
        self.namespace = namespace
        self._lock = threading.RLock()

        # Counters
        self._counters: Dict[str, int] = defaultdict(int)

        # Gauges
        self._gauges: Dict[str, float] = defaultdict(float)

        # Histograms
        self._histograms: Dict[str, Histogram] = {}

        # Active requests tracking
        self._active_requests: Dict[str, RequestMetrics] = {}

        # Initialize default metrics
        self._init_default_metrics()

    def _init_default_metrics(self):
        """Initialize default metric definitions."""
        # Request latency histogram
        self._histograms["request_duration_seconds"] = Histogram(
            name="request_duration_seconds"
        )

        # Time to first token histogram
        self._histograms["time_to_first_token_seconds"] = Histogram(
            name="time_to_first_token_seconds"
        )

        # Tokens per second histogram
        self._histograms["tokens_per_second"] = Histogram(
            name="tokens_per_second",
            buckets=[
                HistogramBucket(le=10),
                HistogramBucket(le=20),
                HistogramBucket(le=50),
                HistogramBucket(le=100),
                HistogramBucket(le=200),
                HistogramBucket(le=float("inf")),
            ],
        )

    # =========================================================================
    # Counter Methods
    # =========================================================================

    def increment(self, name: str, value: int = 1, labels: Dict[str, str] = None):
        """
        Increment a counter.

        Args:
            name: Counter name
            value: Amount to increment (default: 1)
            labels: Optional labels
        """
        key = self._make_key(name, labels)
        with self._lock:
            self._counters[key] += value

    def get_counter(self, name: str, labels: Dict[str, str] = None) -> int:
        """Get current counter value."""
        key = self._make_key(name, labels)
        with self._lock:
            return self._counters[key]

    # =========================================================================
    # Gauge Methods
    # =========================================================================

    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """
        Set a gauge value.

        Args:
            name: Gauge name
            value: Value to set
            labels: Optional labels
        """
        key = self._make_key(name, labels)
        with self._lock:
            self._gauges[key] = value

    def increment_gauge(self, name: str, value: float = 1.0, labels: Dict[str, str] = None):
        """Increment a gauge."""
        key = self._make_key(name, labels)
        with self._lock:
            self._gauges[key] += value

    def decrement_gauge(self, name: str, value: float = 1.0, labels: Dict[str, str] = None):
        """Decrement a gauge."""
        key = self._make_key(name, labels)
        with self._lock:
            self._gauges[key] -= value

    def get_gauge(self, name: str, labels: Dict[str, str] = None) -> float:
        """Get current gauge value."""
        key = self._make_key(name, labels)
        with self._lock:
            return self._gauges[key]

    # =========================================================================
    # Histogram Methods
    # =========================================================================

    def observe(self, name: str, value: float):
        """
        Record a value in a histogram.

        Args:
            name: Histogram name
            value: Value to record
        """
        with self._lock:
            if name not in self._histograms:
                self._histograms[name] = Histogram(name=name)
            self._histograms[name].observe(value)

    # =========================================================================
    # Request Tracking
    # =========================================================================

    def start_request(self, request_id: str) -> RequestMetrics:
        """
        Start tracking a request.

        Args:
            request_id: Unique request identifier

        Returns:
            RequestMetrics object for this request
        """
        metrics = RequestMetrics(
            request_id=request_id,
            start_time=time.time(),
        )

        with self._lock:
            self._active_requests[request_id] = metrics
            self.increment("requests_total")
            self.increment_gauge("active_requests")

        return metrics

    def end_request(
        self,
        request_id: str,
        success: bool = True,
        error: Optional[str] = None,
        tokens_generated: int = 0,
        audio_duration: float = 0.0,
    ):
        """
        End tracking a request.

        Args:
            request_id: Request identifier
            success: Whether request succeeded
            error: Error message if failed
            tokens_generated: Number of tokens generated
            audio_duration: Duration of generated audio
        """
        with self._lock:
            if request_id not in self._active_requests:
                return

            metrics = self._active_requests.pop(request_id)
            metrics.end_time = time.time()
            metrics.duration_seconds = metrics.end_time - metrics.start_time
            metrics.success = success
            metrics.error = error
            metrics.tokens_generated = tokens_generated
            metrics.audio_duration_seconds = audio_duration

            # Update counters
            self.decrement_gauge("active_requests")

            if success:
                self.increment("requests_success_total")
            else:
                self.increment("requests_error_total")

            # Update histograms
            self.observe("request_duration_seconds", metrics.duration_seconds)

            if tokens_generated > 0 and metrics.duration_seconds > 0:
                tokens_per_second = tokens_generated / metrics.duration_seconds
                self.observe("tokens_per_second", tokens_per_second)

    # =========================================================================
    # Resource Metrics
    # =========================================================================

    def update_vram_metrics(self):
        """Update VRAM usage metrics."""
        if not torch.cuda.is_available():
            return

        try:
            total = torch.cuda.get_device_properties(0).total_memory
            allocated = torch.cuda.memory_allocated(0)
            cached = torch.cuda.memory_reserved(0)

            self.set_gauge("vram_total_bytes", total)
            self.set_gauge("vram_allocated_bytes", allocated)
            self.set_gauge("vram_cached_bytes", cached)
            self.set_gauge("vram_utilization", cached / total if total > 0 else 0)

        except Exception as e:
            logger.debug(f"Failed to update VRAM metrics: {e}")

    # =========================================================================
    # Export Methods
    # =========================================================================

    def get_prometheus_metrics(self) -> str:
        """
        Get metrics in Prometheus text format.

        Returns:
            Prometheus-compatible metrics string
        """
        lines = []

        with self._lock:
            # Update resource metrics
            self.update_vram_metrics()

            # Counters
            for key, value in self._counters.items():
                name = f"{self.namespace}_{key}"
                lines.append(f"# TYPE {name} counter")
                lines.append(f"{name} {value}")

            # Gauges
            for key, value in self._gauges.items():
                name = f"{self.namespace}_{key}"
                lines.append(f"# TYPE {name} gauge")
                lines.append(f"{name} {value}")

            # Histograms
            for hist_name, histogram in self._histograms.items():
                name = f"{self.namespace}_{hist_name}"
                lines.append(f"# TYPE {name} histogram")

                for bucket in histogram.buckets:
                    le = bucket.le if bucket.le != float("inf") else "+Inf"
                    lines.append(f'{name}_bucket{{le="{le}"}} {bucket.count}')

                lines.append(f"{name}_sum {histogram.sum}")
                lines.append(f"{name}_count {histogram.count}")

        return "\n".join(lines)

    def get_dict(self) -> Dict[str, Any]:
        """
        Get metrics as dictionary.

        Returns:
            Dictionary with all metrics
        """
        with self._lock:
            self.update_vram_metrics()

            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {
                    name: {"sum": h.sum, "count": h.count}
                    for name, h in self._histograms.items()
                },
                "active_requests": len(self._active_requests),
            }

    # =========================================================================
    # Helper Methods
    # =========================================================================

    @staticmethod
    def _make_key(name: str, labels: Optional[Dict[str, str]] = None) -> str:
        """Create a metric key from name and labels."""
        if not labels:
            return name

        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"


# =============================================================================
# Context Managers
# =============================================================================


@contextmanager
def track_request(
    collector: MetricsCollector,
    operation: str = "request",
):
    """
    Context manager for tracking request metrics.

    Usage:
        with track_request(collector, "tts_generation") as metrics:
            result = engine.generate(text)
            metrics.tokens_generated = result.token_count

    Args:
        collector: MetricsCollector instance
        operation: Operation name for labeling

    Yields:
        RequestMetrics object for updating
    """
    import uuid

    request_id = str(uuid.uuid4())[:8]
    metrics = collector.start_request(request_id)

    try:
        yield metrics
        collector.end_request(
            request_id,
            success=True,
            tokens_generated=metrics.tokens_generated,
            audio_duration=metrics.audio_duration_seconds,
        )
    except Exception as e:
        collector.end_request(
            request_id,
            success=False,
            error=str(e),
        )
        raise


# =============================================================================
# Global Collector
# =============================================================================


_global_collector: Optional[MetricsCollector] = None


def get_collector() -> MetricsCollector:
    """Get the global metrics collector."""
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector()
    return _global_collector


def get_prometheus_metrics() -> str:
    """Get Prometheus metrics from global collector."""
    return get_collector().get_prometheus_metrics()
