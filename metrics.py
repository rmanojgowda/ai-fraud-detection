"""
Prometheus Metrics Module
==========================
Exposes real-time system metrics for Grafana dashboards.

Metrics tracked:
  - fraud_requests_total        (counter)
  - fraud_decisions_total       (counter by decision type)
  - fraud_score_histogram       (histogram of ML scores)
  - fraud_latency_histogram     (histogram of latency)
  - rate_limit_blocks_total     (counter by block type)
  - graph_rings_detected_total  (counter)
  - redis_available             (gauge)
  - model_threshold             (gauge)
"""

from prometheus_client import (
    Counter, Histogram, Gauge, Summary,
    CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
)

# ── Registry ──────────────────────────────────────────────────
# Use default registry
registry = CollectorRegistry(auto_describe=True)

# ── Counters (ever-increasing) ────────────────────────────────

# Total requests processed
REQUESTS_TOTAL = Counter(
    "fraud_requests_total",
    "Total number of fraud check requests",
    ["status"],  # labels: success, rate_limited, error
)

# Decisions made
DECISIONS_TOTAL = Counter(
    "fraud_decisions_total",
    "Total decisions made by the fraud system",
    ["decision"],  # labels: APPROVE, STEP_UP_AUTH, BLOCK
)

# Rate limit blocks
RATE_LIMIT_BLOCKS = Counter(
    "rate_limit_blocks_total",
    "Total rate limit blocks",
    ["block_type"],  # labels: short_window, long_window, card_limit
)

# Graph rings detected
GRAPH_RINGS_TOTAL = Counter(
    "graph_rings_detected_total",
    "Total fraud rings detected by graph layer",
)

# ── Histograms (distribution tracking) ───────────────────────

# ML inference latency
LATENCY_HISTOGRAM = Histogram(
    "fraud_request_latency_seconds",
    "Request latency in seconds",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.075,
             0.1, 0.25, 0.5, 0.75, 1.0, 2.5]
)

# ML risk scores
SCORE_HISTOGRAM = Histogram(
    "fraud_ml_score",
    "Distribution of ML fraud scores",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
             0.6, 0.7, 0.8, 0.9, 1.0]
)

# ── Gauges (current value) ────────────────────────────────────

# Redis availability
REDIS_AVAILABLE = Gauge(
    "redis_available",
    "Whether Redis is currently available (1=yes, 0=no)"
)

# Model threshold
MODEL_THRESHOLD = Gauge(
    "fraud_model_threshold",
    "Current fraud detection threshold"
)

# Active graph nodes
GRAPH_NODES = Gauge(
    "graph_active_nodes",
    "Number of active nodes in fraud graph"
)

# Active graph edges
GRAPH_EDGES = Gauge(
    "graph_active_edges",
    "Number of active edges in fraud graph"
)

# Current fraud rate (rolling)
FRAUD_RATE = Gauge(
    "fraud_rate_percent",
    "Current fraud detection rate as percentage"
)

# Request rate (per second)
REQUESTS_PER_SECOND = Gauge(
    "fraud_requests_per_second",
    "Current request rate per second"
)


# ── Helper Functions ──────────────────────────────────────────

def record_request(decision: str, latency_s: float,
                   ml_score: float, status: str = "success"):
    """Record a completed fraud check request."""
    REQUESTS_TOTAL.labels(status=status).inc()
    DECISIONS_TOTAL.labels(decision=decision).inc()
    LATENCY_HISTOGRAM.observe(latency_s)
    SCORE_HISTOGRAM.observe(min(ml_score, 1.0))


def record_rate_limit(block_type: str):
    """Record a rate limit block."""
    REQUESTS_TOTAL.labels(status="rate_limited").inc()
    RATE_LIMIT_BLOCKS.labels(block_type=block_type).inc()


def record_ring_detected():
    """Record a fraud ring detection."""
    GRAPH_RINGS_TOTAL.inc()


def update_system_state(
    redis_up: bool,
    threshold: float,
    graph_nodes: int,
    graph_edges: int,
    fraud_rate: float
):
    """Update gauge metrics with current system state."""
    REDIS_AVAILABLE.set(1 if redis_up else 0)
    MODEL_THRESHOLD.set(threshold)
    GRAPH_NODES.set(graph_nodes)
    GRAPH_EDGES.set(graph_edges)
    FRAUD_RATE.set(fraud_rate)


def get_metrics_output():
    """Returns Prometheus-formatted metrics string."""
    from prometheus_client import generate_latest, REGISTRY
    return generate_latest(REGISTRY)


# ── Quick Test ────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing metrics module...")

    # Simulate some requests
    record_request("APPROVE",      0.010, 0.05)
    record_request("APPROVE",      0.012, 0.08)
    record_request("STEP_UP_AUTH", 0.015, 0.45)
    record_request("BLOCK",        0.011, 0.92)
    record_rate_limit("short_window")
    record_rate_limit("card_limit")
    record_ring_detected()
    update_system_state(
        redis_up=True,
        threshold=0.7722,
        graph_nodes=15,
        graph_edges=22,
        fraud_rate=12.5
    )

    output = get_metrics_output().decode("utf-8")

    # Show key metrics
    for line in output.split("\n"):
        if line and not line.startswith("#"):
            if any(k in line for k in [
                "fraud_requests", "fraud_decisions",
                "rate_limit", "redis_available",
                "fraud_model_threshold", "graph_active"
            ]):
                print(f"  {line}")

    print("\n✅ Metrics module working!")
    print("   Expose via GET /metrics endpoint in FastAPI")
