from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import time
import json
import logging
import uuid
import os
from datetime import datetime

from fraud_inference import score_transaction, decide, explain, get_model_info
from graph_fraud import FraudGraphDetector
from rate_limiter import DualWindowRateLimiter

# ── Logging Setup ─────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler("logs/fraud_detection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("fraud_api")

def log_event(event: str, data: dict):
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "event":     event,
        **data
    }
    logger.info(json.dumps(entry))

# ── App Setup ─────────────────────────────────────────────────
app = FastAPI(
    title="AI Credit Card Fraud Detection API",
    description=(
        "Real-time fraud detection with LightGBM ML + "
        "Graph Ring Detection + Dual-Window Redis Rate Limiting"
    ),
    version="5.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Shared Instances ──────────────────────────────────────────
# Graph detector (Phase 8 — hardened with TTL + merchant rings)
graph_detector = FraudGraphDetector(edge_ttl=3600)
log_event("startup", {"graph_detector": "initialized"})

# Rate limiter (Phase 9 — dual window: 5/10s + 100/hr)
rate_limiter = DualWindowRateLimiter()
log_event("startup", {
    "rate_limiter": "dual_window",
    "short_window": "5/10s",
    "long_window":  "100/hr",
    "redis":        "connected" if rate_limiter._redis_available else "unavailable"
})

# ── Server Stats ──────────────────────────────────────────────
START_TIME = time.time()
stats = {
    "total_requests": 0,
    "fraud_detected": 0,
    "approved":       0,
    "step_up":        0,
    "blocked_ml":     0,
    "blocked_rate":   0,
    "graph_flagged":  0,
}

# ── Schemas ───────────────────────────────────────────────────
class TransactionRequest(BaseModel):
    # V1-V28 PCA features
    V1: float = 0.0;  V2: float = 0.0;  V3: float = 0.0
    V4: float = 0.0;  V5: float = 0.0;  V6: float = 0.0
    V7: float = 0.0;  V8: float = 0.0;  V9: float = 0.0
    V10: float = 0.0; V11: float = 0.0; V12: float = 0.0
    V13: float = 0.0; V14: float = 0.0; V15: float = 0.0
    V16: float = 0.0; V17: float = 0.0; V18: float = 0.0
    V19: float = 0.0; V20: float = 0.0; V21: float = 0.0
    V22: float = 0.0; V23: float = 0.0; V24: float = 0.0
    V25: float = 0.0; V26: float = 0.0; V27: float = 0.0
    V28: float = 0.0

    # Transaction features
    Amount: float        = 100.0
    tx_count_1min: int   = 1
    tx_count_10min: int  = 3
    tx_count_60min: int  = 10
    hour: int            = 12

    # Graph entity IDs
    card_id: str     = "unknown"
    merchant_id: str = "unknown"
    ip: str          = "0.0.0.0"


class FraudResponse(BaseModel):
    request_id:    str
    risk_score:    float
    ml_score:      float
    graph_score:   float
    decision:      str
    explanation:   list[str]
    graph_signals: list[str]
    rate_limiter:  str
    latency_ms:    float


# ── Endpoints ─────────────────────────────────────────────────

@app.get("/health")
def health():
    uptime     = int(time.time() - START_TIME)
    graph_stats = graph_detector.get_stats()
    model_info  = get_model_info()
    return {
        "status":          "ok",
        "version":         "5.0.0",
        "uptime":          f"{uptime//3600}h {(uptime%3600)//60}m {uptime%60}s",
        "redis":           "connected" if rate_limiter._redis_available else "unavailable",
        "model":           model_info["model_type"],
        "threshold":       model_info["threshold"],
        "graph_nodes":     graph_stats["active_nodes"],
        "graph_edges":     graph_stats["active_edges"],
        "total_requests":  stats["total_requests"],
        "fraud_detected":  stats["fraud_detected"],
        "approved":        stats["approved"],
        "blocked_by_rate": stats["blocked_rate"],
    }


@app.get("/metrics")
def metrics():
    total = max(stats["total_requests"], 1)
    return {
        "total_requests":      stats["total_requests"],
        "fraud_rate_pct":      round(stats["fraud_detected"] / total * 100, 2),
        "graph_flag_rate_pct": round(stats["graph_flagged"]  / total * 100, 2),
        "approval_rate_pct":   round(stats["approved"]       / total * 100, 2),
        "rate_block_rate_pct": round(stats["blocked_rate"]   / total * 100, 2),
        "graph_stats":         graph_detector.get_stats(),
        "rate_limiter_stats":  rate_limiter.get_status("global"),
        "redis_backend":       rate_limiter._redis_available,
    }


@app.get("/graph/rings")
def fraud_rings():
    rings = graph_detector.detect_rings()
    return {
        "total_rings": len(rings),
        "rings":       rings
    }


@app.get("/rate-limiter/status")
def rate_limiter_status(request: Request):
    client_ip = request.client.host
    return rate_limiter.get_status(client_ip)


@app.post("/fraud/check", response_model=FraudResponse)
def check_fraud(tx: TransactionRequest, request: Request):
    request_id = str(uuid.uuid4())[:8]
    client_ip  = request.client.host
    start_time = time.time()

    stats["total_requests"] += 1

    # ── Phase 9: Dual-window rate limiter ─────────────────────
    allowed, reason = rate_limiter.is_allowed(client_ip)
    if not allowed:
        stats["blocked_rate"] += 1
        log_event("rate_limited", {
            "request_id": request_id,
            "client_ip":  client_ip,
            "reason":     reason
        })
        raise HTTPException(status_code=429, detail=reason)

    # ── Build feature vector (all 39 features) ────────────────
    features = {
        **{f"V{i}": getattr(tx, f"V{i}") for i in range(1, 29)},
        "Amount":                 tx.Amount,
        "amount_log":             np.log1p(tx.Amount),
        "amount_sqrt":            np.sqrt(tx.Amount),
        "tx_count_1min":          tx.tx_count_1min,
        "tx_count_10min":         tx.tx_count_10min,
        "tx_count_60min":         tx.tx_count_60min,
        "amount_rolling_mean_1h": tx.Amount,
        "amount_rolling_std_1h":  0.0,
        "amount_deviation":       0.0,
        "hour":                   tx.hour,
        "is_night":               1 if tx.hour < 5 else 0,
    }

    # ── ML score ──────────────────────────────────────────────
    ml_score = score_transaction(features)

    # ── Phase 8: Graph score (hardened — TTL + merchant rings) ─
    graph_score, graph_signals = graph_detector.score_transaction(
        card_id=tx.card_id,
        merchant_id=tx.merchant_id,
        ip_address=tx.ip
    )

    # Add to graph for future transactions
    graph_detector.add_transaction(
        card_id=tx.card_id,
        merchant_id=tx.merchant_id,
        ip_address=tx.ip,
        is_fraud=(ml_score > 0.5)
    )

    # ── Combined score ────────────────────────────────────────
    combined_score = round(0.6 * ml_score + 0.4 * graph_score, 4)

    decision = decide(combined_score)
    reasons  = explain(features, combined_score)
    latency  = round((time.time() - start_time) * 1000, 2)

    # ── Update stats ──────────────────────────────────────────
    if decision == "APPROVE":
        stats["approved"]       += 1
    elif decision == "STEP_UP_AUTH":
        stats["step_up"]        += 1
    else:
        stats["blocked_ml"]     += 1
        stats["fraud_detected"] += 1

    if graph_score > 0.3:
        stats["graph_flagged"] += 1

    # ── Structured log ────────────────────────────────────────
    log_event("transaction_scored", {
        "request_id":  request_id,
        "client_ip":   client_ip,
        "card_id":     tx.card_id,
        "amount":      tx.Amount,
        "ml_score":    round(ml_score, 4),
        "graph_score": round(graph_score, 4),
        "combined":    combined_score,
        "decision":    decision,
        "latency_ms":  latency
    })

    return FraudResponse(
        request_id=request_id,
        risk_score=combined_score,
        ml_score=round(ml_score, 4),
        graph_score=round(graph_score, 4),
        decision=decision,
        explanation=reasons,
        graph_signals=graph_signals,
        rate_limiter="redis" if rate_limiter._redis_available else "in-memory",
        latency_ms=latency
    )
