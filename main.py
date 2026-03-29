from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import time
import redis
import json
import logging
import uuid
import os
from datetime import datetime

from fraud_inference import score_transaction, decide, explain
from graph_fraud import FraudGraphDetector

# ── Logging Setup ─────────────────────────────────
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

# ── App Setup ─────────────────────────────────────
app = FastAPI(
    title="AI Credit Card Fraud Detection API",
    description="Real-time fraud detection with ML + Graph + Redis rate limiting",
    version="4.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Graph Detector (shared instance) ─────────────
graph_detector = FraudGraphDetector()
log_event("startup", {"graph_detector": "initialized"})

# ── Redis Connection ──────────────────────────────
try:
    redis_client    = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
    redis_client.ping()
    REDIS_AVAILABLE = True
    log_event("startup", {"redis": "connected", "port": 6379})
except Exception:
    REDIS_AVAILABLE = False
    log_event("startup", {"redis": "unavailable", "fallback": "in-memory"})

# ── In-Memory Fallback Rate Limiter ───────────────
client_requests = {}
RATE_LIMIT      = 5
WINDOW_SECONDS  = 10

# ── Server Stats ──────────────────────────────────
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

def rate_limiter_memory(client_ip: str) -> bool:
    now        = time.time()
    timestamps = client_requests.get(client_ip, [])
    timestamps = [t for t in timestamps if now - t < WINDOW_SECONDS]
    if len(timestamps) >= RATE_LIMIT:
        return False
    timestamps.append(now)
    client_requests[client_ip] = timestamps
    return True

def rate_limiter_redis(client_ip: str) -> bool:
    key  = f"rate:{client_ip}"
    now  = time.time()
    pipe = redis_client.pipeline()
    pipe.zremrangebyscore(key, 0, now - WINDOW_SECONDS)
    pipe.zcard(key)
    pipe.zadd(key, {str(now): now})
    pipe.expire(key, WINDOW_SECONDS)
    results = pipe.execute()
    return results[1] < RATE_LIMIT

def check_rate_limit(client_ip: str) -> bool:
    if REDIS_AVAILABLE:
        return rate_limiter_redis(client_ip)
    return rate_limiter_memory(client_ip)

# ── Schemas ───────────────────────────────────────
class TransactionRequest(BaseModel):
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
    Amount: float        = 100.0
    tx_count_1min: int   = 1
    tx_count_10min: int  = 3
    tx_count_60min: int  = 10
    hour: int            = 12
    # Graph fields (optional)
    card_id: str     = "unknown"
    merchant_id: str = "unknown"
    ip: str          = "0.0.0.0"

class FraudResponse(BaseModel):
    request_id:       str
    risk_score:       float
    ml_score:         float
    graph_score:      float
    decision:         str
    explanation:      list[str]
    graph_signals:    list[str]
    rate_limiter:     str
    latency_ms:       float

# ── Health ────────────────────────────────────────
@app.get("/health")
def health():
    uptime = int(time.time() - START_TIME)
    return {
        "status":          "ok",
        "version":         "4.0.0",
        "uptime":          f"{uptime//3600}h {(uptime%3600)//60}m {uptime%60}s",
        "redis":           "connected" if REDIS_AVAILABLE else "unavailable",
        "graph_nodes":     graph_detector.graph.number_of_nodes(),
        "graph_edges":     graph_detector.graph.number_of_edges(),
        "total_requests":  stats["total_requests"],
        "fraud_detected":  stats["fraud_detected"],
        "graph_flagged":   stats["graph_flagged"],
        "approved":        stats["approved"],
        "blocked_by_rate": stats["blocked_rate"],
    }

# ── Metrics ───────────────────────────────────────
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
        "redis_backend":       REDIS_AVAILABLE,
    }

# ── Graph Rings Endpoint ──────────────────────────
@app.get("/graph/rings")
def fraud_rings():
    rings = graph_detector.detect_fraud_rings()
    return {
        "total_rings": len(rings),
        "rings":       rings
    }

# ── Rate Limiter Status ───────────────────────────
@app.get("/rate-limiter/status")
def rate_limiter_status(request: Request):
    client_ip = request.client.host
    if REDIS_AVAILABLE:
        key = f"rate:{client_ip}"
        now = time.time()
        redis_client.zremrangebyscore(key, 0, now - WINDOW_SECONDS)
        count = redis_client.zcard(key)
    else:
        now        = time.time()
        timestamps = client_requests.get(client_ip, [])
        count      = len([t for t in timestamps if now - t < WINDOW_SECONDS])
    return {
        "client_ip":          client_ip,
        "requests_in_window": count,
        "limit":              RATE_LIMIT,
        "remaining":          max(0, RATE_LIMIT - count),
        "backend":            "redis" if REDIS_AVAILABLE else "in-memory"
    }

# ── Fraud Check ───────────────────────────────────
@app.post("/fraud/check", response_model=FraudResponse)
def check_fraud(tx: TransactionRequest, request: Request):
    request_id = str(uuid.uuid4())[:8]
    client_ip  = request.client.host
    start_time = time.time()

    stats["total_requests"] += 1

    # Rate limit
    if not check_rate_limit(client_ip):
        stats["blocked_rate"] += 1
        log_event("rate_limited", {"request_id": request_id, "client_ip": client_ip})
        raise HTTPException(status_code=429, detail="Rate limit exceeded.")

    # ML features
    features = {
        **{f"V{i}": getattr(tx, f"V{i}") for i in range(1, 29)},
        "Amount":                tx.Amount,
        "amount_log":            np.log1p(tx.Amount),
        "tx_count_1min":         tx.tx_count_1min,
        "tx_count_10min":        tx.tx_count_10min,
        "tx_count_60min":        tx.tx_count_60min,
        "amount_rolling_mean_1h": tx.Amount,
        "hour":                  tx.hour,
        "is_night":              1 if tx.hour < 5 else 0
    }

    # ML score
    ml_score = score_transaction(features)

    # Graph score
    graph_result = graph_detector.get_graph_risk_score(
        card_id=tx.card_id,
        merchant_id=tx.merchant_id,
        ip=tx.ip
    )
    graph_score = graph_result["graph_risk_score"]

    # Add transaction to graph for future lookups
    graph_detector.add_transaction({
        "card_id":     tx.card_id,
        "merchant_id": tx.merchant_id,
        "ip":          tx.ip,
        "amount":      tx.Amount,
        "is_fraud":    1 if ml_score > 0.5 else 0
    })

    # Combined score (ML + Graph ensemble)
    combined_score = round(0.6 * ml_score + 0.4 * graph_score, 4)

    decision = decide(combined_score)
    reasons  = explain(features, combined_score)
    latency  = round((time.time() - start_time) * 1000, 2)

    # Update stats
    if decision == "APPROVE":
        stats["approved"]       += 1
    elif decision == "STEP_UP_AUTH":
        stats["step_up"]        += 1
    else:
        stats["blocked_ml"]     += 1
        stats["fraud_detected"] += 1

    if graph_score > 0.3:
        stats["graph_flagged"] += 1

    log_event("transaction_scored", {
        "request_id":    request_id,
        "client_ip":     client_ip,
        "card_id":       tx.card_id,
        "amount":        tx.Amount,
        "ml_score":      round(ml_score, 4),
        "graph_score":   round(graph_score, 4),
        "combined":      combined_score,
        "decision":      decision,
        "latency_ms":    latency
    })

    return FraudResponse(
        request_id=request_id,
        risk_score=combined_score,
        ml_score=round(ml_score, 4),
        graph_score=round(graph_score, 4),
        decision=decision,
        explanation=reasons,
        graph_signals=graph_result["graph_signals"],
        rate_limiter="redis" if REDIS_AVAILABLE else "in-memory",
        latency_ms=latency
    )
