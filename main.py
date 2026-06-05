from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from explainability import explain_transaction as shap_explain, format_explanation
from ab_testing import start_experiment, stop_experiment, get_experiment
from fraud_heatmap import record_heatmap, get_heatmap, get_dynamic_threshold
from transaction_replay import save_for_replay, get_replay_system
from geo_risk import score_geo_risk, get_geo_scorer
from geo_risk import score_geo_risk, get_geo_scorer\
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
from metrics import (
    record_request, record_rate_limit, record_ring_detected,
    update_system_state, get_metrics_output,
    CONTENT_TYPE_LATEST
)

# ── Logging ───────────────────────────────────────────────────
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
    entry = {"timestamp": datetime.utcnow().isoformat(),
             "event": event, **data}
    logger.info(json.dumps(entry))

# ── App ───────────────────────────────────────────────────────
app = FastAPI(
    title="AI Credit Card Fraud Detection API",
    description=(
        "Real-time fraud detection with LightGBM ML + "
        "Graph Ring Detection + Dual-Window + Card-Level Rate Limiting + "
        "Prometheus Metrics"
    ),
    version="6.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Shared Instances ──────────────────────────────────────────
graph_detector = FraudGraphDetector(edge_ttl=3600)
rate_limiter   = DualWindowRateLimiter()
model_info     = get_model_info()

log_event("startup", {"version": "6.0.0",
                       "model": model_info["model_type"],
                       "threshold": model_info["threshold"]})
log_event("startup", {
    "rate_limiter": "triple_layer",
    "layers": "5/10s + 100/hr + 3/hr/card",
    "redis": "connected" if rate_limiter._redis_available else "unavailable"
})
log_event("startup", {"prometheus": "enabled", "metrics_path": "/metrics"})

# ── Update Prometheus model threshold ────────────────────────
from metrics import MODEL_THRESHOLD
MODEL_THRESHOLD.set(model_info["threshold"])

# ── Stats ─────────────────────────────────────────────────────
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

# ── Request rate tracking ─────────────────────────────────────
_request_times = []
_request_lock  = __import__('threading').Lock()

def _track_request_rate():
    now = time.time()
    with _request_lock:
        _request_times.append(now)
        # Keep only last 60 seconds
        cutoff = now - 60
        while _request_times and _request_times[0] < cutoff:
            _request_times.pop(0)
        rps = len(_request_times) / 60.0
    from metrics import REQUESTS_PER_SECOND
    REQUESTS_PER_SECOND.set(rps)

# ── Schemas ───────────────────────────────────────────────────
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
    card_id: str         = "unknown"
    merchant_id: str     = "unknown"
    ip: str              = "0.0.0.0"
    country: str         = "IN"
    city: str            = "unknown"


class FraudResponse(BaseModel):
    request_id:    str
    risk_score:    float
    ml_score:      float
    graph_score:   float
    geo_score:     float
    decision:      str
    explanation:   list[str]
    graph_signals: list[str]
    geo_signals:   list[str]
    rate_limiter:  str
    latency_ms:    float


# ── Endpoints ─────────────────────────────────────────────────

@app.get("/health")
def health():
    uptime      = int(time.time() - START_TIME)
    graph_stats = graph_detector.get_stats()
    redis_up    = rate_limiter._redis_available
    total       = max(stats["total_requests"], 1)
    fraud_rate  = stats["fraud_detected"] / total * 100

    # Update Prometheus gauges
    update_system_state(
        redis_up    = redis_up,
        threshold   = model_info["threshold"],
        graph_nodes = graph_stats["active_nodes"],
        graph_edges = graph_stats["active_edges"],
        fraud_rate  = fraud_rate
    )

    return {
        "status":          "ok",
        "version":         "6.0.0",
        "uptime":          f"{uptime//3600}h {(uptime%3600)//60}m {uptime%60}s",
        "redis":           "connected" if redis_up else "unavailable",
        "model":           model_info["model_type"],
        "threshold":       model_info["threshold"],
        "graph_nodes":     graph_stats["active_nodes"],
        "graph_edges":     graph_stats["active_edges"],
        "total_requests":  stats["total_requests"],
        "fraud_detected":  stats["fraud_detected"],
        "fraud_rate_pct":  round(fraud_rate, 2),
        "approved":        stats["approved"],
        "blocked_by_rate": stats["blocked_rate"],
        "prometheus":      "enabled — GET /metrics",
    }


@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint — scraped every 15s by Prometheus."""
    # Update gauges before serving
    graph_stats = graph_detector.get_stats()
    total       = max(stats["total_requests"], 1)
    update_system_state(
        redis_up    = rate_limiter._redis_available,
        threshold   = model_info["threshold"],
        graph_nodes = graph_stats["active_nodes"],
        graph_edges = graph_stats["active_edges"],
        fraud_rate  = stats["fraud_detected"] / total * 100
    )
    return Response(
        content=get_metrics_output(),
        media_type="text/plain; version=0.0.4; charset=utf-8"
    )


@app.get("/stats")
def api_stats():
    total = max(stats["total_requests"], 1)
    return {
        "total_requests":      stats["total_requests"],
        "fraud_rate_pct":      round(stats["fraud_detected"] / total * 100, 2),
        "approval_rate_pct":   round(stats["approved"]       / total * 100, 2),
        "rate_block_rate_pct": round(stats["blocked_rate"]   / total * 100, 2),
        "graph_stats":         graph_detector.get_stats(),
        "rate_limiter_stats":  rate_limiter.get_status("global"),
        "redis_backend":       rate_limiter._redis_available,
    }


@app.get("/graph/rings")
def fraud_rings():
    rings = graph_detector.detect_rings()
    return {"total_rings": len(rings), "rings": rings}


@app.get("/rate-limiter/status")
def rate_limiter_status(request: Request):
    client_ip = request.client.host
    return rate_limiter.get_status(client_ip)


@app.post("/ab-test/start")
def ab_test_start(split_pct: float = 50.0, min_requests: int = 100):
    exp = start_experiment(split_pct=split_pct, min_requests=min_requests)
    return {"message": f"Experiment {exp.experiment_id} started",
            "split": f"{100-split_pct:.0f}% A / {split_pct:.0f}% B"}

@app.get("/ab-test/results")
def ab_test_results():
    exp = get_experiment()
    if not exp:
        return {"message": "No active experiment"}
    return exp.get_results()

@app.post("/ab-test/stop")
def ab_test_stop():
    results = stop_experiment()
    if not results:
        return {"message": "No active experiment to stop"}
    return results


@app.get("/heatmap")
def fraud_heatmap():
    return get_heatmap().get_heatmap_data()

@app.get("/geo/stats")
def geo_stats():
    return get_geo_scorer().get_stats()

@app.get("/replay/stats")
def replay_stats():
    return get_replay_system().get_stats()

@app.post("/replay/run")
def replay_run(limit: int = 100):
    import json
    with open("models/feature_cols.json") as f:
        cols = json.load(f)
    results = get_replay_system().replay_against_model(
        model_path="models/fraud_model.pkl",
        feature_cols=cols,
        threshold=0.7722,
        limit=limit
    )
    return results

@app.post("/fraud/check", response_model=FraudResponse)
def check_fraud(tx: TransactionRequest, request: Request):
    request_id = str(uuid.uuid4())[:8]
    client_ip  = request.client.host
    start_time = time.time()

    stats["total_requests"] += 1
    _track_request_rate()

    # ── Triple-layer rate limiter ─────────────────────────────
    allowed, reason = rate_limiter.is_allowed(client_ip, tx.card_id)
    if not allowed:
        stats["blocked_rate"] += 1

        # Determine block type for Prometheus
        if "Card limit" in reason:
            block_type = "card_limit"
        elif "Hourly" in reason:
            block_type = "long_window"
        else:
            block_type = "short_window"

        record_rate_limit(block_type)
        log_event("rate_limited", {
            "request_id": request_id,
            "client_ip":  client_ip,
            "card_id":    tx.card_id,
            "reason":     reason
        })
        raise HTTPException(status_code=429, detail=reason)

    # ── Build 39 features ────────────────────────────────────
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

    # ── ML score ─────────────────────────────────────────────
    ml_score = score_transaction(features)

    # ── Graph score ───────────────────────────────────────────
    graph_score, graph_signals = graph_detector.score_transaction(
        card_id=tx.card_id,
        merchant_id=tx.merchant_id,
        ip_address=tx.ip
    )
    graph_detector.add_transaction(
        card_id=tx.card_id,
        merchant_id=tx.merchant_id,
        ip_address=tx.ip,
        is_fraud=(ml_score > 0.5)
    )

    # Check for new rings
    rings = graph_detector.detect_rings()
    if rings:
        record_ring_detected()

    # ── Geographic risk score ─────────────────────────────────
    geo_score_country, geo_signals_country = score_geo_risk(
        card_id      = tx.card_id,
        country_code = tx.country,
        city         = tx.city if tx.city != "unknown" else None,
        ip_address   = tx.ip
    )
    geo_score_vfeat, geo_signals_vfeat = score_geo_risk_from_vfeatures(features)

    # Take the higher of the two scores
    if geo_score_vfeat > geo_score_country:
        geo_score   = geo_score_vfeat
        geo_signals = geo_signals_vfeat
    else:
        geo_score   = geo_score_country
        geo_signals = geo_signals_country

    # ── Combined decision ─────────────────────────────────────
    combined_score = round(0.5 * ml_score + 0.3 * graph_score + 0.2 * geo_score, 4)
    decision       = decide(combined_score)
    reasons        = explain(features, combined_score)
    shap_result    = shap_explain(features, top_n=5)
    shap_reasons   = format_explanation(shap_result, decision)
    latency        = round((time.time() - start_time) * 1000, 2)

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

    # ── Record Prometheus metrics ─────────────────────────────
    record_request(
        decision  = decision,
        latency_s = latency / 1000,
        ml_score  = ml_score,
        status    = "success"
    )

    # ── A/B test recording ────────────────────────────────────
    exp = get_experiment()
    if exp and exp.active:
        group = exp.assign_group(tx.card_id)
        exp.record_result(tx.card_id, group, decision, latency, combined_score)
        
    # ── Heatmap recording ─────────────────────────────────────
    record_heatmap(tx.hour, decision, combined_score)

    # ── Save for replay ───────────────────────────────────────
    save_for_replay(
        request_id  = request_id,
        features    = features,
        card_id     = tx.card_id,
        decision    = decision,
        score       = combined_score,
        graph_score = graph_score,
        amount      = tx.Amount,
        hour        = tx.hour,
    )
    # ── Log ───────────────────────────────────────────────────
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
        geo_score=round(geo_score, 4),
        decision=decision,
        explanation=shap_reasons,
        graph_signals=graph_signals,
        geo_signals=geo_signals,
        rate_limiter="redis" if rate_limiter._redis_available else "in-memory",
        latency_ms=latency
    )
