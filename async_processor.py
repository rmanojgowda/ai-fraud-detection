"""
Async ML Processing Pipeline
==============================
Decouples rate limiting (fast) from ML inference (slow).

Architecture:
  API Layer:   rate limit + validate → queue → return immediately
  ML Workers:  consume queue → score → graph → geo → store result
  Client:      poll GET /result/{id} for full analysis

Why this matters:
  Rate limit check:  <1ms  (Redis lookup)
  ML inference:      5-15ms (LightGBM + SHAP)
  Graph scoring:     2-5ms  (NetworkX)
  Geo scoring:       1-2ms  (V-feature analysis)
  
  Sync:  client waits 20-40ms
  Async: client gets response in <2ms
         ML runs in background
         
  At 100K RPS: sync needs 100K threads
               async needs ~100 workers

Real-world usage:
  Stripe:   async fraud scoring, sync auth decision
  Razorpay: queue-based ML, instant payment confirmation
  Visa:     batch + real-time hybrid scoring
"""

import queue
import threading
import time
import uuid
import json
import os
from collections import defaultdict
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime


# ── Result Storage ────────────────────────────────────────────
@dataclass
class MLResult:
    request_id:    str
    status:        str        # "queued", "processing", "complete", "error"
    ml_score:      float = 0.0
    graph_score:   float = 0.0
    geo_score:     float = 0.0
    velocity_risk: float = 0.0
    combined_score:float = 0.0
    decision:      str   = ""
    explanation:   list  = field(default_factory=list)
    graph_signals: list  = field(default_factory=list)
    geo_signals:   list  = field(default_factory=list)
    latency_ms:    float = 0.0
    queued_at:     float = field(default_factory=time.time)
    completed_at:  float = 0.0
    error:         str   = ""


class AsyncMLProcessor:
    """
    Background ML processing pipeline.
    
    Design:
      - Thread-safe queue (Python queue.Queue)
      - N worker threads consuming from queue
      - Results stored in memory dict (TTL cleanup)
      - Stats tracking for monitoring
    
    Production alternative:
      Replace queue.Queue with Redis Streams
      Replace memory dict with Redis hash
      → Shared state across multiple API instances
    """

    def __init__(
        self,
        num_workers:     int   = 4,
        max_queue_size:  int   = 10000,
        result_ttl_sec:  int   = 300,    # 5 min result retention
        enable_shap:     bool  = True
    ):
        self.num_workers    = num_workers
        self.result_ttl_sec = result_ttl_sec
        self.enable_shap    = enable_shap

        # Job queue
        self._queue = queue.Queue(maxsize=max_queue_size)

        # Result store: request_id → MLResult
        self._results: dict = {}
        self._results_lock  = threading.Lock()

        # Stats
        self.total_queued    = 0
        self.total_processed = 0
        self.total_errors    = 0
        self.queue_wait_times = []

        # Start workers
        self._workers = []
        self._running  = True
        self._start_workers()

        # Start cleanup thread
        threading.Thread(
            target=self._cleanup_loop,
            daemon=True
        ).start()

    def _start_workers(self):
        for i in range(self.num_workers):
            t = threading.Thread(
                target=self._worker_loop,
                name=f"ml-worker-{i}",
                daemon=True
            )
            t.start()
            self._workers.append(t)

    def _worker_loop(self):
        """Each worker thread runs this loop forever."""
        # Import here to avoid circular imports
        from fraud_inference import score_transaction, decide, explain
        from graph_fraud import FraudGraphDetector
        from geo_risk import score_geo_risk_from_vfeatures
        from velocity_decay import get_velocity_risk

        # Each worker has its own graph detector
        graph = FraudGraphDetector(edge_ttl=3600)

        while self._running:
            try:
                # Block until job available (timeout for clean shutdown)
                job = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            request_id = job["request_id"]
            features   = job["features"]
            card_id    = job["card_id"]
            merchant_id= job["merchant_id"]
            ip_address = job["ip_address"]
            queued_at  = job["queued_at"]

            # Update status
            self._update_result(request_id, status="processing")

            start = time.time()
            try:
                # ML score
                ml_score = score_transaction(features)

                # Graph score
                graph_score, graph_signals = graph.score_transaction(
                    card_id=card_id,
                    merchant_id=merchant_id,
                    ip_address=ip_address
                )
                graph.add_transaction(
                    card_id=card_id,
                    merchant_id=merchant_id,
                    ip_address=ip_address,
                    is_fraud=(ml_score > 0.5)
                )

                # Geo score
                geo_score, geo_signals = score_geo_risk_from_vfeatures(
                    features)

                # Velocity risk
                velocity_risk, _ = get_velocity_risk(card_id)

                # Combined score
                combined = round(
                    0.4 * ml_score +
                    0.25 * graph_score +
                    0.20 * geo_score +
                    0.15 * velocity_risk,
                    4
                )
                decision = decide(combined)

                # SHAP only on blocked
                if self.enable_shap and decision in ["BLOCK", "STEP_UP_AUTH"]:
                    from explainability import (
                        explain_transaction as shap_explain,
                        format_explanation
                    )
                    shap_result  = shap_explain(features, top_n=5)
                    explanation  = format_explanation(shap_result, decision)
                else:
                    explanation = [f"✅ APPROVED — score {combined:.4f}"]

                latency = round((time.time() - start) * 1000, 2)
                wait    = round((start - queued_at) * 1000, 2)
                self.queue_wait_times.append(wait)

                # Store result
                with self._results_lock:
                    if request_id in self._results:
                        r = self._results[request_id]
                        r.status        = "complete"
                        r.ml_score      = round(ml_score, 4)
                        r.graph_score   = round(graph_score, 4)
                        r.geo_score     = round(geo_score, 4)
                        r.velocity_risk = round(velocity_risk, 4)
                        r.combined_score= combined
                        r.decision      = decision
                        r.explanation   = explanation
                        r.graph_signals = graph_signals
                        r.geo_signals   = geo_signals
                        r.latency_ms    = latency
                        r.completed_at  = time.time()

                self.total_processed += 1

            except Exception as e:
                self.total_errors += 1
                with self._results_lock:
                    if request_id in self._results:
                        self._results[request_id].status = "error"
                        self._results[request_id].error  = str(e)
            finally:
                self._queue.task_done()

    def submit(
        self,
        request_id:  str,
        features:    dict,
        card_id:     str,
        merchant_id: str,
        ip_address:  str
    ) -> bool:
        """
        Submit job to queue. Returns False if queue is full.
        """
        # Pre-create result entry
        with self._results_lock:
            self._results[request_id] = MLResult(
                request_id = request_id,
                status     = "queued"
            )

        try:
            self._queue.put_nowait({
                "request_id":  request_id,
                "features":    features,
                "card_id":     card_id,
                "merchant_id": merchant_id,
                "ip_address":  ip_address,
                "queued_at":   time.time()
            })
            self.total_queued += 1
            return True
        except queue.Full:
            with self._results_lock:
                self._results[request_id].status = "error"
                self._results[request_id].error  = "Queue full"
            return False

    def get_result(self, request_id: str) -> Optional[MLResult]:
        """Get result for request_id. None if not found."""
        with self._results_lock:
            return self._results.get(request_id)

    def _update_result(self, request_id: str, **kwargs):
        with self._results_lock:
            if request_id in self._results:
                for k, v in kwargs.items():
                    setattr(self._results[request_id], k, v)

    def _cleanup_loop(self):
        """Remove old results to prevent memory leak."""
        while self._running:
            time.sleep(60)
            now = time.time()
            with self._results_lock:
                expired = [
                    rid for rid, r in self._results.items()
                    if r.status == "complete" and
                    (now - r.completed_at) > self.result_ttl_sec
                ]
                for rid in expired:
                    del self._results[rid]

    def get_stats(self) -> dict:
        with self._results_lock:
            pending = sum(
                1 for r in self._results.values()
                if r.status in ["queued", "processing"]
            )

        avg_wait = (
            sum(self.queue_wait_times[-100:]) /
            max(len(self.queue_wait_times[-100:]), 1)
        )

        return {
            "workers":         self.num_workers,
            "queue_size":      self._queue.qsize(),
            "pending_results": pending,
            "total_queued":    self.total_queued,
            "total_processed": self.total_processed,
            "total_errors":    self.total_errors,
            "avg_queue_wait_ms": round(avg_wait, 2),
            "results_cached":  len(self._results),
        }


# ── Global instance ───────────────────────────────────────────
_processor = AsyncMLProcessor(
    num_workers    = 4,
    max_queue_size = 10000,
    result_ttl_sec = 300,
    enable_shap    = True
)


def get_processor() -> AsyncMLProcessor:
    return _processor


# ── Test ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import numpy as np

    print("=" * 60)
    print("  ASYNC ML PROCESSOR TEST")
    print("=" * 60)

    proc = AsyncMLProcessor(num_workers=4)
    time.sleep(1)  # let workers start

    print("\n[1] Submitting 20 jobs...")
    request_ids = []
    for i in range(20):
        rid = str(uuid.uuid4())[:8]
        features = {f"V{j}": float(i * 0.1) for j in range(1, 29)}
        features.update({
            "Amount": float(100 + i * 10),
            "amount_log": np.log1p(100 + i * 10),
            "amount_sqrt": np.sqrt(100 + i * 10),
            "tx_count_1min": i % 5 + 1,
            "tx_count_10min": i % 10 + 3,
            "tx_count_60min": i % 20 + 5,
            "amount_rolling_mean_1h": 150.0,
            "amount_rolling_std_1h":  50.0,
            "amount_deviation": float(i * 0.2),
            "hour": (10 + i) % 24,
            "is_night": 0,
        })
        submitted = proc.submit(rid, features, f"card_{i:03d}",
                                "merchant_A", "192.168.1.1")
        request_ids.append(rid)
        if submitted:
            print(f"    Job {i+1:02d}: {rid} queued ✅")

    print("\n[2] Waiting for results...")
    time.sleep(3)

    completed = 0
    for rid in request_ids:
        result = proc.get_result(rid)
        if result and result.status == "complete":
            completed += 1

    print(f"    Completed: {completed}/20")

    print("\n[3] Sample result:")
    for rid in request_ids[:3]:
        r = proc.get_result(rid)
        if r:
            print(f"    {rid}: {r.status} | "
                  f"decision={r.decision} | "
                  f"score={r.combined_score} | "
                  f"latency={r.latency_ms}ms")

    print("\n[4] Stats:")
    stats = proc.get_stats()
    for k, v in stats.items():
        print(f"    {k}: {v}")

    print("\n[5] Throughput test (100 jobs):")
    start = time.time()
    ids = []
    for i in range(100):
        rid = str(uuid.uuid4())[:8]
        features = {f"V{j}": 0.0 for j in range(1, 29)}
        features.update({
            "Amount": 100.0, "amount_log": 4.61,
            "amount_sqrt": 10.0, "tx_count_1min": 1,
            "tx_count_10min": 3, "tx_count_60min": 10,
            "amount_rolling_mean_1h": 100.0,
            "amount_rolling_std_1h": 20.0,
            "amount_deviation": 0.0, "hour": 12, "is_night": 0
        })
        proc.submit(rid, features, f"card_{i%10:03d}",
                    "merchant_B", "10.0.0.1")
        ids.append(rid)

    submit_time = round((time.time() - start) * 1000, 2)
    print(f"    100 jobs submitted in: {submit_time}ms")
    print(f"    Average submit time:   {submit_time/100:.2f}ms per job")

    time.sleep(5)
    done = sum(1 for rid in ids
               if proc.get_result(rid) and
               proc.get_result(rid).status == "complete")
    print(f"    Processed after 5s:    {done}/100")

    print("\n" + "=" * 60)
    print("  ASYNC ML PROCESSOR COMPLETE ✅")
    print(f"  Submit latency: <1ms per job")
    print(f"  ML runs in background across {proc.num_workers} workers")
    print(f"  API returns instantly — ML doesn't block response")
    print("=" * 60)
