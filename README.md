# 🛡️ AI Credit Card Fraud Detection System — v7.0.0

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-red)](https://ai-fraud-detection-rmanojgowda.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-7.0.0-green)](https://fastapi.tiangolo.com)
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.9883-brightgreen)](https://github.com/rmanojgowda/ai-fraud-detection)

👉 **[Try Live Demo](https://ai-fraud-detection-rmanojgowda.streamlit.app)**

Production-grade fraud detection platform with ML, Graph Analysis, Redis Rate Limiting, async pipelines, and real-time observability. Built to mirror how real fintech fraud platforms work — not just a trained model.

---

## 🎯 Key Metrics

| Metric | Value |
|--------|-------|
| ROC-AUC | **0.9883** |
| Precision | **93.44%** (only 4 false positives on 56,962 transactions) |
| Recall | **76%** (18 missed frauds proven indistinguishable — V14 analysis) |
| Peak Throughput (single machine) | **100,437 RPM** (Redis Streams async) |
| Peak Throughput (GCP 100 instances) | **3,543,148 RPM** |
| P95 Latency | **13.4ms** (async endpoint) |
| Batch Processing | **7ms/transaction** (100tx batch) |
| Error Rate | **0%** across all load scenarios |
| Training Data | 284,807 real bank transactions |
| Features | 39 (V1-V28 + 11 engineered) |
| Workers | 8 (matches CPU cores) |

---

## 🚀 Scaling Journey

```
Phase 1 (baseline sklearn):      18,327 RPM
v7 SHAP optimization:            26,692 RPM  (+46%)
v7 Async ML pipeline:            86,286 RPM  (+370%)
v7 Redis Streams:               100,437 RPM  (+448%)
GCP Cloud Run 10 instances:     354,315 RPM
GCP Cloud Run 100 instances:  3,543,148 RPM  ← 50x Razorpay peak
```

Three architectural changes drove this:
1. **Conditional SHAP** — skip explanations on approved transactions (saves 5-10ms on 98% of requests)
2. **Async ML queue** — API returns immediately, ML runs in background
3. **Redis Streams** — Kafka-like persistent queue using existing Redis

---

## 🏗️ Architecture — 4 Defense Layers

```
Client Request
      ↓
┌─────────────────────────────────┐
│  Triple-Layer Rate Limiter      │  ← Redis, <1ms
│  5/10s + 100/hr + 3/hr/card     │  (blocks card-testing attacks)
└──────────────┬──────────────────┘
               ↓
┌─────────────────────────────────┐
│  LightGBM ML Inference          │  ← 39 features, ROC-AUC 0.9883
│  ThreadPoolExecutor parallel    │  (releases GIL for true parallelism)
└──────────────┬──────────────────┘
               ↓
┌─────────────────────────────────┐
│  Redis-Backed Graph Detection   │  ← shared across ALL 8 workers
│  IP rings + merchant rings      │  (fraud rings visible to every worker)
└──────────────┬──────────────────┘
               ↓
┌─────────────────────────────────┐
│  Geographic Risk Scoring        │  ← V-feature validated on real data
│  Country risk + impossible      │  81.3% recall, 0 false positives
│  travel + VPN detection         │
└──────────────┬──────────────────┘
               ↓
      APPROVE / STEP_UP_AUTH / BLOCK
```

---

## ✨ 9 Production Features

| Feature | Description | Interview Value |
|---------|-------------|-----------------|
| **Card-Level Rate Limiting** | 3 attempts/hr per card — closes IP rotation bypass | "Identified gap myself during testing" |
| **Prometheus + Grafana** | Live metrics dashboard, P95 latency graphs | Production observability |
| **SHAP Explainability** | Per-transaction feature explanation on blocked txns | ML beyond accuracy numbers |
| **A/B Testing Framework** | Deterministic traffic splitting, winner detection | Safe model deployment |
| **Hourly Fraud Heatmap** | Fraud peaks at 2-4am → dynamic threshold adjustment | Business insight |
| **Transaction Replay** | Save blocked txns, replay vs new model | MLOps thinking |
| **Geographic Risk** | V-feature based, validated on real data (81.3% recall) | Real data validation |
| **Webhook Alerts** | Slack/HTTP alerts on fraud ring detection | Production awareness |
| **Velocity Decay** | Recent transactions weighted higher — burst attack detection | Same count, different risk |

---

## 📊 API Endpoints

```
POST /fraud/check          ← sync inference (full pipeline)
POST /fraud/check/async    ← async (returns immediately, poll result)
POST /fraud/check/stream   ← Redis Streams (Kafka-like, persistent)
POST /fraud/check/batch    ← batch up to 1000 transactions (7ms/tx)

GET  /health               ← system status
GET  /metrics              ← Prometheus metrics
GET  /stats                ← fraud rates, approval rates
GET  /graph/rings          ← active fraud rings
GET  /heatmap              ← hourly fraud pattern
GET  /alerts/history       ← recent webhook alerts
GET  /velocity/stats       ← velocity decay stats
GET  /ab-test/results      ← A/B experiment results
GET  /replay/stats         ← transaction replay stats
GET  /geo/stats            ← geographic risk stats
```

---

## 🔥 Load Test Results — 8 Workers

| Scenario | RPM | P95ms | Errors |
|----------|-----|-------|--------|
| Baseline (2 threads) | 5,153 | 24.2ms | 0 |
| Normal Load (10 threads) | 14,921 | 36.7ms | 0 |
| Stress Test (50 threads) | 14,892 | 32.3ms | 0 |
| Breaking Point (100 threads) | 12,608 | **43.7ms** | **0** |
| Async endpoint (100 threads) | 30,036 | **13.4ms** | 0 |
| Stream endpoint | 35,431 | 214ms | 0 |
| Batch (100 tx/req) | 7,383 effective RPM | 8ms/tx | 0 |

**Zero errors across all 8 scenarios at 100 concurrent threads.**

---

## 🔥 Chaos Test Results

```
Before Redis failure : [200, 200, 200, 200, 200] ✅
During Redis failure : [200, 200, 200, 200, 200] ✅ (fallback)
After Redis recovery : [429, 429, 429, 429, 429] ✅ (restored)

Graceful degradation : ✅ PASSED
Full recovery        : ✅ PASSED
```

---

## 🧠 ML Pipeline

**Dataset:** 284,807 real European bank transactions (0.17% fraud — 545:1 imbalance)

**Why LightGBM:**
- 12.5s training vs 2-3 min (sklearn)
- 98.9% fewer false positives (4 vs 395)
- Releases Python GIL → true parallelism with ThreadPoolExecutor

**Threshold Decision:**
```
Unconstrained minimum: 0.02 → 66% precision (rejected)
With precision ≥ 90% constraint: 0.7722 → saves ₹1,800 vs default 0.5
```

**Why 76% recall (not 85%):**
- Investigated 18 missed frauds: V14 mean = -1.7 vs caught frauds at -8.2
- Score gap = 0.736 — unbridgeable by any threshold or technique
- Tested 5 LightGBM configs + SMOTE + ensemble — all converge to same 18 cases
- Dataset limitation, not modeling limitation
- Accepted tradeoff: 98.9% fewer false positives > recovering 9% recall

---

## 🕸️ Redis-Backed Shared Graph

**Problem solved:**
```
Before: 8 workers × independent graphs
  Worker 1 detects ring → Workers 2-8 are blind

After: 8 workers × ONE shared Redis graph
  card_004 on any worker → ring detected ✅
```

**Proven in test:**
```
Worker 1 adds card_001, card_002, card_003 (same IP)
Worker 2 scores card_004: score=0.60 ← SEES the ring ✅
Memory fallback: score=0.00 ← blind ❌
```

---

## 🌍 Geographic Risk Scoring

**Two modes:**
1. **Country/IP mode** — country risk levels, impossible travel, VPN detection
2. **V-feature mode** — data-driven, validated on real creditcard.csv

**V-feature validation:**
```
Fraud V14 mean:  -6.97  (577x more negative than normal)
Normal V14 mean: +0.01

At threshold 0.20:
  Recall: 81.3%
  Precision: 100% (zero false positives)
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/rmanojgowda/ai-fraud-detection.git
cd ai-fraud-detection
python -m venv venv
.\venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

Download [creditcard.csv](https://www.kaggle.com/mlg-ulb/creditcardfraud) → `data/`

```bash
python train_model.py
D:\Redis\redis-server.exe          # Terminal 1
uvicorn main:app --workers 8       # Terminal 2
streamlit run dashboard.py         # Terminal 3
```

---

## 📁 Project Structure

```
ai-fraud-detection/
├── main.py                  # FastAPI v7.0.0 — all endpoints
├── train_model.py           # LightGBM training pipeline
├── fraud_inference.py       # ML scoring + decisions
├── graph_fraud.py           # NetworkX graph (fallback)
├── redis_graph.py           # Redis-backed shared graph ← NEW
├── rate_limiter.py          # Triple-layer rate limiter
├── explainability.py        # SHAP explainability
├── ab_testing.py            # A/B testing framework
├── fraud_heatmap.py         # Hourly heatmap + dynamic thresholds
├── transaction_replay.py    # MLOps replay system
├── geo_risk.py              # Geographic risk (real-data validated)
├── webhook_alerts.py        # Slack/HTTP fraud alerts
├── velocity_decay.py        # Exponential decay velocity
├── batch_processor.py       # Batch endpoint (7ms/tx)
├── async_processor.py       # Async ML pipeline
├── redis_stream_processor.py# Redis Streams (Kafka-like)
├── metrics.py               # Prometheus metrics
├── dashboard.py             # Streamlit 3-tab dashboard
├── load_test.py             # 8-scenario load test
├── chaos_test.py            # Redis failure simulation
└── final_scale_test.py      # Complete scaling comparison
```

---

## 🎯 Design Decisions

**Why rate limiting before ML?**
ML costs ~10ms. Rate limiting costs ~1ms. Blocking card-testing attacks before ML saves 50%+ compute under attack.

**Why LightGBM over sklearn?**
10x faster training, 98.9% fewer false positives. sklearn had better recall (85% vs 76%) but 395 false positives — blocking 395 legitimate customers per test window is unacceptable.

**Why Redis graph over NetworkX?**
8 workers × independent in-memory graphs = inconsistent ring detection. Redis gives shared state across all workers and survives restarts.

**Why async over sync?**
Sync endpoint: ML blocks API thread (20-40ms). Async endpoint: API returns in <1ms, ML runs in background. 3x throughput improvement for same hardware.

**Why SHAP only on blocked?**
Nobody needs to know why a transaction was approved. SHAP on every request wastes 5-10ms on 98% of traffic. Conditional SHAP gives same fraud explainability at 3x throughput.

---

## 👤 Author

**Manoj Gowda B G**
B.E. Information Science & Engineering
Siddaganga Institute of Technology, Tumkur (2026)

Built to demonstrate production-grade fraud detection engineering — layered defense, distributed systems, MLOps thinking, and honest architectural tradeoffs.

---

## 📄 License

MIT License — free to use and modify.
