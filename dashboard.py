import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
import os
import time
from collections import defaultdict

# ── Page Config ───────────────────────────────────
st.set_page_config(
    page_title="AI Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)

# ── Load Model (Direct — No Backend Needed) ───────
@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model = joblib.load(os.path.join(BASE_DIR, "models", "fraud_model.pkl"))
    with open(os.path.join(BASE_DIR, "models", "feature_cols.json")) as f:
        feature_cols = json.load(f)
    with open(os.path.join(BASE_DIR, "models", "threshold.json")) as f:
        threshold = json.load(f)["threshold"]
    return model, feature_cols, threshold

model, FEATURE_COLS, THRESHOLD = load_model()

# ── In-Memory Rate Limiter ────────────────────────
if "rate_tracker" not in st.session_state:
    st.session_state.rate_tracker = defaultdict(list)
if "total_requests" not in st.session_state:
    st.session_state.total_requests = 0
if "fraud_detected" not in st.session_state:
    st.session_state.fraud_detected = 0
if "graph" not in st.session_state:
    st.session_state.graph = defaultdict(set)
if "graph_fraud_count" not in st.session_state:
    st.session_state.graph_fraud_count = defaultdict(int)
if "graph_tx_count" not in st.session_state:
    st.session_state.graph_tx_count = defaultdict(int)

RATE_LIMIT = 5
WINDOW_SEC = 10

def check_rate_limit(client_id: str) -> bool:
    now = time.time()
    timestamps = st.session_state.rate_tracker[client_id]
    timestamps = [t for t in timestamps if now - t < WINDOW_SEC]
    if len(timestamps) >= RATE_LIMIT:
        st.session_state.rate_tracker[client_id] = timestamps
        return False
    timestamps.append(now)
    st.session_state.rate_tracker[client_id] = timestamps
    return True

# ── ML Scoring ────────────────────────────────────
def score_transaction(features: dict) -> float:
    df = pd.DataFrame([features], columns=FEATURE_COLS)
    return float(model.predict_proba(df)[0][1])

def decide(risk: float) -> str:
    if risk < 0.25:   return "APPROVE"
    elif risk < 0.6:  return "STEP_UP_AUTH"
    else:             return "BLOCK"

def explain(features: dict, risk: float) -> list:
    reasons = []
    if features.get("tx_count_1min", 0) > 3:
        reasons.append("High transaction velocity in last 1 minute")
    if features.get("tx_count_10min", 0) > 10:
        reasons.append("Unusual frequency in last 10 minutes")
    if features.get("is_night", 0) == 1:
        reasons.append("Transaction occurred at night")
    if features.get("Amount", 0) > 2000:
        reasons.append("Unusually high transaction amount")
    if risk > 0.6:
        reasons.append("Strong fraud pattern detected by ML model")
    if not reasons:
        reasons.append("Transaction behavior within normal limits")
    return reasons

# ── Graph Detection ───────────────────────────────
def add_to_graph(card_id, merchant_id, ip, is_fraud):
    g = st.session_state.graph
    g[f"card_{card_id}"].add(f"merchant_{merchant_id}")
    g[f"card_{card_id}"].add(f"ip_{ip}")
    g[f"merchant_{merchant_id}"].add(f"ip_{ip}")
    for node in [f"card_{card_id}", f"merchant_{merchant_id}", f"ip_{ip}"]:
        st.session_state.graph_tx_count[node] += 1
        if is_fraud:
            st.session_state.graph_fraud_count[node] += 1

def get_graph_score(card_id, merchant_id, ip) -> tuple:
    g          = st.session_state.graph
    ip_node    = f"ip_{ip}"
    card_node  = f"card_{card_id}"
    risk       = 0.0
    signals    = []

    # Cards sharing same IP
    cards_on_ip = [n for n in g if n.startswith("card_") and ip_node in g[n]]
    if len(cards_on_ip) >= 2:
        risk += 0.4
        signals.append(f"IP shared by {len(cards_on_ip)} different cards — possible fraud ring")
    if len(cards_on_ip) >= 4:
        risk += 0.3
        signals.append(f"FRAUD RING: {len(cards_on_ip)} cards sharing same IP and merchant")

    # Known fraud card
    tx  = st.session_state.graph_tx_count.get(card_node, 0)
    frd = st.session_state.graph_fraud_count.get(card_node, 0)
    if tx > 0 and frd / tx > 0.3:
        risk += 0.4
        signals.append(f"Card has {frd/tx*100:.0f}% historical fraud rate")

    if not signals:
        signals.append("No suspicious graph patterns detected")

    return min(round(risk, 4), 1.0), signals

def detect_rings():
    g     = st.session_state.graph
    rings = []
    ips   = [n for n in g if n.startswith("ip_")]
    for ip_node in ips:
        cards = [n for n in g if n.startswith("card_") and ip_node in g[n]]
        merchants = [n for n in g if n.startswith("merchant_") and ip_node in g[n]]
        if len(cards) >= 2:
            fraud_count = sum(st.session_state.graph_fraud_count.get(c, 0) for c in cards)
            rings.append({
                "cards": cards, "merchants": merchants,
                "ips": [ip_node], "total_fraud": fraud_count,
                "ring_pattern": f"{len(cards)} cards → {len(merchants)} merchants → 1 IPs",
                "risk": "HIGH" if len(cards) >= 4 else "MEDIUM"
            })
    return sorted(rings, key=lambda x: len(x["cards"]), reverse=True)

# ══════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════
st.title("🛡️ AI Credit Card Fraud Detection")
st.markdown("Real-time fraud scoring with **ML + Graph Detection + Rate Limiting**")

# Status bar
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Status",         "🟢 Online")
c2.metric("Model",          "LightGBM")
c3.metric("ROC-AUC",        "0.9883")
c4.metric("Total Requests", st.session_state.total_requests)
c5.metric("Fraud Detected", st.session_state.fraud_detected)
st.divider()

tab1, tab2, tab3 = st.tabs([
    "🔍 Transaction Checker",
    "🕸️ Fraud Ring Monitor",
    "📊 System Metrics"
])

# ══════════════════════════════════════════════════
# TAB 1
# ══════════════════════════════════════════════════
with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📋 Transaction Details")
        amount   = st.number_input("Transaction Amount (₹)", 1.0, 100000.0, 2500.0, 100.0)
        hour     = st.slider("Hour of Day", 0, 23, 2)
        tx_1min  = st.number_input("Transactions in last 1 min",  0, 50,  value=4)
        tx_10min = st.number_input("Transactions in last 10 min", 0, 100, value=12)
        tx_60min = st.number_input("Transactions in last 60 min", 0, 200, value=20)

        st.markdown("**Entity IDs** *(for graph detection)*")
        card_id     = st.text_input("Card ID",     value="card_001")
        merchant_id = st.text_input("Merchant ID", value="merchant_A")
        ip_addr     = st.text_input("IP Address",  value="10.0.0.1")

        with st.expander("Advanced: V1-V28 PCA Features"):
            v_vals = {}
            vcols  = st.columns(4)
            for i in range(1, 29):
                with vcols[(i-1) % 4]:
                    v_vals[f"V{i}"] = st.number_input(f"V{i}", value=0.0, step=0.1, key=f"v{i}")

        st.divider()
        check_btn  = st.button("🔍 Check Fraud",                 use_container_width=True, type="primary")
        attack_btn = st.button("🚨 Simulate Card-Testing Attack", use_container_width=True)

    with col2:
        st.subheader("📊 Analysis Result")

        if check_btn:
            allowed = check_rate_limit("user")
            st.session_state.total_requests += 1

            if not allowed:
                st.error("🚫 **Rate Limit Exceeded** — Too many requests!")
                st.info("Rate limiter blocked this request. Wait 10 seconds and try again.")
            else:
                features = {
                    **{f"V{i}": v_vals.get(f"V{i}", 0.0) for i in range(1, 29)},
                    "Amount":                amount,
                    "amount_log":            np.log1p(amount),
                    "tx_count_1min":         tx_1min,
                    "tx_count_10min":        tx_10min,
                    "tx_count_60min":        tx_60min,
                    "amount_rolling_mean_1h": amount,
                    "hour":                  hour,
                    "is_night":              1 if hour < 5 else 0
                }

                start      = time.time()
                ml_score   = score_transaction(features)
                graph_score, g_signals = get_graph_score(card_id, merchant_id, ip_addr)
                combined   = round(0.6 * ml_score + 0.4 * graph_score, 4)
                decision   = decide(combined)
                reasons    = explain(features, combined)
                latency    = round((time.time() - start) * 1000, 2)

                add_to_graph(card_id, merchant_id, ip_addr, 1 if decision == "BLOCK" else 0)

                if decision == "BLOCK":
                    st.session_state.fraud_detected += 1

                st.caption(f"Latency: `{latency}ms`")

                sc1, sc2, sc3 = st.columns(3)
                sc1.metric("Combined Risk", f"{combined:.4f}", f"{combined*100:.1f}%")
                sc2.metric("ML Score",      f"{ml_score:.4f}")
                sc3.metric("Graph Score",   f"{graph_score:.4f}")

                st.markdown("**Risk Level**")
                color = "🔴" if combined > 0.6 else "🟡" if combined > 0.25 else "🟢"
                st.progress(min(combined, 1.0))
                st.markdown(f"{color} **{combined*100:.1f}% fraud probability**")

                if decision == "APPROVE":
                    st.success("✅ APPROVE — Transaction is safe")
                elif decision == "STEP_UP_AUTH":
                    st.warning("⚠️ STEP_UP_AUTH — Additional verification required")
                else:
                    st.error("🚫 BLOCK — Transaction blocked as fraud")

                st.markdown("**ML Signals**")
                for r in reasons:
                    st.markdown(f"• {r}")

                st.markdown("**🕸️ Graph Signals**")
                no_pattern = all("No suspicious" in s for s in g_signals)
                if no_pattern:
                    st.info("No graph anomalies detected")
                else:
                    for s in g_signals:
                        if "FRAUD RING" in s:
                            st.error(f"🚨 {s}")
                        elif "No suspicious" not in s:
                            st.warning(f"⚠️ {s}")

        if attack_btn:
            st.warning("🚨 Simulating card-testing attack — 10 rapid requests...")
            results  = []
            progress = st.progress(0)
            for i in range(10):
                st.session_state.total_requests += 1
                if check_rate_limit("attacker"):
                    results.append(f"Request {i+1}: ✅ Processed")
                else:
                    results.append(f"Request {i+1}: 🚫 BLOCKED (Rate Limited)")
                progress.progress((i+1)/10)
                time.sleep(0.05)
            for r in results:
                st.markdown(r)
            blocked = sum(1 for r in results if "BLOCKED" in r)
            st.info(f"📊 {blocked}/10 requests blocked by rate limiter")

# ══════════════════════════════════════════════════
# TAB 2
# ══════════════════════════════════════════════════
with tab2:
    st.subheader("🕸️ Live Fraud Ring Monitor")

    rings = detect_rings()
    if not rings:
        st.info("No fraud rings detected yet. Use the Simulate button below to build a ring.")
    else:
        st.metric("Total Rings Detected", len(rings))
        for i, ring in enumerate(rings, 1):
            color = "🔴" if ring['risk'] == "HIGH" else "🟡"
            with st.expander(f"{color} Ring {i} — {ring['ring_pattern']} | {ring['risk']}", expanded=(i==1)):
                r1, r2, r3 = st.columns(3)
                r1.metric("Cards",     len(ring['cards']))
                r2.metric("Merchants", len(ring['merchants']))
                r3.metric("IPs",       len(ring['ips']))
                st.code(", ".join(ring['cards']))
                if ring['risk'] == "HIGH":
                    st.error(f"🚨 HIGH RISK: {len(ring['cards'])} cards sharing same infrastructure")

    st.divider()
    st.markdown("**🧪 Simulate a Fraud Ring**")
    num_cards  = st.slider("Number of cards", 2, 8, 5)
    ring_merch = st.text_input("Ring Merchant ID", value="FRAUD_MERCHANT")
    ring_ip    = st.text_input("Ring IP Address",  value="192.168.99.1")

    if st.button("🚀 Simulate Fraud Ring", type="primary"):
        st.info(f"Sending {num_cards} cards through {ring_merch} @ {ring_ip}...")
        prog = st.progress(0)
        sim_results = []

        for i in range(1, num_cards + 1):
            features = {
                **{f"V{j}": 0.0 for j in range(1, 29)},
                "Amount": 1.0, "amount_log": np.log1p(1.0),
                "tx_count_1min": 3, "tx_count_10min": 10,
                "tx_count_60min": 30, "amount_rolling_mean_1h": 1.0,
                "hour": 2, "is_night": 1
            }
            ml_score   = score_transaction(features)
            graph_score, g_signals = get_graph_score(f"ring_card_{i:03d}", ring_merch, ring_ip)
            combined   = round(0.6 * ml_score + 0.4 * graph_score, 4)
            decision   = decide(combined)
            add_to_graph(f"ring_card_{i:03d}", ring_merch, ring_ip, 0)
            sim_results.append({
                "card": f"ring_card_{i:03d}",
                "combined": combined, "graph": graph_score,
                "decision": decision, "signals": g_signals
            })
            prog.progress(i / num_cards)

        for r in sim_results:
            color = "🔴" if r['decision'] == "BLOCK" else "🟡" if r['decision'] == "STEP_UP_AUTH" else "🟢"
            st.markdown(f"{color} **{r['card']}** → Combined: `{r['combined']}` | Graph: `{r['graph']}` | {r['decision']}")
            for s in r['signals']:
                if "No suspicious" not in s:
                    st.caption(f"  ↳ ⚠️ {s}")

        st.success("Done! Scroll up and refresh to see the ring.")
        st.rerun()

# ══════════════════════════════════════════════════
# TAB 3
# ══════════════════════════════════════════════════
with tab3:
    st.subheader("📊 System Metrics")

    total = max(st.session_state.total_requests, 1)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Requests",  st.session_state.total_requests)
    c2.metric("Fraud Detected",  st.session_state.fraud_detected)
    c3.metric("Fraud Rate",      f"{st.session_state.fraud_detected/total*100:.1f}%")
    c4.metric("Graph Nodes",     len(st.session_state.graph))

    st.divider()
    st.markdown("**Proven Performance (Local Load Test)**")
    perf_data = {
        "Scenario":   ["Baseline", "Light", "Normal", "Medium", "Heavy", "Stress", "Peak", "Breaking"],
        "Threads":    [2, 5, 10, 20, 30, 50, 75, 100],
        "RPM":        [5382, 9876, 11818, 18327, 15081, 18214, 17538, 17438],
        "P95 (ms)":   [30.6, 43.3, 77.2, 93.3, 100.7, 52.2, 56.2, 47.2],
        "Errors":     [0, 0, 0, 0, 0, 0, 0, 0]
    }
    st.dataframe(pd.DataFrame(perf_data), use_container_width=True)

    st.markdown("**Chaos Test Results**")
    st.success("✅ Redis failure → Graceful fallback (zero downtime)")
    st.success("✅ Redis recovery → Full reconnection")
    st.success("✅ Zero errors across all 8 load scenarios")

    st.divider()
    st.markdown("**Model Performance**")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ROC-AUC",       "0.9883")
    m2.metric("Fraud Recall",  "76% (93% precision)")
    m3.metric("Frauds Caught", "57/75")
    m4.metric("Threshold",     "0.7722")

# Footer
st.divider()
st.markdown("""
<div style='text-align:center; color:gray; font-size:12px'>
🛡️ AI Credit Card Fraud Detection System &nbsp;|&nbsp;
FastAPI + Streamlit + LightGBM + Graph ML &nbsp;|&nbsp;
Redis Rate Limiting &nbsp;|&nbsp; ROC-AUC: 0.9883 &nbsp;|&nbsp;
<a href='https://github.com/rmanojgowda/ai-fraud-detection'>GitHub</a>
</div>
""", unsafe_allow_html=True)
