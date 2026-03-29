import streamlit as st
import requests
import time

# ── Page Config ───────────────────────────────────
st.set_page_config(
    page_title="AI Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)

# ── Header ────────────────────────────────────────
st.title("🛡️ AI Credit Card Fraud Detection")
st.markdown("Real-time fraud scoring with **ML + Graph Detection + Redis Rate Limiting**")
st.divider()

API_URL        = "http://127.0.0.1:8000/fraud/check"
HEALTH_URL     = "http://127.0.0.1:8000/health"
RINGS_URL      = "http://127.0.0.1:8000/graph/rings"
METRICS_URL    = "http://127.0.0.1:8000/metrics"

# ── System Status Bar ─────────────────────────────
try:
    health = requests.get(HEALTH_URL, timeout=2).json()
    col_h1, col_h2, col_h3, col_h4, col_h5 = st.columns(5)
    col_h1.metric("Status",         "🟢 Online")
    col_h2.metric("Uptime",         health.get("uptime", "N/A"))
    col_h3.metric("Total Requests", health.get("total_requests", 0))
    col_h4.metric("Fraud Detected", health.get("fraud_detected", 0))
    col_h5.metric("Graph Nodes",    health.get("graph_nodes", 0))
    st.divider()
except:
    st.error("❌ Backend not reachable. Start uvicorn first.")
    st.stop()

# ── Tabs ──────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🔍 Transaction Checker",
    "🕸️ Fraud Ring Monitor",
    "📊 System Metrics"
])

# ══════════════════════════════════════════════════
# TAB 1 — Transaction Checker
# ══════════════════════════════════════════════════
with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📋 Transaction Details")

        amount = st.number_input("Transaction Amount (₹)",
            min_value=1.0, max_value=100000.0, value=2500.0, step=100.0)
        hour   = st.slider("Hour of Day", 0, 23, 2)

        st.markdown("**Velocity Signals**")
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
                    v_vals[f"V{i}"] = st.number_input(
                        f"V{i}", value=0.0, step=0.1,
                        key=f"v{i}", label_visibility="visible"
                    )

        st.divider()
        check_btn  = st.button("🔍 Check Fraud",                  use_container_width=True, type="primary")
        attack_btn = st.button("🚨 Simulate Card-Testing Attack",  use_container_width=True)

    # ── Results Panel ─────────────────────────────
    with col2:
        st.subheader("📊 Analysis Result")

        if check_btn:
            payload = {
                **{f"V{i}": v_vals.get(f"V{i}", 0.0) for i in range(1, 29)},
                "Amount":        amount,
                "tx_count_1min":  tx_1min,
                "tx_count_10min": tx_10min,
                "tx_count_60min": tx_60min,
                "hour":           hour,
                "card_id":        card_id,
                "merchant_id":    merchant_id,
                "ip":             ip_addr,
            }

            with st.spinner("Analyzing transaction..."):
                try:
                    res = requests.post(API_URL, json=payload, timeout=5)

                    if res.status_code == 429:
                        st.error("🚫 **Rate Limit Exceeded** — Too many requests blocked!")
                        st.info("Your rate limiter (Redis) is working correctly.")

                    elif res.status_code == 200:
                        data        = res.json()
                        risk        = data["risk_score"]
                        ml_score    = data["ml_score"]
                        graph_score = data["graph_score"]
                        decision    = data["decision"]
                        reasons     = data["explanation"]
                        g_signals   = data["graph_signals"]
                        req_id      = data["request_id"]
                        latency     = data["latency_ms"]

                        # Request ID + Latency
                        st.caption(f"Request ID: `{req_id}` | Latency: `{latency}ms`")

                        # ── Score Comparison ──────────────────────
                        st.markdown("**Score Breakdown**")
                        sc1, sc2, sc3 = st.columns(3)
                        sc1.metric("Combined Risk", f"{risk:.4f}",
                            delta=f"{risk*100:.1f}%")
                        sc2.metric("ML Score",    f"{ml_score:.4f}")
                        sc3.metric("Graph Score", f"{graph_score:.4f}",
                            delta="🕸️ Graph" if graph_score > 0 else None)

                        # ── Risk Gauge ────────────────────────────
                        st.markdown("**Risk Level**")
                        color = "🔴" if risk > 0.6 else "🟡" if risk > 0.25 else "🟢"
                        st.progress(min(risk, 1.0))
                        st.markdown(f"{color} **{risk*100:.1f}% fraud probability**")

                        # ── Decision ──────────────────────────────
                        st.markdown("**Decision**")
                        if decision == "APPROVE":
                            st.success("✅ APPROVE — Transaction is safe")
                        elif decision == "STEP_UP_AUTH":
                            st.warning("⚠️ STEP_UP_AUTH — Additional verification required")
                        else:
                            st.error("🚫 BLOCK — Transaction blocked as fraud")

                        # ── ML Explanation ────────────────────────
                        st.markdown("**ML Signals**")
                        for r in reasons:
                            st.markdown(f"• {r}")

                        # ── Graph Signals ─────────────────────────
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

                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot reach backend on port 8000.")

        # ── Attack Simulation ─────────────────────
        if attack_btn:
            st.warning("🚨 Simulating card-testing attack...")
            payload = {
                **{f"V{i}": 0.0 for i in range(1, 29)},
                "Amount": 1.0, "tx_count_1min": 10,
                "tx_count_10min": 50, "tx_count_60min": 100,
                "hour": 3, "card_id": "attacker",
                "merchant_id": "M_attack", "ip": "1.2.3.4"
            }

            results  = []
            progress = st.progress(0)
            for i in range(10):
                try:
                    res = requests.post(API_URL, json=payload, timeout=3)
                    if res.status_code == 429:
                        results.append(f"Request {i+1}: 🚫 BLOCKED (429)")
                    else:
                        results.append(f"Request {i+1}: ✅ Processed")
                except:
                    results.append(f"Request {i+1}: ❌ Error")
                progress.progress((i+1)/10)
                time.sleep(0.1)

            for r in results:
                st.markdown(r)
            blocked = sum(1 for r in results if "BLOCKED" in r)
            st.info(f"📊 {blocked}/10 requests blocked by Redis rate limiter")

# ══════════════════════════════════════════════════
# TAB 2 — Fraud Ring Monitor
# ══════════════════════════════════════════════════
with tab2:
    st.subheader("🕸️ Live Fraud Ring Monitor")
    st.markdown("Detected fraud rings based on shared card → merchant → IP patterns")

    if st.button("🔄 Refresh Rings"):
        pass  # triggers rerun

    try:
        rings_data = requests.get(RINGS_URL, timeout=3).json()
        rings      = rings_data.get("rings", [])
        total      = rings_data.get("total_rings", 0)

        if total == 0:
            st.info("No fraud rings detected yet. Send transactions with shared merchant/IP to build the graph.")
        else:
            st.metric("Total Rings Detected", total)
            st.divider()

            for i, ring in enumerate(rings, 1):
                risk_color = "🔴" if ring['risk'] == "HIGH" else "🟡"
                with st.expander(
                    f"{risk_color} Ring {i} — {ring['ring_pattern']} | Risk: {ring['risk']}",
                    expanded=(i == 1)
                ):
                    r1, r2, r3 = st.columns(3)
                    r1.metric("Cards in Ring",    len(ring['cards']))
                    r2.metric("Shared Merchants", len(ring['merchants']))
                    r3.metric("Shared IPs",       len(ring['ips']))

                    st.markdown("**Cards:**")
                    st.code(", ".join(ring['cards']))

                    st.markdown("**Merchants:**")
                    st.code(", ".join(ring['merchants']))

                    st.markdown("**IPs:**")
                    st.code(", ".join(ring['ips']))

                    if ring['risk'] == "HIGH":
                        st.error(f"🚨 HIGH RISK: {len(ring['cards'])} cards sharing same infrastructure")
                    else:
                        st.warning(f"⚠️ MEDIUM RISK: Suspicious shared pattern detected")

    except Exception as e:
        st.error(f"Could not fetch rings: {e}")

    # ── Simulate Fraud Ring ────────────────────────
    st.divider()
    st.markdown("**🧪 Simulate a Fraud Ring**")
    st.markdown("Send multiple cards through the same merchant + IP to trigger ring detection.")

    num_cards   = st.slider("Number of cards in ring", 2, 8, 5)
    ring_merch  = st.text_input("Ring Merchant ID", value="RING_MERCHANT")
    ring_ip     = st.text_input("Ring IP Address",  value="192.168.99.1")

    if st.button("🚀 Simulate Fraud Ring", type="primary"):
        st.info(f"Sending {num_cards} cards through {ring_merch} @ {ring_ip}...")
        sim_results = []
        prog = st.progress(0)

        for i in range(1, num_cards + 1):
            payload = {
                **{f"V{i2}": 0.0 for i2 in range(1, 29)},
                "Amount": 1.0, "tx_count_1min": 3,
                "tx_count_10min": 10, "tx_count_60min": 30,
                "hour": 2,
                "card_id":     f"ring_card_{i:03d}",
                "merchant_id": ring_merch,
                "ip":          ring_ip
            }
            try:
                res  = requests.post(API_URL, json=payload, timeout=5)
                if res.status_code == 200:
                    data = res.json()
                    sim_results.append({
                        "card":    f"ring_card_{i:03d}",
                        "risk":    data['risk_score'],
                        "graph":   data['graph_score'],
                        "decision": data['decision'],
                        "signals": data['graph_signals']
                    })
                time.sleep(0.3)
            except:
                pass
            prog.progress(i / num_cards)

        st.markdown("**Results:**")
        for r in sim_results:
            color = "🔴" if r['decision'] == "BLOCK" else "🟡" if r['decision'] == "STEP_UP_AUTH" else "🟢"
            st.markdown(
                f"{color} **{r['card']}** → "
                f"Combined: `{r['risk']}` | "
                f"Graph: `{r['graph']}` | "
                f"{r['decision']}"
            )
            for s in r['signals']:
                if "No suspicious" not in s:
                    st.caption(f"  ↳ ⚠️ {s}")

        st.success("Done! Check the Fraud Ring Monitor above (click Refresh).")

# ══════════════════════════════════════════════════
# TAB 3 — System Metrics
# ══════════════════════════════════════════════════
with tab3:
    st.subheader("📊 System Metrics")

    if st.button("🔄 Refresh Metrics"):
        pass

    try:
        m = requests.get(METRICS_URL, timeout=3).json()
        h = requests.get(HEALTH_URL,  timeout=3).json()

        # Request stats
        st.markdown("**Request Statistics**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Requests",   m['total_requests'])
        c2.metric("Fraud Rate",        f"{m['fraud_rate_pct']}%")
        c3.metric("Approval Rate",     f"{m['approval_rate_pct']}%")
        c4.metric("Rate Block Rate",   f"{m['rate_block_rate_pct']}%")

        st.divider()

        # Graph stats
        st.markdown("**Graph Statistics**")
        gs = m.get('graph_stats', {})
        g1, g2, g3, g4 = st.columns(4)
        g1.metric("Graph Nodes",      gs.get('total_nodes', 0))
        g2.metric("Graph Edges",      gs.get('total_edges', 0))
        g3.metric("Fraud Rings",      gs.get('fraud_rings', 0))
        g4.metric("High Risk Rings",  gs.get('high_risk_rings', 0))

        st.divider()

        # System health
        st.markdown("**System Health**")
        s1, s2, s3 = st.columns(3)
        s1.metric("Redis Backend", "✅ Connected" if m['redis_backend'] else "⚠️ Fallback")
        s2.metric("Uptime",        h.get('uptime', 'N/A'))
        s3.metric("Graph Flagged", h.get('graph_flagged', 0))

        st.divider()
        st.markdown("**Raw Metrics JSON**")
        st.json(m)

    except Exception as e:
        st.error(f"Could not fetch metrics: {e}")

# ── Footer ────────────────────────────────────────
st.divider()
st.markdown("""
<div style='text-align:center; color:gray; font-size:12px'>
🛡️ AI Credit Card Fraud Detection System &nbsp;|&nbsp;
FastAPI + Streamlit + Gradient Boosting + Graph ML &nbsp;|&nbsp;
Redis Rate Limiting &nbsp;|&nbsp; ROC-AUC: 0.9850
</div>
""", unsafe_allow_html=True)
