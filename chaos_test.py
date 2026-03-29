import requests
import time

API_URL = "http://127.0.0.1:8000/fraud/check"

PAYLOAD = {
    **{f"V{i}": 0.0 for i in range(1, 29)},
    "Amount": 100.0, "tx_count_1min": 1,
    "tx_count_10min": 3, "tx_count_60min": 10,
    "hour": 14, "card_id": "chaos_card",
    "merchant_id": "M_chaos", "ip": "10.0.0.99"
}

def send_requests(label: str, count: int):
    results = []
    for i in range(count):
        try:
            res = requests.post(API_URL, json=PAYLOAD, timeout=5)
            results.append(res.status_code)
            print(f"  Request {i+1}: {res.status_code} ✅")
        except Exception as e:
            results.append("error")
            print(f"  Request {i+1}: ERROR ❌ ({str(e)[:40]})")
        time.sleep(0.5)
    return results

print("=" * 55)
print("  🔥 CHAOS TEST — Redis Failure Simulation")
print("=" * 55)

# ── Phase 1: Before chaos ─────────────────────────
print("\n[1/3] Sending 5 requests with Redis UP...")
before = send_requests("before", 5)
print(f"\n  Result: {before}")
all_ok = all(s in [200, 429] for s in before)
print(f"  Status: {'✅ All healthy' if all_ok else '❌ Issues detected'}")

# ── Phase 2: During chaos ─────────────────────────
print("\n" + "="*55)
print("  ⚠️  ACTION REQUIRED:")
print("  1. Go to Redis terminal")
print("  2. Press Ctrl+C to STOP Redis")
print("  3. Come back here and press Enter")
print("="*55)
input("  Press Enter when Redis is stopped: ")

print("\n[2/3] Sending 5 requests with Redis DOWN...")
print("  (System should fall back to in-memory limiter)")
during = send_requests("during", 5)
print(f"\n  Result: {during}")
graceful = all(s in [200, 429] for s in during)
print(f"  Graceful degradation: {'✅ YES — system kept running!' if graceful else '❌ NO — system crashed'}")

# ── Phase 3: Recovery ─────────────────────────────
print("\n" + "="*55)
print("  ⚠️  ACTION REQUIRED:")
print("  1. Restart Redis: D:\\Redis\\redis-server.exe")
print("  2. Wait 3 seconds")
print("  3. Come back here and press Enter")
print("="*55)
input("  Press Enter when Redis is back: ")

time.sleep(2)  # give Redis time to fully start

print("\n[3/3] Sending 5 requests with Redis RESTORED...")
after = send_requests("after", 5)
print(f"\n  Result: {after}")
recovered = all(s in [200, 429] for s in after)
print(f"  Full recovery: {'✅ YES — Redis reconnected!' if recovered else '❌ NO — still issues'}")

# ── Final Verdict ─────────────────────────────────
print("\n" + "="*55)
print("  CHAOS TEST RESULTS")
print("="*55)
print(f"  Before chaos    : {before}")
print(f"  During chaos    : {during}")
print(f"  After recovery  : {after}")
print()
print(f"  Graceful degradation : {'✅ PASSED' if graceful else '❌ FAILED'}")
print(f"  Full recovery        : {'✅ PASSED' if recovered else '❌ FAILED'}")

if graceful and recovered:
    print("\n  🏆 SYSTEM IS PRODUCTION RESILIENT!")
    print("  Redis failure handled gracefully with zero downtime.")
else:
    print("\n  ⚠️  System needs improvement in fault tolerance.")
print("="*55)
