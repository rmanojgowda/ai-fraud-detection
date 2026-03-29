import threading
import requests
import time
import json
import os
from datetime import datetime

API_URL = "http://127.0.0.1:8000/fraud/check"

TRANSACTIONS = [
    {**{f"V{i}": 0.0 for i in range(1,29)},
     "Amount": 50.0,  "tx_count_1min": 1,
     "tx_count_10min": 2,  "tx_count_60min": 5,
     "hour": 14, "card_id": "card_normal",
     "merchant_id": "M01", "ip": "10.0.0.1"},

    {**{f"V{i}": 0.0 for i in range(1,29)},
     "Amount": 2500.0, "tx_count_1min": 4,
     "tx_count_10min": 12, "tx_count_60min": 20,
     "hour": 2, "card_id": "card_suspicious",
     "merchant_id": "M99", "ip": "192.168.1.1"},

    {**{f"V{i}": 0.0 for i in range(1,29)},
     "Amount": 100.0, "tx_count_1min": 8,
     "tx_count_10min": 30, "tx_count_60min": 60,
     "hour": 3, "card_id": "card_velocity",
     "merchant_id": "M50", "ip": "172.16.0.1"},
]

# ── Shared Results ────────────────────────────────
class TestResults:
    def __init__(self):
        self.success    = 0
        self.rate_limit = 0
        self.errors     = 0
        self.latencies  = []
        self.lock       = threading.Lock()

    def add(self, status: int, latency: float):
        with self.lock:
            self.latencies.append(latency)
            if status == 200:
                self.success    += 1
            elif status == 429:
                self.rate_limit += 1
            else:
                self.errors     += 1

    def add_error(self):
        with self.lock:
            self.errors += 1

# ── Single Request ────────────────────────────────
def send_request(results: TestResults, tx_index: int):
    payload = TRANSACTIONS[tx_index % len(TRANSACTIONS)]
    start   = time.time()
    try:
        res     = requests.post(API_URL, json=payload, timeout=10)
        latency = (time.time() - start) * 1000
        results.add(res.status_code, latency)
    except Exception:
        results.add_error()

# ── Run One Test Scenario ─────────────────────────
def run_scenario(name: str, total: int, concurrent: int) -> dict:
    print(f"\n  ▶ {name}")
    print(f"    Requests: {total} | Concurrency: {concurrent}")

    results    = TestResults()
    start_time = time.time()
    threads    = []

    for i in range(total):
        t = threading.Thread(target=send_request, args=(results, i))
        threads.append(t)
        t.start()

        # Throttle to maintain concurrency level
        active = sum(1 for t in threads if t.is_alive())
        while active >= concurrent:
            time.sleep(0.005)
            active = sum(1 for t in threads if t.is_alive())

    for t in threads:
        t.join()

    total_time = time.time() - start_time
    latencies  = sorted(results.latencies)
    n          = len(latencies)

    if n == 0:
        print("    ❌ No successful requests!")
        return {}

    metrics = {
        "name":          name,
        "total":         total,
        "concurrent":    concurrent,
        "success":       results.success,
        "rate_limited":  results.rate_limit,
        "errors":        results.errors,
        "total_time_s":  round(total_time, 2),
        "rps":           round(total / total_time, 1),
        "rpm":           round(total / total_time * 60, 0),
        "lat_min":       round(latencies[0], 1),
        "lat_avg":       round(sum(latencies)/n, 1),
        "lat_p50":       round(latencies[int(n*0.50)], 1),
        "lat_p95":       round(latencies[min(int(n*0.95), n-1)], 1),
        "lat_p99":       round(latencies[min(int(n*0.99), n-1)], 1),
        "lat_max":       round(latencies[-1], 1),
        "error_rate":    round(results.errors / total * 100, 2),
    }

    print(f"    ✅ Success: {metrics['success']} | "
          f"🚫 Rate: {metrics['rate_limited']} | "
          f"❌ Errors: {metrics['errors']}")
    print(f"    ⚡ {metrics['rpm']:.0f} req/min | "
          f"P50: {metrics['lat_p50']}ms | "
          f"P95: {metrics['lat_p95']}ms | "
          f"P99: {metrics['lat_p99']}ms")

    return metrics

# ── Chaos Test: Kill Redis Mid-Test ───────────────
def chaos_test_redis():
    print("\n" + "="*55)
    print("  🔥 CHAOS TEST — Redis Failure Simulation")
    print("="*55)
    print("  Sending 5 requests (Redis up)...")

    results_before = []
    for i in range(5):
        try:
            res = requests.post(API_URL,
                json=TRANSACTIONS[0], timeout=5)
            results_before.append(res.status_code)
            time.sleep(0.5)
        except:
            results_before.append("error")

    print(f"  Before chaos: {results_before}")
    print("\n  ⚠️  Now manually stop Redis (Ctrl+C in Redis terminal)")
    print("  Then press Enter here to continue the test...")
    input()

    print("  Sending 5 requests (Redis down)...")
    results_after = []
    for i in range(5):
        try:
            res = requests.post(API_URL,
                json=TRANSACTIONS[0], timeout=5)
            results_after.append(res.status_code)
            time.sleep(0.5)
        except Exception as e:
            results_after.append(f"error: {str(e)[:20]}")

    print(f"  During chaos: {results_after}")

    print("\n  ▶ Start Redis again, then press Enter...")
    input()

    print("  Sending 5 requests (Redis restored)...")
    results_recovery = []
    for i in range(5):
        try:
            res = requests.post(API_URL,
                json=TRANSACTIONS[0], timeout=5)
            results_recovery.append(res.status_code)
            time.sleep(0.5)
        except:
            results_recovery.append("error")

    print(f"  After recovery: {results_recovery}")

    all_ok   = all(s == 200 or s == 429 for s in results_after)
    recovery = all(s == 200 or s == 429 for s in results_recovery)

    print(f"\n  Graceful degradation: {'✅ YES' if all_ok else '❌ NO'}")
    print(f"  Full recovery:        {'✅ YES' if recovery else '❌ NO'}")

# ── Bottleneck Analysis ───────────────────────────
def bottleneck_analysis(scenarios: list):
    print("\n" + "="*55)
    print("  🔍 BOTTLENECK ANALYSIS")
    print("="*55)

    if len(scenarios) < 2:
        return

    # Find where P95 degrades
    print("\n  Concurrency vs Latency (P95):")
    for s in scenarios:
        if not s:
            continue
        bar_len = int(s['lat_p95'] / 10)
        bar     = "█" * min(bar_len, 50)
        print(f"  {s['concurrent']:3d} threads: {bar} {s['lat_p95']}ms P95")

    # Find where errors start
    error_scenarios = [s for s in scenarios if s and s['errors'] > 0]
    if error_scenarios:
        first_error = error_scenarios[0]
        print(f"\n  ⚠️  Errors start at: {first_error['concurrent']} concurrent threads")
        print(f"     Error rate: {first_error['error_rate']}%")
    else:
        print(f"\n  ✅ No errors across all concurrency levels")

    # Throughput peak
    peak = max((s for s in scenarios if s), key=lambda x: x['rpm'])
    print(f"\n  🏆 Peak throughput: {peak['rpm']:.0f} req/min "
          f"at {peak['concurrent']} concurrent threads")

    # P95 threshold
    fast = [s for s in scenarios if s and s['lat_p95'] < 100]
    if fast:
        best = max(fast, key=lambda x: x['concurrent'])
        print(f"  ⚡ Max concurrency under 100ms P95: {best['concurrent']} threads")

# ── Main ──────────────────────────────────────────
def main():
    print("=" * 55)
    print("  FRAUD DETECTION — SCALABILITY & STRESS TEST")
    print("=" * 55)
    print(f"  Target : {API_URL}")
    print(f"  Time   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check server is up
    try:
        r = requests.get("http://127.0.0.1:8000/health", timeout=3)
        health = r.json()
        print(f"  Server : ✅ {health.get('version')} | "
              f"Redis: {health.get('redis')}")
    except:
        print("  ❌ Server not reachable. Start uvicorn first.")
        return

    all_scenarios = []

    # ── Phase 1: Baseline ─────────────────────────
    print("\n" + "="*55)
    print("  PHASE 1 — BASELINE (Low Load)")
    print("="*55)
    all_scenarios.append(run_scenario("Baseline",      50,  2))
    time.sleep(12)  # wait for rate limit window to reset
    all_scenarios.append(run_scenario("Light Load",    50,  5))
    time.sleep(12)

    # ── Phase 2: Normal Load ──────────────────────
    print("\n" + "="*55)
    print("  PHASE 2 — NORMAL LOAD")
    print("="*55)
    all_scenarios.append(run_scenario("Normal Load",   100, 10))
    time.sleep(12)
    all_scenarios.append(run_scenario("Medium Load",   100, 20))
    time.sleep(12)

    # ── Phase 3: Stress Test ──────────────────────
    print("\n" + "="*55)
    print("  PHASE 3 — STRESS TEST (High Load)")
    print("="*55)
    all_scenarios.append(run_scenario("Heavy Load",    200, 30))
    time.sleep(12)
    all_scenarios.append(run_scenario("Stress Test",   200, 50))
    time.sleep(12)

    # ── Phase 4: Breaking Point ───────────────────
    print("\n" + "="*55)
    print("  PHASE 4 — BREAKING POINT")
    print("="*55)
    all_scenarios.append(run_scenario("Peak Load",     300, 75))
    time.sleep(12)
    all_scenarios.append(run_scenario("Breaking Point",300, 100))
    time.sleep(12)

    # ── Bottleneck Analysis ───────────────────────
    bottleneck_analysis(all_scenarios)

    # ── Final Summary ─────────────────────────────
    print("\n" + "="*55)
    print("  FINAL RESULTS SUMMARY")
    print("="*55)
    print(f"\n  {'Scenario':<20} {'RPM':>8} {'P95ms':>8} "
          f"{'Errors':>8} {'RateBlk':>8}")
    print("  " + "-"*52)
    for s in all_scenarios:
        if not s:
            continue
        print(f"  {s['name']:<20} {s['rpm']:>8.0f} "
              f"{s['lat_p95']:>8.1f} "
              f"{s['errors']:>8} "
              f"{s['rate_limited']:>8}")

    # ── Save Report ───────────────────────────────
    os.makedirs("logs", exist_ok=True)
    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "scenarios": all_scenarios
    }
    with open("logs/scalability_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n  💾 Full report: logs/scalability_report.json")
    print("="*55)

    # ── Optional Chaos Test ───────────────────────
    print("\n  Run chaos test (Redis failure simulation)? (y/n): ", end="")
    choice = input().strip().lower()
    if choice == 'y':
        chaos_test_redis()

if __name__ == "__main__":
    main()
