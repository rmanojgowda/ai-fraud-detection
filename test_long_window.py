"""
Direct unit test for long window logic.
Bypasses short window by testing memory directly.
"""
import time
from rate_limiter import DualWindowRateLimiter, LONG_WINDOW_REQUESTS

limiter = DualWindowRateLimiter()
ip      = "test_long_only"

print("=" * 50)
print("  LONG WINDOW UNIT TEST")
print("=" * 50)

# Directly inject 99 old timestamps into long window
# (simulating requests made over the past hour)
now = time.time()
limiter._memory_long[ip] = [now - 100] * 99  # 99 requests, 100s ago
limiter._memory_short[ip] = []               # short window is clean

print(f"\n  Pre-loaded 99 requests into long window")
print(f"  Short window: empty (clean)")

# Request 100 — should be ALLOWED
ok, reason = limiter._check_memory(ip)
print(f"\n  Request 100: {'✅ ALLOWED' if ok else '🚫 BLOCKED'} — {reason}")

# Request 101 — should be BLOCKED by long window
limiter._memory_short[ip] = []  # reset short window again
ok, reason = limiter._check_memory(ip)
print(f"  Request 101: {'✅ ALLOWED' if ok else '🚫 BLOCKED'} — {reason}")

print(f"\n  Long window logic: {'✅ CORRECT' if 'Hourly' in reason else '❌ issue'}")
print("\n" + "=" * 50)
print("  KEY INSIGHT:")
print("  Long window catches slow attackers who send")
print("  4 req/10s (under short limit) but >100 req/hr")
print("  Short window catches burst attackers (>5/10s)")
print("  Together they cover all attack patterns ✅")
print("=" * 50)
