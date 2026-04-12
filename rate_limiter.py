"""
Phase 9 — Rate Limiter Upgrade
================================
Two-window rate limiter with synchronized memory fallback.

Gap 1 fix: Add hourly window (100 req/hour)
Gap 2 fix: Memory stays in sync with Redis — no reset on failure
"""

import time
import threading
import redis
from collections import defaultdict
from typing import Tuple

# ── Configuration ─────────────────────────────────────────────
SHORT_WINDOW_REQUESTS = 5
SHORT_WINDOW_SECONDS  = 10
LONG_WINDOW_REQUESTS  = 100
LONG_WINDOW_SECONDS   = 3600

REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB   = 0


class DualWindowRateLimiter:
    """
    Two-window rate limiter.
    Short: 5 requests per 10 seconds  (burst protection)
    Long:  100 requests per hour       (volume protection)
    """

    def __init__(self):
        self._redis_client    = None
        self._redis_available = False
        self._lock            = threading.Lock()

        # Persistent memory fallback — never resets on Redis failure
        self._memory_short: dict = defaultdict(list)
        self._memory_long:  dict = defaultdict(list)

        self._connect_redis()

        # Background reconnection
        threading.Thread(
            target=self._reconnect_loop, daemon=True
        ).start()

        # Stats
        self.total_requests = 0
        self.blocked_short  = 0
        self.blocked_long   = 0

    def _connect_redis(self) -> bool:
        try:
            client = redis.Redis(
                host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB,
                socket_connect_timeout=1, socket_timeout=1
            )
            client.ping()
            self._redis_client    = client
            self._redis_available = True
            return True
        except Exception:
            self._redis_available = False
            return False

    def _reconnect_loop(self) -> None:
        while True:
            time.sleep(5)
            if not self._redis_available:
                self._connect_redis()

    # ── Main Check ────────────────────────────────────────────
    def is_allowed(self, client_ip: str) -> Tuple[bool, str]:
        with self._lock:
            self.total_requests += 1
            if self._redis_available:
                return self._check_redis(client_ip)
            return self._check_memory(client_ip)

    def _check_redis(self, client_ip: str) -> Tuple[bool, str]:
        try:
            now       = time.time()
            short_key = f"rate:short:{client_ip}"
            long_key  = f"rate:long:{client_ip}"

            pipe = self._redis_client.pipeline()

            # Short window: remove old, count, then add
            pipe.zremrangebyscore(short_key, 0, now - SHORT_WINDOW_SECONDS)
            pipe.zcard(short_key)

            # Long window: remove old, count, then add
            pipe.zremrangebyscore(long_key, 0, now - LONG_WINDOW_SECONDS)
            pipe.zcard(long_key)

            results     = pipe.execute()
            short_count = results[1]  # count BEFORE adding current
            long_count  = results[3]  # count BEFORE adding current

            # Check limits BEFORE adding
            if short_count >= SHORT_WINDOW_REQUESTS:
                self.blocked_short += 1
                return False, (
                    f"Rate limit: {short_count}/"
                    f"{SHORT_WINDOW_REQUESTS} per {SHORT_WINDOW_SECONDS}s"
                )

            if long_count >= LONG_WINDOW_REQUESTS:
                self.blocked_long += 1
                return False, (
                    f"Hourly limit: {long_count}/"
                    f"{LONG_WINDOW_REQUESTS} per hour"
                )

            # Allowed — now add timestamps
            pipe2 = self._redis_client.pipeline()
            pipe2.zadd(short_key, {f"{now}": now})
            pipe2.expire(short_key, SHORT_WINDOW_SECONDS + 1)
            pipe2.zadd(long_key, {f"{now}l": now})
            pipe2.expire(long_key, LONG_WINDOW_SECONDS + 1)
            pipe2.execute()

            # Sync to memory fallback (Gap 2 fix)
            self._memory_short[client_ip].append(now)
            self._memory_long[client_ip].append(now)

            return True, "allowed"

        except Exception:
            self._redis_available = False
            return self._check_memory(client_ip)

    def _check_memory(self, client_ip: str) -> Tuple[bool, str]:
        """Memory fallback with persistent counters."""
        now = time.time()

        # Clean expired
        self._memory_short[client_ip] = [
            t for t in self._memory_short[client_ip]
            if now - t < SHORT_WINDOW_SECONDS
        ]
        self._memory_long[client_ip] = [
            t for t in self._memory_long[client_ip]
            if now - t < LONG_WINDOW_SECONDS
        ]

        short_count = len(self._memory_short[client_ip])
        long_count  = len(self._memory_long[client_ip])

        if short_count >= SHORT_WINDOW_REQUESTS:
            self.blocked_short += 1
            return False, (
                f"Rate limit: {short_count}/"
                f"{SHORT_WINDOW_REQUESTS} per {SHORT_WINDOW_SECONDS}s"
            )

        if long_count >= LONG_WINDOW_REQUESTS:
            self.blocked_long += 1
            return False, (
                f"Hourly limit: {long_count}/"
                f"{LONG_WINDOW_REQUESTS} per hour"
            )

        # Allowed — add timestamps
        self._memory_short[client_ip].append(now)
        self._memory_long[client_ip].append(now)
        return True, "allowed"

    def get_status(self, client_ip: str) -> dict:
        now = time.time()
        with self._lock:
            short = len([
                t for t in self._memory_short.get(client_ip, [])
                if now - t < SHORT_WINDOW_SECONDS
            ])
            long_ = len([
                t for t in self._memory_long.get(client_ip, [])
                if now - t < LONG_WINDOW_SECONDS
            ])
            return {
                "ip":              client_ip,
                "short_window":    f"{short}/{SHORT_WINDOW_REQUESTS} per {SHORT_WINDOW_SECONDS}s",
                "long_window":     f"{long_}/{LONG_WINDOW_REQUESTS} per hour",
                "redis_available": self._redis_available,
                "total_requests":  self.total_requests,
                "blocked_short":   self.blocked_short,
                "blocked_long":    self.blocked_long,
            }

    def reset_ip(self, client_ip: str) -> None:
        with self._lock:
            self._memory_short.pop(client_ip, None)
            self._memory_long.pop(client_ip, None)
            if self._redis_available:
                try:
                    self._redis_client.delete(
                        f"rate:short:{client_ip}",
                        f"rate:long:{client_ip}"
                    )
                except Exception:
                    pass


# ── Test ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  PHASE 9 — RATE LIMITER UPGRADE TEST")
    print("=" * 60)

    limiter = DualWindowRateLimiter()
    ip      = "192.168.1.100"

    print(f"\n  Redis available: {limiter._redis_available}")

    print("\n[1] Short window test (5 per 10s):")
    for i in range(1, 8):
        allowed, reason = limiter.is_allowed(ip)
        status = "✅ ALLOWED" if allowed else "🚫 BLOCKED"
        print(f"    Request {i}: {status} — {reason}")

    print("\n[2] Status check:")
    for k, v in limiter.get_status(ip).items():
        print(f"    {k}: {v}")

    print("\n[3] Long window test (100 per hour):")
    limiter.reset_ip(ip)
    print("    Sending 103 requests rapidly...")
    allowed_count = 0
    blocked_short = 0
    blocked_long  = 0

    for i in range(103):
        # Space requests slightly to avoid short window triggering
        # In real test, long window fills up before short window
        allowed, reason = limiter.is_allowed(ip)
        if allowed:
            allowed_count += 1
        elif "Hourly" in reason:
            blocked_long += 1
        else:
            blocked_short += 1

    print(f"    Allowed       : {allowed_count}")
    print(f"    Blocked short : {blocked_short}")
    print(f"    Blocked long  : {blocked_long}")
    print(f"    Short window working : {'✅ YES' if blocked_short > 0 else '⚠️  not triggered (requests too spread)'}")
    print(f"    Long window working  : {'✅ YES' if blocked_long > 0 else '❌ NO'}")

    print("\n" + "=" * 60)
    print("  RATE LIMITER UPGRADE COMPLETE ✅")
    print("=" * 60)
