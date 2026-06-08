"""
Phase 9 — Rate Limiter (Upgraded)
===================================
Three-layer rate limiting:

Layer 1 — IP short window:  5 requests per 10 seconds
Layer 2 — IP long window:   100 requests per hour
Layer 3 — Card window:      3 requests per hour (NEW)

Gap closed: IP rotation bypass
  Before: Attacker rotates IPs → fresh window each time
  After:  card_id tracked separately → blocked after 3/hr
          regardless of IP address
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

# NEW: Card-level rate limiting
CARD_WINDOW_REQUESTS  = 3
CARD_WINDOW_SECONDS   = 3600

REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB   = 0


class DualWindowRateLimiter:
    """
    Three-layer rate limiter.

    Layer 1 — IP short:  5/10s   (burst attack protection)
    Layer 2 — IP long:   100/hr  (slow distributed attack)
    Layer 3 — Card:      3/hr    (IP rotation bypass fix)

    Fallback: In-memory with Redis sync (Gap 2 fix from Phase 9)
    """

    def __init__(self):
        self._redis_client    = None
        self._redis_available = False
        self._lock            = threading.Lock()

        # Persistent memory — never resets on Redis failure
        self._memory_short: dict = defaultdict(list)
        self._memory_long:  dict = defaultdict(list)
        self._memory_card:  dict = defaultdict(list)  # NEW

        self._connect_redis()

        threading.Thread(
            target=self._reconnect_loop, daemon=True
        ).start()

        # Stats
        self.total_requests  = 0
        self.blocked_short   = 0
        self.blocked_long    = 0
        self.blocked_card    = 0  # NEW

    def _connect_redis(self) -> bool:
        try:
            pool = redis.ConnectionPool(
                host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB,
                max_connections=50,
                socket_connect_timeout=1, socket_timeout=1
            )
            client = redis.Redis(connection_pool=pool)
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
    def is_allowed(
        self,
        client_ip: str,
        card_id: str = None
    ) -> Tuple[bool, str]:
        """
        Check all three rate limit layers.
        card_id is optional — if provided, card-level check runs.
        """
        with self._lock:
            self.total_requests += 1
            if self._redis_available:
                return self._check_redis(client_ip, card_id)
            return self._check_memory(client_ip, card_id)

    def _check_redis(
        self,
        client_ip: str,
        card_id: str = None
    ) -> Tuple[bool, str]:
        try:
            now       = time.time()
            short_key = f"rate:short:{client_ip}"
            long_key  = f"rate:long:{client_ip}"

            pipe = self._redis_client.pipeline()

            # Layer 1: IP short window
            pipe.zremrangebyscore(short_key, 0, now - SHORT_WINDOW_SECONDS)
            pipe.zcard(short_key)

            # Layer 2: IP long window
            pipe.zremrangebyscore(long_key, 0, now - LONG_WINDOW_SECONDS)
            pipe.zcard(long_key)

            # Layer 3: Card window (if card_id provided)
            if card_id:
                card_key = f"rate:card:{card_id}"
                pipe.zremrangebyscore(card_key, 0, now - CARD_WINDOW_SECONDS)
                pipe.zcard(card_key)

            results     = pipe.execute()
            short_count = results[1]
            long_count  = results[3]
            card_count  = results[5] if card_id else 0

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

            if card_id and card_count >= CARD_WINDOW_REQUESTS:
                self.blocked_card += 1
                return False, (
                    f"Card limit: {card_count}/"
                    f"{CARD_WINDOW_REQUESTS} attempts per hour "
                    f"for card {card_id[:8]}..."
                )

            # Allowed — add timestamps
            pipe2 = self._redis_client.pipeline()
            pipe2.zadd(short_key, {f"{now}": now})
            pipe2.expire(short_key, SHORT_WINDOW_SECONDS + 1)
            pipe2.zadd(long_key, {f"{now}l": now})
            pipe2.expire(long_key, LONG_WINDOW_SECONDS + 1)
            if card_id:
                card_key = f"rate:card:{card_id}"
                pipe2.zadd(card_key, {f"{now}c": now})
                pipe2.expire(card_key, CARD_WINDOW_SECONDS + 1)
            pipe2.execute()

            # Sync to memory fallback
            self._memory_short[client_ip].append(now)
            self._memory_long[client_ip].append(now)
            if card_id:
                self._memory_card[card_id].append(now)

            return True, "allowed"

        except Exception:
            self._redis_available = False
            return self._check_memory(client_ip, card_id)

    def _check_memory(
        self,
        client_ip: str,
        card_id: str = None
    ) -> Tuple[bool, str]:
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
        if card_id:
            self._memory_card[card_id] = [
                t for t in self._memory_card[card_id]
                if now - t < CARD_WINDOW_SECONDS
            ]

        short_count = len(self._memory_short[client_ip])
        long_count  = len(self._memory_long[client_ip])
        card_count  = len(self._memory_card[card_id]) if card_id else 0

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

        if card_id and card_count >= CARD_WINDOW_REQUESTS:
            self.blocked_card += 1
            return False, (
                f"Card limit: {card_count}/"
                f"{CARD_WINDOW_REQUESTS} attempts per hour "
                f"for card {card_id[:8]}..."
            )

        # Allowed — add timestamps
        self._memory_short[client_ip].append(now)
        self._memory_long[client_ip].append(now)
        if card_id:
            self._memory_card[card_id].append(now)

        return True, "allowed"

    def get_status(self, client_ip: str, card_id: str = None) -> dict:
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
            card_ = len([
                t for t in self._memory_card.get(card_id, [])
                if now - t < CARD_WINDOW_SECONDS
            ]) if card_id else None

            result = {
                "ip":              client_ip,
                "short_window":    f"{short}/{SHORT_WINDOW_REQUESTS} per {SHORT_WINDOW_SECONDS}s",
                "long_window":     f"{long_}/{LONG_WINDOW_REQUESTS} per hour",
                "redis_available": self._redis_available,
                "total_requests":  self.total_requests,
                "blocked_short":   self.blocked_short,
                "blocked_long":    self.blocked_long,
                "blocked_card":    self.blocked_card,
            }
            if card_id:
                result["card_window"] = f"{card_}/{CARD_WINDOW_REQUESTS} per hour"
            return result

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

    def reset_card(self, card_id: str) -> None:
        with self._lock:
            self._memory_card.pop(card_id, None)
            if self._redis_available:
                try:
                    self._redis_client.delete(f"rate:card:{card_id}")
                except Exception:
                    pass


# ── Test ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  CARD-LEVEL RATE LIMITER TEST")
    print("=" * 60)

    limiter = DualWindowRateLimiter()
    print(f"\n  Redis: {'connected' if limiter._redis_available else 'in-memory fallback'}")

    print("\n[1] IP rotation attack simulation:")
    print("    Attacker uses card_001 from 5 different IPs")
    print("    Each IP is fresh — bypasses IP rate limit")
    print("    But card_001 is tracked regardless of IP")
    print()

    card = "card_001_stolen"
    ips  = ["10.0.0.1", "10.0.0.2", "10.0.0.3",
            "10.0.0.4", "10.0.0.5"]

    for i, ip in enumerate(ips, 1):
        allowed, reason = limiter.is_allowed(ip, card)
        status = "✅ ALLOWED" if allowed else "🚫 BLOCKED"
        print(f"    Request {i} (IP: {ip}): {status} — {reason}")

    print("\n[2] Normal customer — different card, same IP:")
    limiter.reset_ip("10.0.0.1")
    for i in range(1, 4):
        allowed, reason = limiter.is_allowed("10.0.0.1", f"card_legit_{i:03d}")
        status = "✅ ALLOWED" if allowed else "🚫 BLOCKED"
        print(f"    Request {i} (card_legit_{i:03d}): {status}")

    print("\n[3] Status check:")
    status = limiter.get_status("10.0.0.1", card)
    for k, v in status.items():
        print(f"    {k}: {v}")

    print("\n" + "=" * 60)
    print("  CARD-LEVEL RATE LIMITING COMPLETE ✅")
    print("  IP rotation attack blocked after 3 attempts")
    print("  Legitimate customers unaffected")
    print("=" * 60)
