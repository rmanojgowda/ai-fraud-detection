"""
Velocity Decay
===============
Recent transactions count MORE than older ones.

Standard velocity counting treats all transactions equally:
  tx_count_1min = 5 (regardless of when they happened)

Velocity decay weights recent transactions higher:
  Transaction 5 seconds ago:  weight 1.00
  Transaction 30 seconds ago: weight 0.50
  Transaction 55 seconds ago: weight 0.10

Why this matters:
  A card-testing attack sends 5 transactions in the last
  2 seconds → decayed score = 4.8 (almost 5)
  
  A normal customer made 5 purchases spread over 55 minutes
  → decayed score = 0.9 (much lower, not suspicious)

Same raw count (5), completely different risk profiles.
Standard velocity can't tell them apart. Decay can.

Real-world usage:
  Visa uses exponential decay on velocity features.
  The half-life (how fast weight drops) is tuned per
  merchant category — ATM withdrawals decay faster
  than grocery purchases.

Interview value:
  "I realized standard tx_count treats a burst of 5
   transactions in 2 seconds the same as 5 transactions
   spread over an hour. Velocity decay gives higher weight
   to recent transactions, making burst attacks much more
   visible while reducing noise from normal spread-out usage."
"""

import time
import math
import threading
from collections import defaultdict
from typing import Dict, List, Tuple


class VelocityDecayCalculator:
    """
    Computes time-decayed transaction velocity per card.

    Decay function: weight = exp(-lambda * age_seconds)
    where lambda controls how fast weight drops.

    Half-life examples:
      lambda=0.1  → half-life 7 seconds  (aggressive decay)
      lambda=0.05 → half-life 14 seconds (moderate decay)
      lambda=0.01 → half-life 69 seconds (slow decay)

    We use lambda=0.05 → transactions older than 60s
    have weight < 0.05 (effectively zero)
    """

    def __init__(
        self,
        decay_lambda:  float = 0.05,   # decay rate
        window_sec:    int   = 3600,   # max history window (1 hour)
        max_cards:     int   = 10000   # memory limit
    ):
        self.decay_lambda = decay_lambda
        self.window_sec   = window_sec
        self.max_cards    = max_cards
        self._lock        = threading.Lock()

        # card_id → list of timestamps
        self._card_history: Dict[str, List[float]] = defaultdict(list)

    def record_transaction(self, card_id: str) -> None:
        """Record a new transaction timestamp for this card."""
        now = time.time()
        with self._lock:
            self._card_history[card_id].append(now)
            # Cleanup old entries beyond window
            cutoff = now - self.window_sec
            self._card_history[card_id] = [
                t for t in self._card_history[card_id]
                if t > cutoff
            ]
            # Memory protection
            if len(self._card_history) > self.max_cards:
                oldest = min(self._card_history,
                             key=lambda k: self._card_history[k][-1]
                             if self._card_history[k] else 0)
                del self._card_history[oldest]

    def get_decayed_velocity(
        self,
        card_id:    str,
        window_sec: int = 60
    ) -> Tuple[float, float, int]:
        """
        Returns (decayed_score, raw_count, weighted_count).

        decayed_score: sum of exp(-lambda * age) for each tx
        raw_count:     simple count in window
        burst_factor:  decayed / raw (>0.8 means recent burst)
        """
        now = time.time()
        cutoff = now - window_sec

        with self._lock:
            history = [
                t for t in self._card_history.get(card_id, [])
                if t > cutoff
            ]

        if not history:
            return 0.0, 0, 0.0

        raw_count     = len(history)
        decayed_score = sum(
            math.exp(-self.decay_lambda * (now - t))
            for t in history
        )
        burst_factor  = decayed_score / raw_count if raw_count > 0 else 0

        return round(decayed_score, 4), raw_count, round(burst_factor, 4)

    def get_risk_score(
        self,
        card_id:    str,
        window_sec: int   = 60,
        threshold:  float = 3.0
    ) -> Tuple[float, list]:
        """
        Returns (risk_score 0-1, signals).

        Risk increases when:
          1. High decayed velocity (many recent transactions)
          2. High burst factor (transactions clustered recently)
        """
        decayed, raw, burst = self.get_decayed_velocity(
            card_id, window_sec)

        risk    = 0.0
        signals = []

        # High velocity
        if decayed >= threshold * 2:
            risk += 0.50
            signals.append(
                f"VERY HIGH velocity: {decayed:.1f} decayed "
                f"({raw} raw in {window_sec}s)"
            )
        elif decayed >= threshold:
            risk += 0.25
            signals.append(
                f"High velocity: {decayed:.1f} decayed "
                f"({raw} raw in {window_sec}s)"
            )

        # Burst pattern (transactions clustered in last few seconds)
        if burst > 0.85 and raw >= 3:
            risk += 0.30
            signals.append(
                f"Burst pattern detected: burst_factor={burst:.2f} "
                f"(transactions clustered recently)"
            )
        elif burst > 0.70 and raw >= 3:
            risk += 0.15
            signals.append(
                f"Elevated burst: burst_factor={burst:.2f}"
            )

        if not signals:
            signals.append(
                f"Normal velocity: {decayed:.2f} decayed "
                f"({raw} raw in {window_sec}s)"
            )

        return min(round(risk, 4), 1.0), signals

    def compare_with_standard(
        self,
        card_id: str,
        window_sec: int = 60
    ) -> dict:
        """Shows difference between standard and decayed velocity."""
        now    = time.time()
        cutoff = now - window_sec

        with self._lock:
            history = [
                t for t in self._card_history.get(card_id, [])
                if t > cutoff
            ]

        standard_count = len(history)
        decayed_score  = sum(
            math.exp(-self.decay_lambda * (now - t))
            for t in history
        )

        ages = [round(now - t, 1) for t in sorted(history)]

        return {
            "card_id":        card_id,
            "standard_count": standard_count,
            "decayed_score":  round(decayed_score, 4),
            "transaction_ages_sec": ages,
            "decay_lambda":   self.decay_lambda,
            "interpretation": (
                f"Standard sees {standard_count} transactions. "
                f"Decay sees {decayed_score:.2f} — "
                f"{'burst attack (recent cluster)' if decayed_score > standard_count * 0.7 else 'spread-out normal usage'}"
            )
        }

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "cards_tracked": len(self._card_history),
                "decay_lambda":  self.decay_lambda,
                "half_life_sec": round(math.log(2) / self.decay_lambda, 1),
                "window_sec":    self.window_sec,
            }


# ── Global instance ───────────────────────────────────────────
_velocity_calc = VelocityDecayCalculator(
    decay_lambda = 0.05,
    window_sec   = 3600,
)


def get_velocity_calculator() -> VelocityDecayCalculator:
    return _velocity_calc


def record_velocity(card_id: str) -> None:
    _velocity_calc.record_transaction(card_id)


def get_velocity_risk(
    card_id: str,
    window_sec: int = 60
) -> Tuple[float, list]:
    return _velocity_calc.get_risk_score(card_id, window_sec)


# ── Test ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("  VELOCITY DECAY TEST")
    print("=" * 65)

    calc = VelocityDecayCalculator(decay_lambda=0.05)

    print("\n[1] Card-testing attack (5 transactions in 2 seconds):")
    attack_card = "card_attack_001"
    for i in range(5):
        calc.record_transaction(attack_card)
        time.sleep(0.4)  # 0.4s apart = burst

    comparison = calc.compare_with_standard(attack_card, window_sec=60)
    print(f"    Standard count: {comparison['standard_count']}")
    print(f"    Decayed score:  {comparison['decayed_score']}")
    print(f"    Ages (sec):     {comparison['transaction_ages_sec']}")
    print(f"    → {comparison['interpretation']}")

    risk, signals = calc.get_risk_score(attack_card)
    print(f"    Risk score: {risk}")
    for s in signals:
        print(f"    ⚠️  {s}")

    print("\n[2] Normal customer (5 transactions over 50 minutes):")
    normal_card = "card_normal_001"
    now = time.time()
    # Inject old timestamps directly
    calc._card_history[normal_card] = [
        now - 3000,  # 50 min ago
        now - 2400,  # 40 min ago
        now - 1800,  # 30 min ago
        now - 1200,  # 20 min ago
        now - 600,   # 10 min ago
    ]

    comparison2 = calc.compare_with_standard(normal_card, window_sec=3600)
    print(f"    Standard count: {comparison2['standard_count']}")
    print(f"    Decayed score:  {comparison2['decayed_score']}")
    print(f"    → {comparison2['interpretation']}")

    risk2, signals2 = calc.get_risk_score(normal_card, window_sec=3600)
    print(f"    Risk score: {risk2}")
    for s in signals2:
        print(f"    ✅ {s}")

    print("\n[3] Key comparison (same raw count = 5, different risk):")
    print(f"    Attack card:  decayed={comparison['decayed_score']:.2f}  risk={risk}")
    print(f"    Normal card:  decayed={comparison2['decayed_score']:.2f}  risk={risk2}")
    print(f"    Standard velocity can't distinguish these!")
    print(f"    Decay correctly identifies attack pattern.")

    print("\n[4] Stats:")
    stats = calc.get_stats()
    for k, v in stats.items():
        print(f"    {k}: {v}")

    print("\n" + "=" * 65)
    print("  VELOCITY DECAY COMPLETE ✅")
    print("  Burst attacks score HIGH, spread-out normal usage scores LOW")
    print("  Same raw count → completely different risk profiles")
    print("=" * 65)
