"""
A/B Testing Framework
======================
Safely deploy new model versions by splitting traffic.

How it works:
  Every request gets assigned to Group A or Group B
  based on a hash of the card_id (deterministic — same
  card always goes to same group, preventing flip-flopping)

  Group A: Current production model (control)
  Group B: New challenger model (treatment)

  We track metrics per group and decide winner after
  sufficient traffic (default: 1000 requests per group)

Real-world usage:
  Visa/Stripe use this to deploy new fraud models
  without risking full rollout of a worse model
"""

import hashlib
import time
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GroupMetrics:
    """Tracks metrics for one A/B group."""
    name:             str
    requests:         int   = 0
    approvals:        int   = 0
    blocks:           int   = 0
    step_ups:         int   = 0
    total_latency_ms: float = 0.0
    fraud_amounts:    list  = field(default_factory=list)
    scores:           list  = field(default_factory=list)

    @property
    def fraud_rate(self) -> float:
        return self.blocks / max(self.requests, 1) * 100

    @property
    def false_alarm_estimate(self) -> float:
        """Approximation — blocks that are likely legitimate."""
        return self.step_ups / max(self.requests, 1) * 100

    @property
    def avg_latency(self) -> float:
        return self.total_latency_ms / max(self.requests, 1)

    @property
    def avg_score(self) -> float:
        return sum(self.scores) / max(len(self.scores), 1)

    def record(self, decision: str, latency_ms: float, score: float):
        self.requests         += 1
        self.total_latency_ms += latency_ms
        self.scores.append(score)
        if decision == "APPROVE":
            self.approvals += 1
        elif decision == "STEP_UP_AUTH":
            self.step_ups  += 1
        else:
            self.blocks    += 1

    def to_dict(self) -> dict:
        return {
            "name":               self.name,
            "requests":           self.requests,
            "fraud_rate_pct":     round(self.fraud_rate, 3),
            "step_up_rate_pct":   round(self.false_alarm_estimate, 3),
            "avg_latency_ms":     round(self.avg_latency, 2),
            "avg_score":          round(self.avg_score, 4),
            "approvals":          self.approvals,
            "blocks":             self.blocks,
            "step_ups":           self.step_ups,
        }


class ABTestingFramework:
    """
    Traffic-splitting A/B test framework.

    Key design decisions:
      1. Deterministic assignment: hash(card_id) → always same group
         Why: Prevents the same card from seeing different decisions
              on repeat transactions (consistency matters for fraud)

      2. Configurable split: default 50/50, can be 90/10 for safety
         Why: New models start at 10% traffic, scale up if metrics improve

      3. Statistical significance check: need min_requests before deciding
         Why: Small samples have high variance — wait for enough data

      4. Metrics tracked: fraud_rate, latency, step_up_rate
         Why: These three cover accuracy, performance, and customer impact
    """

    def __init__(
        self,
        split_pct:    float = 50.0,
        min_requests: int   = 100,
        experiment_id: str  = "exp_001"
    ):
        self.split_pct     = split_pct      # % traffic to Group B
        self.min_requests  = min_requests
        self.experiment_id = experiment_id
        self.start_time    = time.time()
        self.active        = True

        self.group_a = GroupMetrics(name="A_control")
        self.group_b = GroupMetrics(name="B_challenger")

        self._log_file = f"logs/ab_test_{experiment_id}.jsonl"
        os.makedirs("logs", exist_ok=True)

    def assign_group(self, card_id: str) -> str:
        """
        Deterministically assigns card to group A or B.

        Uses MD5 hash of card_id → consistent per card.
        hash(card_id) % 100 < split_pct → Group B
        Otherwise → Group A

        Example:
          card_001 → hash → 23 → 23 < 50 → Group B
          card_002 → hash → 67 → 67 >= 50 → Group A
          card_001 → hash → 23 → ALWAYS Group B (deterministic)
        """
        hash_val = int(hashlib.md5(card_id.encode()).hexdigest(), 16) % 100
        return "B" if hash_val < self.split_pct else "A"

    def record_result(
        self,
        card_id:    str,
        group:      str,
        decision:   str,
        latency_ms: float,
        score:      float
    ) -> None:
        """Records transaction result for the assigned group."""
        if group == "A":
            self.group_a.record(decision, latency_ms, score)
        else:
            self.group_b.record(decision, latency_ms, score)

        # Log to file for analysis
        log_entry = {
            "timestamp":     time.time(),
            "experiment_id": self.experiment_id,
            "card_id":       card_id[:8] + "...",  # anonymized
            "group":         group,
            "decision":      decision,
            "score":         round(score, 4),
            "latency_ms":    round(latency_ms, 2),
        }
        try:
            with open(self._log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception:
            pass

    def get_results(self) -> dict:
        """Returns current A/B test results with winner analysis."""
        a = self.group_a.to_dict()
        b = self.group_b.to_dict()

        # Determine winner
        winner        = None
        winner_reason = []
        sufficient    = (
            self.group_a.requests >= self.min_requests and
            self.group_b.requests >= self.min_requests
        )

        if sufficient:
            a_score = 0
            b_score = 0

            # Compare fraud detection rate
            if b["fraud_rate_pct"] > a["fraud_rate_pct"] * 1.05:
                b_score += 2
                winner_reason.append(
                    f"B detects more fraud: {b['fraud_rate_pct']:.2f}% vs {a['fraud_rate_pct']:.2f}%"
                )
            elif a["fraud_rate_pct"] > b["fraud_rate_pct"] * 1.05:
                a_score += 2
                winner_reason.append(
                    f"A detects more fraud: {a['fraud_rate_pct']:.2f}% vs {b['fraud_rate_pct']:.2f}%"
                )

            # Compare latency
            if b["avg_latency_ms"] < a["avg_latency_ms"] * 0.95:
                b_score += 1
                winner_reason.append(
                    f"B is faster: {b['avg_latency_ms']:.1f}ms vs {a['avg_latency_ms']:.1f}ms"
                )
            elif a["avg_latency_ms"] < b["avg_latency_ms"] * 0.95:
                a_score += 1
                winner_reason.append(
                    f"A is faster: {a['avg_latency_ms']:.1f}ms vs {b['avg_latency_ms']:.1f}ms"
                )

            # Compare step-up rate (proxy for false alarms)
            if b["step_up_rate_pct"] < a["step_up_rate_pct"] * 0.95:
                b_score += 1
                winner_reason.append(
                    f"B has fewer false alarms: {b['step_up_rate_pct']:.2f}% vs {a['step_up_rate_pct']:.2f}%"
                )
            elif a["step_up_rate_pct"] < b["step_up_rate_pct"] * 0.95:
                a_score += 1
                winner_reason.append(
                    f"A has fewer false alarms: {a['step_up_rate_pct']:.2f}% vs {b['step_up_rate_pct']:.2f}%"
                )

            if b_score > a_score:
                winner = "B_challenger"
            elif a_score > b_score:
                winner = "A_control"
            else:
                winner = "tie — no significant difference"

        elapsed = int(time.time() - self.start_time)

        return {
            "experiment_id":   self.experiment_id,
            "active":          self.active,
            "split_pct":       f"{100-self.split_pct:.0f}% A / {self.split_pct:.0f}% B",
            "elapsed_seconds": elapsed,
            "sufficient_data": sufficient,
            "min_requests":    self.min_requests,
            "group_a":         a,
            "group_b":         b,
            "winner":          winner,
            "winner_reasons":  winner_reason,
            "recommendation":  (
                f"Deploy {winner} to 100% traffic"
                if sufficient and winner and "tie" not in winner
                else "Continue collecting data"
                if not sufficient
                else "No clear winner — keep current model"
            )
        }

    def stop(self) -> dict:
        """Stop the experiment and return final results."""
        self.active = False
        return self.get_results()


# ── Global experiment instance ────────────────────────────────
_current_experiment: Optional[ABTestingFramework] = None


def get_experiment() -> Optional[ABTestingFramework]:
    return _current_experiment


def start_experiment(
    split_pct: float = 50.0,
    min_requests: int = 100,
    experiment_id: str = None
) -> ABTestingFramework:
    global _current_experiment
    exp_id = experiment_id or f"exp_{int(time.time())}"
    _current_experiment = ABTestingFramework(
        split_pct=split_pct,
        min_requests=min_requests,
        experiment_id=exp_id
    )
    return _current_experiment


def stop_experiment() -> Optional[dict]:
    global _current_experiment
    if _current_experiment:
        results = _current_experiment.stop()
        _current_experiment = None
        return results
    return None


# ── Test ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  A/B TESTING FRAMEWORK TEST")
    print("=" * 60)

    import random
    random.seed(42)

    # Start experiment: 50/50 split, need 20 requests to decide
    exp = start_experiment(split_pct=50.0, min_requests=20,
                           experiment_id="test_001")

    print(f"\n  Experiment: {exp.experiment_id}")
    print(f"  Split: {exp.split_pct}% to Group B")

    print("\n[1] Simulating 50 transactions...")
    cards       = [f"card_{i:03d}" for i in range(1, 21)]
    decisions_a = ["APPROVE", "APPROVE", "APPROVE", "BLOCK", "STEP_UP_AUTH"]
    decisions_b = ["APPROVE", "APPROVE", "BLOCK", "BLOCK", "STEP_UP_AUTH"]

    group_counts = {"A": 0, "B": 0}
    for i in range(50):
        card  = random.choice(cards)
        group = exp.assign_group(card)
        group_counts[group] += 1

        # Simulate Group B detecting slightly more fraud
        if group == "A":
            decision = random.choice(decisions_a)
            latency  = random.uniform(8, 15)
            score    = random.uniform(0.1, 0.9)
        else:
            decision = random.choice(decisions_b)
            latency  = random.uniform(7, 13)  # slightly faster
            score    = random.uniform(0.1, 0.9)

        exp.record_result(card, group, decision, latency, score)

    print(f"    Group A received: {group_counts['A']} requests")
    print(f"    Group B received: {group_counts['B']} requests")

    print("\n[2] Check assignment is deterministic:")
    test_card = "card_001"
    groups = [exp.assign_group(test_card) for _ in range(5)]
    print(f"    card_001 always assigned to: {set(groups)} ✅")

    print("\n[3] Results:")
    results = exp.get_results()
    print(f"    Sufficient data: {results['sufficient_data']}")
    print(f"    Group A — Fraud: {results['group_a']['fraud_rate_pct']:.1f}%  "
          f"Latency: {results['group_a']['avg_latency_ms']:.1f}ms  "
          f"Requests: {results['group_a']['requests']}")
    print(f"    Group B — Fraud: {results['group_b']['fraud_rate_pct']:.1f}%  "
          f"Latency: {results['group_b']['avg_latency_ms']:.1f}ms  "
          f"Requests: {results['group_b']['requests']}")
    print(f"    Winner: {results['winner']}")
    print(f"    Recommendation: {results['recommendation']}")

    if results['winner_reasons']:
        print("    Reasons:")
        for r in results['winner_reasons']:
            print(f"      • {r}")

    print("\n" + "=" * 60)
    print("  A/B FRAMEWORK COMPLETE ✅")
    print("  Same card always goes to same group (deterministic)")
    print("  Tracks fraud rate, latency, false alarm rate per group")
    print("  Auto-determines winner after sufficient data")
    print("=" * 60)
