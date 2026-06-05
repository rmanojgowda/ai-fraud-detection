"""
Transaction Replay System
==========================
Saves blocked/flagged transactions and replays them
against new model versions to measure improvement.

Real-world usage at banks:
  1. Deploy new model version
  2. Replay last 30 days of blocked transactions
  3. Compare: did new model catch more fraud?
  4. If yes → full rollout. If no → rollback.

This is called "shadow mode" or "replay testing" in MLOps.

Interview value:
  "I built a replay system that saves every blocked transaction.
   When we retrain the model, we replay these transactions
   and measure whether the new model improves detection.
   This gives us confidence before full deployment — we're
   not flying blind."
"""

import json
import os
import time
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict
from typing import Optional


class TransactionReplaySystem:
    """
    Saves transactions and replays them against model versions.

    Storage format: JSONL (one JSON per line)
    File: logs/replay_transactions.jsonl

    Each saved transaction:
    {
      "timestamp":    1234567890.0,
      "request_id":   "abc123",
      "features":     {...39 features...},
      "card_id":      "card_001",
      "original_decision": "BLOCK",
      "original_score":    0.95,
      "amount":       149.62,
      "hour":         2,
      "save_reason":  "blocked" | "fraud_ring" | "all"
    }
    """

    def __init__(
        self,
        save_path:    str = "logs/replay_transactions.jsonl",
        max_records:  int = 10000,
        save_mode:    str = "blocked"  # "blocked", "all", "fraud_ring"
    ):
        self.save_path   = save_path
        self.max_records = max_records
        self.save_mode   = save_mode
        self._record_count = 0

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Count existing records
        if os.path.exists(save_path):
            with open(save_path) as f:
                self._record_count = sum(1 for _ in f)

    def should_save(self, decision: str, graph_score: float) -> bool:
        """Decides whether to save this transaction."""
        if self._record_count >= self.max_records:
            return False
        if self.save_mode == "all":
            return True
        if self.save_mode == "blocked":
            return decision == "BLOCK"
        if self.save_mode == "fraud_ring":
            return graph_score > 0.3
        return decision in ["BLOCK", "STEP_UP_AUTH"]

    def save_transaction(
        self,
        request_id:  str,
        features:    dict,
        card_id:     str,
        decision:    str,
        score:       float,
        graph_score: float,
        amount:      float,
        hour:        int,
    ) -> bool:
        """Save a transaction for later replay."""
        if not self.should_save(decision, graph_score):
            return False

        record = {
            "timestamp":          time.time(),
            "saved_at":           datetime.utcnow().isoformat(),
            "request_id":         request_id,
            "card_id":            card_id[:8] + "...",  # anonymized
            "features":           {k: round(float(v), 6)
                                   for k, v in features.items()},
            "original_decision":  decision,
            "original_score":     round(score, 6),
            "original_graph":     round(graph_score, 6),
            "amount":             round(amount, 2),
            "hour":               hour,
            "save_reason":        self.save_mode,
        }

        try:
            with open(self.save_path, "a") as f:
                f.write(json.dumps(record) + "\n")
            self._record_count += 1
            return True
        except Exception:
            return False

    def load_transactions(
        self,
        limit:       Optional[int] = None,
        min_score:   float = 0.0,
        decision_filter: Optional[str] = None
    ) -> list:
        """Load saved transactions with optional filters."""
        records = []
        if not os.path.exists(self.save_path):
            return records

        with open(self.save_path) as f:
            for line in f:
                try:
                    rec = json.loads(line.strip())
                    if rec["original_score"] < min_score:
                        continue
                    if decision_filter and rec["original_decision"] != decision_filter:
                        continue
                    records.append(rec)
                    if limit and len(records) >= limit:
                        break
                except Exception:
                    continue

        return records

    def replay_against_model(
        self,
        model_path:   str,
        feature_cols: list,
        threshold:    float = 0.7722,
        limit:        Optional[int] = None
    ) -> dict:
        """
        Replay saved transactions against a new model version.

        Returns comparison metrics:
          - How many would be caught by new model
          - How many would be missed by new model
          - Score distribution changes
        """
        transactions = self.load_transactions(limit=limit)
        if not transactions:
            return {"error": "No transactions to replay"}

        # Load new model
        try:
            new_model = joblib.load(model_path)
        except Exception as e:
            return {"error": f"Cannot load model: {e}"}

        results = {
            "total_replayed":        0,
            "original_blocks":       0,
            "new_model_blocks":      0,
            "newly_caught":          0,  # new catches that old missed
            "newly_missed":          0,  # old catches that new misses
            "score_improved":        0,  # new score > original
            "score_degraded":        0,  # new score < original
            "avg_score_original":    0.0,
            "avg_score_new":         0.0,
            "decision_changes":      [],
        }

        original_scores = []
        new_scores      = []

        for rec in transactions:
            try:
                features   = rec["features"]
                df         = pd.DataFrame([features], columns=feature_cols)
                new_score  = float(new_model.predict_proba(df)[0][1])
                orig_score = rec["original_score"]
                orig_dec   = rec["original_decision"]
                new_dec    = "BLOCK" if new_score >= threshold else \
                             "STEP_UP_AUTH" if new_score >= 0.25 else "APPROVE"

                results["total_replayed"] += 1
                original_scores.append(orig_score)
                new_scores.append(new_score)

                if orig_dec == "BLOCK":
                    results["original_blocks"] += 1
                if new_dec == "BLOCK":
                    results["new_model_blocks"] += 1
                if orig_dec != "BLOCK" and new_dec == "BLOCK":
                    results["newly_caught"] += 1
                if orig_dec == "BLOCK" and new_dec != "BLOCK":
                    results["newly_missed"] += 1
                if new_score > orig_score + 0.05:
                    results["score_improved"] += 1
                elif new_score < orig_score - 0.05:
                    results["score_degraded"] += 1

                # Track significant decision changes
                if orig_dec != new_dec:
                    results["decision_changes"].append({
                        "request_id":  rec["request_id"],
                        "amount":      rec["amount"],
                        "hour":        rec["hour"],
                        "original":    orig_dec,
                        "new":         new_dec,
                        "orig_score":  round(orig_score, 4),
                        "new_score":   round(new_score, 4),
                    })

            except Exception:
                continue

        if original_scores:
            results["avg_score_original"] = round(
                sum(original_scores) / len(original_scores), 4)
            results["avg_score_new"] = round(
                sum(new_scores) / len(new_scores), 4)

        # Verdict
        if results["newly_caught"] > results["newly_missed"]:
            results["verdict"] = "NEW_MODEL_BETTER"
            results["recommendation"] = (
                f"New model catches {results['newly_caught']} more transactions. "
                f"Recommend deploying new model."
            )
        elif results["newly_missed"] > results["newly_caught"]:
            results["verdict"] = "OLD_MODEL_BETTER"
            results["recommendation"] = (
                f"New model misses {results['newly_missed']} transactions. "
                f"Keep current model."
            )
        else:
            results["verdict"] = "NO_CHANGE"
            results["recommendation"] = "Models perform equally. No change needed."

        # Limit decision changes shown
        results["decision_changes"] = results["decision_changes"][:10]
        results["total_decision_changes"] = len(results["decision_changes"])

        return results

    def get_stats(self) -> dict:
        """Returns stats about saved transactions."""
        if not os.path.exists(self.save_path):
            return {"total_saved": 0}

        decisions = defaultdict(int)
        hours     = defaultdict(int)
        amounts   = []

        with open(self.save_path) as f:
            for line in f:
                try:
                    rec = json.loads(line.strip())
                    decisions[rec["original_decision"]] += 1
                    hours[rec["hour"]] += 1
                    amounts.append(rec["amount"])
                except Exception:
                    continue

        return {
            "total_saved":         self._record_count,
            "save_mode":           self.save_mode,
            "max_records":         self.max_records,
            "by_decision":         dict(decisions),
            "avg_amount":          round(sum(amounts)/max(len(amounts),1), 2),
            "peak_hour":           max(hours, key=hours.get) if hours else None,
            "save_path":           self.save_path,
        }

    def clear(self) -> None:
        """Clear all saved transactions."""
        if os.path.exists(self.save_path):
            os.remove(self.save_path)
        self._record_count = 0


# ── Global instance ───────────────────────────────────────────
_replay_system = TransactionReplaySystem(
    save_path  = "logs/replay_transactions.jsonl",
    max_records= 10000,
    save_mode  = "blocked"
)


def get_replay_system() -> TransactionReplaySystem:
    return _replay_system


def save_for_replay(
    request_id:  str,
    features:    dict,
    card_id:     str,
    decision:    str,
    score:       float,
    graph_score: float,
    amount:      float,
    hour:        int,
) -> None:
    _replay_system.save_transaction(
        request_id, features, card_id,
        decision, score, graph_score, amount, hour
    )


# ── Test ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    print("=" * 60)
    print("  TRANSACTION REPLAY SYSTEM TEST")
    print("=" * 60)

    # Use temp file for test
    replay = TransactionReplaySystem(
        save_path  = "logs/test_replay.jsonl",
        max_records= 1000,
        save_mode  = "blocked"
    )
    replay.clear()

    print("\n[1] Saving 10 blocked transactions...")
    for i in range(10):
        features = {f"V{j}": float(i * 0.1) for j in range(1, 29)}
        features.update({
            "Amount": float(100 + i * 50),
            "amount_log": np.log1p(100 + i * 50),
            "amount_sqrt": np.sqrt(100 + i * 50),
            "tx_count_1min": i % 5 + 1,
            "tx_count_10min": i % 10 + 1,
            "tx_count_60min": i % 20 + 1,
            "amount_rolling_mean_1h": 200.0,
            "amount_rolling_std_1h": 50.0,
            "amount_deviation": float(i * 0.5),
            "hour": (2 + i) % 24,
            "is_night": 1 if (2 + i) % 24 < 5 else 0,
        })
        saved = replay.save_transaction(
            request_id  = f"req_{i:03d}",
            features    = features,
            card_id     = f"card_{i:03d}",
            decision    = "BLOCK",
            score       = 0.85 + i * 0.01,
            graph_score = 0.3,
            amount      = 100 + i * 50,
            hour        = (2 + i) % 24,
        )
        print(f"    Transaction {i+1}: {'✅ saved' if saved else '❌ skipped'}")

    print("\n[2] Stats:")
    stats = replay.get_stats()
    for k, v in stats.items():
        print(f"    {k}: {v}")

    print("\n[3] Replay against current model:")
    with open("models/feature_cols.json") as f:
        cols = json.load(f)

    results = replay.replay_against_model(
        model_path   = "models/fraud_model.pkl",
        feature_cols = cols,
        threshold    = 0.7722,
    )
    print(f"    Total replayed    : {results['total_replayed']}")
    print(f"    Original blocks   : {results['original_blocks']}")
    print(f"    New model blocks  : {results.get('new_model_blocks', 0)}")
    print(f"    Newly caught      : {results.get('newly_caught', 0)}")
    print(f"    Newly missed      : {results.get('newly_missed', 0)}")
    print(f"    Verdict           : {results.get('verdict', 'N/A')}")
    print(f"    Recommendation    : {results.get('recommendation', 'N/A')}")

    print("\n" + "=" * 60)
    print("  REPLAY SYSTEM COMPLETE ✅")
    print("  Blocked transactions saved for model comparison")
    print("  Replay shows whether new model is better/worse")
    print("=" * 60)

    # Cleanup test file
    replay.clear()
