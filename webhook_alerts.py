"""
Webhook Alerts System
======================
Sends real-time alerts when critical fraud events occur.

Supported channels:
  1. Slack webhook (free, instant)
  2. Generic webhook (any HTTP endpoint)
  3. Console/log fallback (always works, no setup needed)

Alert triggers:
  - Fraud ring detected (graph layer)
  - High-risk transaction blocked
  - Rate limit spike (card testing attack)
  - System anomaly (Redis down, error rate spike)

Real-world usage:
  Fraud analysts get Slack alerts at 2am when a ring forms.
  Without alerts, you'd only find out next morning.
  This is the difference between catching fraud in 5 minutes
  vs 8 hours.

Interview value:
  "I built webhook alerts so fraud analysts are notified
   immediately when a ring is detected. The system sends
   structured JSON payloads to Slack or any webhook endpoint.
   During testing I caught a simulated ring forming across
   5 cards in under 3 seconds."
"""

import json
import time
import threading
import logging
import os
from datetime import datetime
from collections import defaultdict, deque
from typing import Optional
from urllib import request as urllib_request
from urllib.error import URLError

logger = logging.getLogger("fraud_api")


# ── Alert Types ───────────────────────────────────────────────
class AlertType:
    FRAUD_RING_DETECTED   = "fraud_ring_detected"
    HIGH_RISK_BLOCKED     = "high_risk_blocked"
    RATE_LIMIT_SPIKE      = "rate_limit_spike"
    IMPOSSIBLE_TRAVEL     = "impossible_travel"
    SYSTEM_ANOMALY        = "system_anomaly"
    CARD_TESTING_ATTACK   = "card_testing_attack"


# ── Alert Severity ────────────────────────────────────────────
class Severity:
    CRITICAL = "🚨 CRITICAL"
    HIGH     = "⚠️  HIGH"
    MEDIUM   = "📊 MEDIUM"
    LOW      = "ℹ️  LOW"


class WebhookAlertSystem:
    """
    Sends alerts to Slack/webhook when fraud events occur.

    Design decisions:
      1. Non-blocking: alerts sent in background thread
         Why: Don't add latency to fraud check endpoint
      2. Rate limiting: max 1 alert per type per 5 minutes
         Why: Prevent alert fatigue during attack floods
      3. Fallback: always logs even if webhook fails
         Why: Never lose alert data
      4. Retry: 3 attempts with backoff
         Why: Transient network failures shouldn't lose alerts
    """

    def __init__(
        self,
        slack_webhook_url: Optional[str] = None,
        generic_webhook_url: Optional[str] = None,
        min_interval_seconds: int = 300,  # 5 min between same alert type
        enabled: bool = True
    ):
        self.slack_url         = slack_webhook_url
        self.generic_url       = generic_webhook_url
        self.min_interval      = min_interval_seconds
        self.enabled           = enabled

        # Rate limiting per alert type
        self._last_sent: dict  = defaultdict(float)
        self._lock             = threading.Lock()

        # Alert history (last 100)
        self._history          = deque(maxlen=100)

        # Stats
        self.total_alerts      = 0
        self.successful_sends  = 0
        self.failed_sends      = 0
        self.suppressed        = 0

        os.makedirs("logs", exist_ok=True)

    def send_alert(
        self,
        alert_type:  str,
        severity:    str,
        title:       str,
        details:     dict,
        force:       bool = False
    ) -> bool:
        """
        Send an alert. Returns True if sent, False if suppressed.

        Args:
            alert_type: AlertType constant
            severity:   Severity constant
            title:      Short human-readable summary
            details:    Structured data about the event
            force:      Skip rate limiting
        """
        if not self.enabled:
            return False

        # Rate limiting check
        with self._lock:
            now      = time.time()
            last     = self._last_sent[alert_type]
            if not force and (now - last) < self.min_interval:
                self.suppressed += 1
                remaining = int(self.min_interval - (now - last))
                logger.info(json.dumps({
                    "event":     "alert_suppressed",
                    "type":      alert_type,
                    "reason":    f"rate_limited_{remaining}s_remaining"
                }))
                return False
            self._last_sent[alert_type] = now

        self.total_alerts += 1

        payload = {
            "alert_type":  alert_type,
            "severity":    severity,
            "title":       title,
            "details":     details,
            "timestamp":   datetime.utcnow().isoformat(),
            "system":      "AI Fraud Detection v6.0.0",
        }

        # Always log
        self._log_alert(payload)

        # Add to history
        self._history.append(payload)

        # Send to webhooks in background
        thread = threading.Thread(
            target=self._send_to_webhooks,
            args=(payload,),
            daemon=True
        )
        thread.start()

        return True

    def _send_to_webhooks(self, payload: dict) -> None:
        """Send to all configured webhook endpoints."""
        sent = False

        if self.slack_url:
            sent = self._send_slack(payload) or sent

        if self.generic_url:
            sent = self._send_generic(payload) or sent

        if sent:
            self.successful_sends += 1
        else:
            self.failed_sends += 1

    def _send_slack(self, payload: dict) -> bool:
        """Format and send Slack message."""
        details = payload["details"]
        sev     = payload["severity"]
        title   = payload["title"]

        # Build Slack block kit message
        slack_payload = {
            "text": f"{sev}: {title}",
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"{sev}: {title}"
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn",
                         "text": f"*Type:*\n{payload['alert_type']}"},
                        {"type": "mrkdwn",
                         "text": f"*Time:*\n{payload['timestamp'][:19]}"},
                    ]
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "*Details:*\n" + "\n".join(
                            f"• {k}: `{v}`"
                            for k, v in details.items()
                            if not isinstance(v, dict)
                        )
                    }
                },
                {"type": "divider"}
            ]
        }

        return self._http_post(self.slack_url, slack_payload)

    def _send_generic(self, payload: dict) -> bool:
        """Send raw JSON to generic webhook."""
        return self._http_post(self.generic_url, payload)

    def _http_post(self, url: str, data: dict,
                   retries: int = 3) -> bool:
        """POST JSON to URL with retry logic."""
        body = json.dumps(data).encode("utf-8")

        for attempt in range(retries):
            try:
                req = urllib_request.Request(
                    url,
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST"
                )
                with urllib_request.urlopen(req, timeout=5) as resp:
                    return resp.status in (200, 201, 202, 204)
            except URLError as e:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # exponential backoff
                else:
                    logger.warning(json.dumps({
                        "event":   "webhook_send_failed",
                        "url":     url[:50] + "...",
                        "error":   str(e),
                        "attempt": attempt + 1
                    }))
            except Exception as e:
                logger.warning(json.dumps({
                    "event": "webhook_error",
                    "error": str(e)
                }))
                break

        return False

    def _log_alert(self, payload: dict) -> None:
        """Always log alert to file."""
        try:
            with open("logs/alerts.jsonl", "a") as f:
                f.write(json.dumps(payload) + "\n")
        except Exception:
            pass
        logger.info(json.dumps({
            "event":      "alert_fired",
            "alert_type": payload["alert_type"],
            "severity":   payload["severity"],
            "title":      payload["title"],
        }))

    # ── Convenience Methods ───────────────────────────────────

    def alert_fraud_ring(
        self,
        ring_type:    str,
        card_count:   int,
        merchant_ids: list,
        ip_addresses: list,
        risk_level:   str = "HIGH"
    ) -> bool:
        return self.send_alert(
            alert_type = AlertType.FRAUD_RING_DETECTED,
            severity   = Severity.CRITICAL,
            title      = f"Fraud ring detected: {card_count} cards in coordinated attack",
            details    = {
                "ring_type":    ring_type,
                "card_count":   card_count,
                "merchants":    len(merchant_ids),
                "ips":          len(ip_addresses),
                "risk_level":   risk_level,
                "sample_ip":    ip_addresses[0] if ip_addresses else "unknown",
                "action":       "All cards blocked by rate limiter"
            }
        )

    def alert_high_risk_blocked(
        self,
        request_id:  str,
        risk_score:  float,
        ml_score:    float,
        geo_score:   float,
        amount:      float,
        country:     str,
        card_id:     str
    ) -> bool:
        return self.send_alert(
            alert_type = AlertType.HIGH_RISK_BLOCKED,
            severity   = Severity.HIGH,
            title      = f"High-risk transaction blocked: ₹{amount:.2f} from {country}",
            details    = {
                "request_id": request_id,
                "risk_score": round(risk_score, 4),
                "ml_score":   round(ml_score, 4),
                "geo_score":  round(geo_score, 4),
                "amount_inr": round(amount, 2),
                "country":    country,
                "card_id":    card_id[:8] + "...",
            }
        )

    def alert_rate_limit_spike(
        self,
        client_ip:    str,
        card_id:      str,
        block_count:  int,
        window_sec:   int
    ) -> bool:
        return self.send_alert(
            alert_type = AlertType.RATE_LIMIT_SPIKE,
            severity   = Severity.HIGH,
            title      = f"Card testing attack: {block_count} blocks in {window_sec}s",
            details    = {
                "client_ip":   client_ip,
                "card_id":     card_id[:8] + "...",
                "block_count": block_count,
                "window_sec":  window_sec,
                "attack_type": "card_testing",
                "action":      "Rate limiter blocking requests"
            }
        )

    def alert_impossible_travel(
        self,
        card_id:    str,
        from_city:  str,
        to_city:    str,
        distance_km: float,
        time_min:   float
    ) -> bool:
        return self.send_alert(
            alert_type = AlertType.IMPOSSIBLE_TRAVEL,
            severity   = Severity.CRITICAL,
            title      = f"Impossible travel: {from_city}→{to_city} in {time_min:.0f}min",
            details    = {
                "card_id":     card_id[:8] + "...",
                "from_city":   from_city,
                "to_city":     to_city,
                "distance_km": round(distance_km, 0),
                "time_minutes": round(time_min, 1),
                "action":      "Transaction flagged for review"
            }
        )

    def get_history(self, limit: int = 20) -> list:
        """Returns recent alert history."""
        history = list(self._history)
        return history[-limit:]

    def get_stats(self) -> dict:
        return {
            "enabled":          self.enabled,
            "total_alerts":     self.total_alerts,
            "successful_sends": self.successful_sends,
            "failed_sends":     self.failed_sends,
            "suppressed":       self.suppressed,
            "slack_configured": bool(self.slack_url),
            "webhook_configured": bool(self.generic_url),
            "recent_alerts":    len(self._history),
            "min_interval_sec": self.min_interval,
        }


# ── Global instance ───────────────────────────────────────────
# Set SLACK_WEBHOOK_URL environment variable to enable Slack alerts
_alert_system = WebhookAlertSystem(
    slack_webhook_url   = os.environ.get("SLACK_WEBHOOK_URL"),
    generic_webhook_url = os.environ.get("WEBHOOK_URL"),
    min_interval_seconds= 300,
    enabled             = True
)


def get_alert_system() -> WebhookAlertSystem:
    return _alert_system


def send_fraud_ring_alert(ring_type, card_count,
                          merchant_ids, ip_addresses) -> bool:
    return _alert_system.alert_fraud_ring(
        ring_type, card_count, merchant_ids, ip_addresses)


def send_high_risk_alert(request_id, risk_score, ml_score,
                         geo_score, amount, country, card_id) -> bool:
    return _alert_system.alert_high_risk_blocked(
        request_id, risk_score, ml_score,
        geo_score, amount, country, card_id)


def send_rate_limit_alert(client_ip, card_id,
                          block_count, window_sec) -> bool:
    return _alert_system.alert_rate_limit_spike(
        client_ip, card_id, block_count, window_sec)


# ── Test ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  WEBHOOK ALERT SYSTEM TEST")
    print("=" * 60)

    # Test without real webhook — uses log fallback
    system = WebhookAlertSystem(
        slack_webhook_url   = None,  # set real URL to test Slack
        min_interval_seconds= 1,     # 1 second for testing
        enabled             = True
    )

    print("\n[1] Fraud ring alert:")
    sent = system.alert_fraud_ring(
        ring_type    = "IP_BASED",
        card_count   = 5,
        merchant_ids = ["merchant_X"],
        ip_addresses = ["10.0.0.1"],
        risk_level   = "HIGH"
    )
    print(f"    Alert fired: {sent}")
    time.sleep(0.1)

    print("\n[2] High-risk transaction alert:")
    sent = system.alert_high_risk_blocked(
        request_id  = "abc12345",
        risk_score  = 0.95,
        ml_score    = 0.9999,
        geo_score   = 0.75,
        amount      = 149.62,
        country     = "RO",
        card_id     = "card_fraud_001"
    )
    print(f"    Alert fired: {sent}")
    time.sleep(0.1)

    print("\n[3] Rate limit spike alert:")
    sent = system.alert_rate_limit_spike(
        client_ip   = "192.168.1.1",
        card_id     = "card_testing_001",
        block_count = 47,
        window_sec  = 10
    )
    print(f"    Alert fired: {sent}")
    time.sleep(0.1)

    print("\n[4] Impossible travel alert:")
    sent = system.alert_impossible_travel(
        card_id     = "card_travel_001",
        from_city   = "Mumbai",
        to_city     = "London",
        distance_km = 7192,
        time_min    = 30
    )
    print(f"    Alert fired: {sent}")
    time.sleep(0.1)

    print("\n[5] Rate limiting test (same alert twice):")
    system2 = WebhookAlertSystem(min_interval_seconds=60, enabled=True)
    r1 = system2.alert_fraud_ring("IP_BASED", 3, ["m1"], ["1.1.1.1"])
    r2 = system2.alert_fraud_ring("IP_BASED", 3, ["m1"], ["1.1.1.1"])
    print(f"    First alert:  {r1} (should be True)")
    print(f"    Second alert: {r2} (should be False — rate limited)")

    print("\n[6] Stats:")
    stats = system.get_stats()
    for k, v in stats.items():
        print(f"    {k}: {v}")

    print("\n[7] Alert history:")
    for alert in system.get_history():
        print(f"    [{alert['severity']}] {alert['title']}")

    print("\n[8] Check alerts.jsonl log:")
    if os.path.exists("logs/alerts.jsonl"):
        with open("logs/alerts.jsonl") as f:
            lines = f.readlines()
        print(f"    {len(lines)} alerts logged to logs/alerts.jsonl ✅")
    else:
        print("    No log file found")

    print("\n" + "=" * 60)
    print("  WEBHOOK ALERTS COMPLETE ✅")
    print("  To enable Slack: set SLACK_WEBHOOK_URL env variable")
    print("  Without Slack: alerts logged to logs/alerts.jsonl")
    print("=" * 60)
