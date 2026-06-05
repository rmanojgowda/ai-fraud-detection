"""
Geographic Risk Scoring
========================
Flags transactions from unusual geographic locations.

Real-world signals:
  1. Country mismatch: Card issued in India, used in Romania
  2. High-risk countries: Known fraud hotspots
  3. Impossible travel: Transaction in Mumbai at 10am,
     then London at 11am (physically impossible)
  4. VPN/Proxy detection: IP matches known VPN ranges

Note: We don't have real IP geolocation in demo,
so we use IP patterns and card country metadata.

Interview value:
  "I added geographic risk scoring. A card issued in India
   transacting from a high-risk country gets a risk bump.
   We also detect impossible travel — two transactions from
   different continents within 1 hour is physically impossible,
   which is a strong fraud signal."
"""

import time
import math
from collections import defaultdict
from typing import Optional, Tuple
import threading


# ── High-Risk Country List ────────────────────────────────────
# Based on card fraud statistics (illustrative)
HIGH_RISK_COUNTRIES = {
    "RO", "RU", "UA", "NG", "GH", "KE",  # Eastern Europe + West Africa
    "PK", "BD", "VN", "ID",               # South/SE Asia fraud hotspots
    "BR", "MX", "CO",                     # Latin America
}

# Medium risk (elevated but not high)
MEDIUM_RISK_COUNTRIES = {
    "CN", "TR", "EG", "MA", "TN",
    "PH", "TH", "MY",
}

# Low risk (trusted)
LOW_RISK_COUNTRIES = {
    "US", "GB", "DE", "FR", "AU", "CA",
    "JP", "SG", "NL", "SE", "NO", "CH",
    "IN",  # India — home country in our case
}

# Approximate city coordinates for impossible travel detection
CITY_COORDS = {
    "Mumbai":    (19.08, 72.88),
    "Delhi":     (28.61, 77.21),
    "Bangalore": (12.97, 77.59),
    "London":    (51.51, -0.13),
    "New York":  (40.71, -74.01),
    "Paris":     (48.85, 2.35),
    "Singapore": (1.35, 103.82),
    "Dubai":     (25.20, 55.27),
    "Tokyo":     (35.68, 139.69),
    "Sydney":    (-33.87, 151.21),
}

# Max speed for "possible" travel (km/h)
# Commercial flight ~900 km/h, with airport time ~600 km/h effective
MAX_TRAVEL_SPEED_KMH = 600


def haversine_km(lat1: float, lon1: float,
                 lat2: float, lon2: float) -> float:
    """Distance between two coordinates in km."""
    R   = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (math.sin(dphi/2)**2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(dlam/2)**2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


class GeoRiskScorer:
    """
    Scores geographic risk for transactions.

    Signals:
      1. Country risk level (high/medium/low)
      2. Home country mismatch
      3. Impossible travel detection
      4. IP pattern analysis (VPN-like IPs)
    """

    def __init__(self, home_country: str = "IN"):
        self.home_country = home_country
        self._lock        = threading.Lock()

        # Track last known location per card
        # card_id → {"country": "IN", "city": "Mumbai",
        #             "lat": 19.08, "lon": 72.88, "timestamp": 123}
        self._card_locations: dict = defaultdict(dict)

    def score_transaction(
        self,
        card_id:         str,
        country_code:    str,          # ISO 2-letter: "IN", "US", "RO"
        city:            Optional[str] = None,
        ip_address:      str           = "0.0.0.0",
    ) -> Tuple[float, list]:
        """
        Returns (geo_risk_score, signals).
        Score 0.0 to 1.0.
        """
        risk    = 0.0
        signals = []

        country = country_code.upper()

        # ── Signal 1: High-risk country ───────────────────────
        if country in HIGH_RISK_COUNTRIES:
            risk += 0.35
            signals.append(
                f"High-risk country: {country} "
                f"(elevated fraud rate)"
            )
        elif country in MEDIUM_RISK_COUNTRIES:
            risk += 0.15
            signals.append(
                f"Medium-risk country: {country}"
            )

        # ── Signal 2: Home country mismatch ───────────────────
        if country != self.home_country and country not in LOW_RISK_COUNTRIES:
            risk += 0.20
            signals.append(
                f"Country mismatch: card home={self.home_country}, "
                f"transaction={country}"
            )

        # ── Signal 3: Impossible travel detection ─────────────
        with self._lock:
            last = self._card_locations.get(card_id, {})

        if last and city and city in CITY_COORDS:
            prev_city = last.get("city")
            prev_time = last.get("timestamp", 0)

            if prev_city and prev_city in CITY_COORDS:
                lat1, lon1 = CITY_COORDS[prev_city]
                lat2, lon2 = CITY_COORDS[city]
                distance_km = haversine_km(lat1, lon1, lat2, lon2)
                time_hours  = (time.time() - prev_time) / 3600

                if time_hours > 0 and distance_km > 100:
                    required_speed = distance_km / time_hours
                    if required_speed > MAX_TRAVEL_SPEED_KMH:
                        risk += 0.50
                        signals.append(
                            f"IMPOSSIBLE TRAVEL: {prev_city} → {city} "
                            f"({distance_km:.0f}km in {time_hours*60:.0f}min, "
                            f"requires {required_speed:.0f}km/h)"
                        )

        # ── Signal 4: VPN/Proxy IP pattern ────────────────────
        if self._is_vpn_like(ip_address):
            risk += 0.20
            signals.append(
                f"VPN/Proxy detected: IP {ip_address} "
                f"matches known VPN ranges"
            )

        # ── Update card location ───────────────────────────────
        if city and city in CITY_COORDS:
            lat, lon = CITY_COORDS[city]
            with self._lock:
                self._card_locations[card_id] = {
                    "country":   country,
                    "city":      city,
                    "lat":       lat,
                    "lon":       lon,
                    "timestamp": time.time(),
                }

        if not signals:
            signals.append(
                f"No geographic risk signals (country: {country})"
            )

        return min(round(risk, 4), 1.0), signals

    def _is_vpn_like(self, ip: str) -> bool:
        """
        Simple VPN detection based on IP patterns.
        Real systems use MaxMind or IP2Location databases.
        """
        # Known VPN/datacenter IP ranges (simplified)
        vpn_prefixes = [
            "10.8.",    # common OpenVPN range
            "10.9.",
            "172.16.",  # private ranges used by VPNs
            "192.168.99.",
        ]
        return any(ip.startswith(p) for p in vpn_prefixes)

    def get_country_risk_level(self, country: str) -> str:
        c = country.upper()
        if c in HIGH_RISK_COUNTRIES:
            return "HIGH"
        elif c in MEDIUM_RISK_COUNTRIES:
            return "MEDIUM"
        elif c in LOW_RISK_COUNTRIES:
            return "LOW"
        return "UNKNOWN"

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "cards_tracked":  len(self._card_locations),
                "home_country":   self.home_country,
                "high_risk_countries": len(HIGH_RISK_COUNTRIES),
                "medium_risk_countries": len(MEDIUM_RISK_COUNTRIES),
            }


# ── Global instance ───────────────────────────────────────────
_geo_scorer = GeoRiskScorer(home_country="IN")


def get_geo_scorer() -> GeoRiskScorer:
    return _geo_scorer


def score_geo_risk(
    card_id:      str,
    country_code: str,
    city:         Optional[str] = None,
    ip_address:   str = "0.0.0.0"
) -> Tuple[float, list]:
    return _geo_scorer.score_transaction(
        card_id, country_code, city, ip_address
    )


# ── Test ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  GEOGRAPHIC RISK SCORING TEST")
    print("=" * 60)

    scorer = GeoRiskScorer(home_country="IN")

    tests = [
        ("card_001", "IN", "Mumbai",    "192.168.1.1",  "Normal — India"),
        ("card_002", "US", "New York",  "192.168.1.2",  "Low risk country"),
        ("card_003", "RO", None,        "192.168.1.3",  "High-risk country"),
        ("card_004", "RU", None,        "10.8.0.1",     "High-risk + VPN"),
        ("card_005", "NG", None,        "192.168.1.5",  "High-risk Nigeria"),
    ]

    print("\n[1] Country risk tests:")
    for card, country, city, ip, desc in tests:
        score, sigs = scorer.score_transaction(card, country, city, ip)
        print(f"\n  {desc}:")
        print(f"    Card: {card} | Country: {country} | Score: {score:.2f}")
        for s in sigs:
            print(f"    ⚠️  {s}")

    print("\n[2] Impossible travel test:")
    scorer2 = GeoRiskScorer(home_country="IN")

    # First transaction: Mumbai
    s1, sig1 = scorer2.score_transaction(
        "card_travel", "IN", "Mumbai", "192.168.1.1"
    )
    print(f"  Transaction 1 (Mumbai): score={s1:.2f}")

    # Simulate 30 minutes passing
    import time
    scorer2._card_locations["card_travel"]["timestamp"] = time.time() - 1800

    # Second transaction: London (impossible in 30 min)
    s2, sig2 = scorer2.score_transaction(
        "card_travel", "GB", "London", "192.168.1.1"
    )
    print(f"  Transaction 2 (London, 30min later): score={s2:.2f}")
    for s in sig2:
        if "IMPOSSIBLE" in s or "mismatch" in s.lower():
            print(f"    🚨 {s}")

    print("\n[3] Risk levels:")
    for c in ["IN", "US", "RO", "RU", "CN", "XX"]:
        level = scorer.get_country_risk_level(c)
        print(f"    {c}: {level}")

    print("\n" + "=" * 60)
    print("  GEO RISK SCORING COMPLETE ✅")
    print("  Country risk + impossible travel + VPN detection")
    print("=" * 60)
