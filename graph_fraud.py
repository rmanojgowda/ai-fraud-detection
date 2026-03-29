import networkx as nx
import pandas as pd
import numpy as np
import json
from collections import defaultdict

class FraudGraphDetector:
    def __init__(self):
        self.graph        = nx.Graph()
        self.transactions = []

    def add_transaction(self, tx: dict):
        card_id     = f"card_{tx.get('card_id', 'unknown')}"
        merchant_id = f"merchant_{tx.get('merchant_id', 'unknown')}"
        ip_addr     = f"ip_{tx.get('ip', 'unknown')}"
        amount      = tx.get('amount', 0)
        is_fraud    = tx.get('is_fraud', 0)

        for node in [card_id, merchant_id, ip_addr]:
            if node not in self.graph:
                self.graph.add_node(node,
                    node_type=node.split("_")[0],
                    fraud_count=0,
                    tx_count=0
                )

        self.graph.add_edge(card_id,     merchant_id, amount=amount, is_fraud=is_fraud)
        self.graph.add_edge(merchant_id, ip_addr,     amount=amount, is_fraud=is_fraud)
        self.graph.add_edge(card_id,     ip_addr,     amount=amount, is_fraud=is_fraud)

        for node in [card_id, merchant_id, ip_addr]:
            self.graph.nodes[node]['tx_count']    += 1
            self.graph.nodes[node]['fraud_count'] += is_fraud

        self.transactions.append(tx)

    def get_graph_risk_score(self, card_id: str, merchant_id: str, ip: str) -> dict:
        card_node     = f"card_{card_id}"
        merchant_node = f"merchant_{merchant_id}"
        ip_node       = f"ip_{ip}"

        risk_signals = []
        risk_score   = 0.0

        # Signal 1: Known fraud card
        if self.graph.has_node(card_node):
            data        = self.graph.nodes[card_node]
            tx_count    = data.get('tx_count', 0)
            fraud_count = data.get('fraud_count', 0)
            if tx_count > 0 and fraud_count / tx_count > 0.3:
                risk_score += 0.4
                risk_signals.append(
                    f"Card has {fraud_count/tx_count*100:.0f}% historical fraud rate"
                )

        # Signal 2: Suspicious merchant
        if self.graph.has_node(merchant_node):
            data        = self.graph.nodes[merchant_node]
            tx_count    = data.get('tx_count', 0)
            fraud_count = data.get('fraud_count', 0)
            if tx_count > 2 and fraud_count / max(tx_count, 1) > 0.2:
                risk_score += 0.3
                risk_signals.append(
                    f"Merchant linked to {fraud_count} fraud cases"
                )

        # Signal 3: Shared IP (lowered threshold to 2 cards)
        if self.graph.has_node(ip_node):
            ip_neighbors   = list(self.graph.neighbors(ip_node))
            card_neighbors = [n for n in ip_neighbors if n.startswith("card_")]
            if len(card_neighbors) >= 2:
                risk_score += 0.4
                risk_signals.append(
                    f"IP shared by {len(card_neighbors)} different cards — possible fraud ring"
                )

        # Signal 4: Connected to fraud entities
        if self.graph.has_node(card_node):
            component   = nx.node_connected_component(self.graph, card_node)
            fraud_nodes = [
                n for n in component
                if self.graph.nodes[n].get('fraud_count', 0) > 0
            ]
            if len(fraud_nodes) >= 1:
                risk_score += 0.2
                risk_signals.append(
                    f"Card connected to {len(fraud_nodes)} fraud-flagged entities in graph"
                )

        # Signal 5: IP shared by many cards (ring pattern)
        if self.graph.has_node(ip_node):
            ip_neighbors   = list(self.graph.neighbors(ip_node))
            card_neighbors = [n for n in ip_neighbors if n.startswith("card_")]
            if len(card_neighbors) >= 4:
                risk_score += 0.3
                risk_signals.append(
                    f"FRAUD RING: {len(card_neighbors)} cards sharing same IP and merchant"
                )

        if not risk_signals:
            risk_signals.append("No suspicious graph patterns detected")

        return {
            "graph_risk_score": min(round(risk_score, 4), 1.0),
            "graph_signals":    risk_signals,
            "nodes_in_graph":   self.graph.number_of_nodes(),
            "edges_in_graph":   self.graph.number_of_edges(),
        }

    def detect_fraud_rings(self) -> list:
        rings = []
        for component in nx.connected_components(self.graph):
            nodes     = list(component)
            cards     = [n for n in nodes if n.startswith("card_")]
            merchants = [n for n in nodes if n.startswith("merchant_")]
            ips       = [n for n in nodes if n.startswith("ip_")]

            # Lowered threshold: 2+ cards sharing infrastructure
            if len(cards) >= 2 and (len(merchants) >= 1 or len(ips) >= 1):
                total_fraud = sum(
                    self.graph.nodes[n].get('fraud_count', 0)
                    for n in nodes
                )
                # Check IP sharing (key fraud ring signal)
                shared_ip = len(ips) > 0 and len(cards) >= 2
                risk = "HIGH" if total_fraud > 0 else ("HIGH" if shared_ip and len(cards) >= 4 else "MEDIUM")

                rings.append({
                    "cards":        cards,
                    "merchants":    merchants,
                    "ips":          ips,
                    "total_fraud":  total_fraud,
                    "ring_size":    len(nodes),
                    "shared_ip":    shared_ip,
                    "risk":         risk,
                    "ring_pattern": f"{len(cards)} cards → {len(merchants)} merchants → {len(ips)} IPs"
                })

        return sorted(rings, key=lambda x: len(x['cards']), reverse=True)

    def get_stats(self) -> dict:
        rings = self.detect_fraud_rings()
        high_risk = [r for r in rings if r['risk'] == 'HIGH']
        return {
            "total_nodes":       self.graph.number_of_nodes(),
            "total_edges":       self.graph.number_of_edges(),
            "total_components":  nx.number_connected_components(self.graph),
            "fraud_rings":       len(rings),
            "high_risk_rings":   len(high_risk),
        }


# ── Demo ──────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("   GRAPH-BASED FRAUD DETECTION — DEMO")
    print("=" * 55)

    detector = FraudGraphDetector()

    print("\n[1/4] Adding normal transactions...")
    for tx in [
        {"card_id": "1001", "merchant_id": "M01", "ip": "10.0.0.1",  "amount": 50,  "is_fraud": 0},
        {"card_id": "1002", "merchant_id": "M02", "ip": "10.0.0.2",  "amount": 120, "is_fraud": 0},
    ]:
        detector.add_transaction(tx)

    print("[2/4] Adding fraud ring (5 cards, same merchant+IP)...")
    for i, card in enumerate(["9001","9002","9003","9004","9005"]):
        detector.add_transaction({
            "card_id": card, "merchant_id": "M99",
            "ip": "192.168.1.1", "amount": 1, "is_fraud": 1
        })

    print("\n[3/4] Graph Statistics:")
    stats = detector.get_stats()
    for k, v in stats.items():
        print(f"      {k}: {v}")

    print("\n[4/4] Scoring new transaction from ring IP...")
    result = detector.get_graph_risk_score("9001", "M99", "192.168.1.1")
    print(f"\n      Graph Risk Score : {result['graph_risk_score']}")
    print(f"      Signals:")
    for s in result['graph_signals']:
        print(f"        • {s}")

    print("\n      Fraud Rings:")
    for i, ring in enumerate(detector.detect_fraud_rings(), 1):
        print(f"\n      Ring {i}: {ring['ring_pattern']}")
        print(f"        Risk    : {ring['risk']}")
        print(f"        Cards   : {ring['cards']}")
        print(f"        IPs     : {ring['ips']}")

    print("\n" + "=" * 55)
    print("GRAPH DETECTION COMPLETE ✅")
    print("=" * 55)
