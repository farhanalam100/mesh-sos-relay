"""
A single mesh node. Nodes only know about neighbors within simulated radio range.
Routing is simple TTL-limited flooding with dedup (good enough for a LoRa-style mesh
demo: no routing tables needed, works even if the topology changes mid-simulation).
"""

import asyncio
import random
import math

from src.classifier import classify


class Node:
    def __init__(self, node_id: str, x: float, y: float, is_gateway: bool = False,
                 drop_prob: float = 0.03):
        self.id = node_id
        self.x = x
        self.y = y
        self.is_gateway = is_gateway
        self.drop_prob = drop_prob      # chance a hop is lost in transit (simulates interference/range)
        self.neighbors: list["Node"] = []
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.seen_ids: set[str] = set()
        self.delivered: list = []       # messages that reached this node as gateway

    def distance_to(self, other: "Node") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    async def receive(self, message):
        await self.inbox.put(message)

    async def run(self, log_fn=None):
        """Process inbox forever: dedup, decrement TTL, forward to neighbors (flood),
        or hand off to the gateway callback if this node is the gateway."""
        while True:
            message = await self.inbox.get()

            if message.id in self.seen_ids:
                continue  # already handled this message, drop the duplicate
            self.seen_ids.add(message.id)

            message.record_hop(self.id)
            if log_fn:
                log_fn(f"[{self.id}] received {message.id[:6]} (hop {message.hop_count})")

            if self.is_gateway:
                result = classify(message.body)
                message.category = result["category"]
                message.severity = result["severity"]
                message.confidence = result["confidence"]
                self.delivered.append(message)
                if log_fn:
                    flag = " [NEEDS REVIEW - low confidence]" if result["needs_review"] else ""
                    log_fn(f"[{self.id}] ** GATEWAY delivered ** {message.id[:6]} "
                           f"in {message.hop_count} hops via {message.path} -> "
                           f"{result['category']} (severity {result['severity']}, "
                           f"conf {result['confidence']:.2f}){flag}")
                continue  # gateway doesn't need to re-flood further

            if not message.is_alive():
                if log_fn:
                    log_fn(f"[{self.id}] dropped {message.id[:6]} (TTL expired)")
                continue

            for neighbor in self.neighbors:
                if random.random() < self.drop_prob:
                    continue  # simulated transmission loss
                await neighbor.receive(message)
