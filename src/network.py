"""
Builds a random mesh topology, seeds it with SOS messages from scattered nodes,
and runs the simulation until messages either reach the gateway or die out.

Run directly:  python -m src.network
"""

import asyncio
import random

from src.node import Node
from src.message import SOSMessage

RANGE_RADIUS = 34.0   # two nodes are neighbors if within this simulated distance


def build_network(num_nodes: int = 12, area: float = 100.0, seed: int = 42) -> list[Node]:
    random.seed(seed)
    nodes = [
        Node(node_id=f"n{i}", x=random.uniform(0, area), y=random.uniform(0, area))
        for i in range(num_nodes)
    ]
    nodes[0].is_gateway = True
    nodes[0].id = "gateway"

    for a in nodes:
        for b in nodes:
            if a is not b and a.distance_to(b) <= RANGE_RADIUS:
                a.neighbors.append(b)

    return nodes


SAMPLE_MESSAGES = [
    "Building collapsed, two people trapped under debris, need rescue team now",
    "My father is having chest pains, no ambulance can reach us, please send help",
    "We are safe on the roof but water is rising fast, need evacuation boat",
    "Just checking if anyone in the family group has heard from mom",
    "Running low on drinking water, six of us stranded near the old bridge",
    "Road is blocked, can someone confirm if the shelter at the school is open",
]


async def seed_and_run(nodes: list[Node], num_messages: int = 6, duration: float = 3.0):
    log = []
    def log_fn(line):
        log.append(line)

    tasks = [asyncio.create_task(n.run(log_fn=log_fn)) for n in nodes]

    senders = [n for n in nodes if not n.is_gateway]
    for i in range(num_messages):
        sender = random.choice(senders)
        msg = SOSMessage(
            body=SAMPLE_MESSAGES[i % len(SAMPLE_MESSAGES)],
            sender_id=sender.id,
            lat=sender.x,
            lon=sender.y,
        )
        await sender.receive(msg)

    await asyncio.sleep(duration)  # let messages propagate through the mesh
    for t in tasks:
        t.cancel()

    gateway = next(n for n in nodes if n.is_gateway)
    return gateway.delivered, log


if __name__ == "__main__":
    nodes = build_network()
    delivered, log = asyncio.run(seed_and_run(nodes, num_messages=len(SAMPLE_MESSAGES)))

    print("\n".join(log))

    # this is the priority queue behavior: worst cases surface first,
    # regardless of the order messages happened to arrive in
    delivered_sorted = sorted(delivered, key=lambda m: (-m.severity, -m.confidence))

    print(f"\n--- {len(delivered)} message(s) reached the gateway, sorted by urgency ---")
    for m in delivered_sorted:
        flag = "  [NEEDS REVIEW]" if m.confidence < 0.4 else ""
        print(f"  sev={m.severity} [{m.category:18s} conf={m.confidence:.2f}] "
              f"\"{m.body[:55]}...\" ({m.hop_count} hops){flag}")
