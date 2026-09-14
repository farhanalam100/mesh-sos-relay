"""
Backend for the live dashboard. Builds the mesh topology once per browser
connection and keeps it fixed -- the graph stays visible and stable the
whole time. Each "Run disaster simulation" click sends a fresh batch of
SOS messages through that SAME mesh, one at a time (each message's full
journey completes, or times out, before the next one starts), so the
viewer can actually follow a single signal traveling hop by hop instead
of watching several overlap at once.

Messages are classified immediately when created (not only on arrival at
the gateway), so severity is known from the very first hop and every
pulse in the mesh can be colored by urgency as it travels.

A "stop" command from the browser interrupts the flood between messages.

Run: uvicorn backend.app:app --reload --port 8000
"""

import asyncio
import random

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from src.node import Node
from src.message import SOSMessage
from src.network import RANGE_RADIUS, SAMPLE_MESSAGES
from src.classifier import classify

app = FastAPI()

HOP_DELAY = 0.8
MAX_HOPS = 8
MESSAGE_TIMEOUT = HOP_DELAY * MAX_HOPS + 1.5


def build_network(num_nodes: int = 14, area: float = 100.0, seed: int | None = None):
    if seed is not None:
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


def topology_payload(nodes):
    return {
        "type": "topology",
        "nodes": [
            {"id": n.id, "x": n.x, "y": n.y, "is_gateway": n.is_gateway,
             "neighbors": [nb.id for nb in n.neighbors]}
            for n in nodes
        ],
    }


@app.websocket("/ws/simulate")
async def simulate(ws: WebSocket):
    await ws.accept()

    nodes = build_network(seed=random.randint(0, 100000))
    await ws.send_json(topology_payload(nodes))

    events = asyncio.Queue()
    delivery_events: dict[str, asyncio.Event] = {}
    needs_review_by_id: dict[str, bool] = {}
    stop_event = asyncio.Event()
    start_queue: asyncio.Queue = asyncio.Queue()

    async def instrumented_run(node: Node):
        while not stop_event.is_set():
            message = await node.inbox.get()
            if message.id in node.seen_ids:
                continue
            node.seen_ids.add(message.id)
            message.record_hop(node.id)

            await asyncio.sleep(HOP_DELAY)

            from_node = message.path[-2] if len(message.path) >= 2 else message.sender_id
            await events.put({
                "type": "hop",
                "message_id": message.id,
                "from_node": from_node,
                "to_node": node.id,
                "hop_count": message.hop_count,
                "category": message.category,
                "severity": message.severity,
            })

            if node.is_gateway:
                node.delivered.append(message)
                await events.put({
                    "type": "delivered",
                    "message": message.to_dict(),
                    "needs_review": needs_review_by_id.get(message.id, False),
                })
                if message.id in delivery_events:
                    delivery_events[message.id].set()
                continue

            if not message.is_alive():
                continue

            for neighbor in node.neighbors:
                if random.random() < node.drop_prob:
                    continue
                await neighbor.receive(message)

    node_tasks = [asyncio.create_task(instrumented_run(n)) for n in nodes]

    async def forward_events():
        while not stop_event.is_set():
            event = await events.get()
            await ws.send_json(event)

    forward_task = asyncio.create_task(forward_events())

    async def run_flood(num_messages: int = 12):
        stop_event.clear()
        await ws.send_json({"type": "run_start"})
        senders = [n for n in nodes if not n.is_gateway]
        # Send messages quickly without waiting for each to finish so the
        # frontend sees activity immediately. Delivery events are still
        # emitted asynchronously by the instrumented_run tasks.
        for _ in range(num_messages):
            if stop_event.is_set():
                break

            sender = random.choice(senders)
            msg = SOSMessage(
                body=random.choice(SAMPLE_MESSAGES),
                sender_id=sender.id,
                lat=sender.x,
                lon=sender.y,
            )
            result = classify(msg.body)
            msg.category = result["category"]
            msg.severity = result["severity"]
            msg.confidence = result["confidence"]
            needs_review_by_id[msg.id] = result["needs_review"]

            # register delivery watcher but do not block the loop on it
            delivery_events[msg.id] = asyncio.Event()
            await sender.receive(msg)

            # small inter-send pause so the UI sees a steady stream
            await asyncio.sleep(0.18)

        # allow a short grace period for in-flight deliveries to arrive
        await asyncio.sleep(0.8)

        await ws.send_json({"type": "done"})

    async def listener():
        try:
            while True:
                text = await ws.receive_text()
                if text == "start":
                    await start_queue.put(True)
                elif text == "stop":
                    stop_event.set()
        except WebSocketDisconnect:
            stop_event.set()

    listener_task = asyncio.create_task(listener())

    try:
        while True:
            await start_queue.get()
            await run_flood()
    except WebSocketDisconnect:
        pass
    finally:
        listener_task.cancel()
        for t in node_tasks:
            t.cancel()
        forward_task.cancel()


app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
