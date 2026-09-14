"""
SOS message schema for the mesh network simulation.

Every message that enters the mesh carries enough metadata to:
  - be routed hop-by-hop without a central coordinator (id, ttl, path)
  - be triaged by the ML classifier once it reaches a gateway (body, category, severity)
"""

from dataclasses import dataclass, field
from enum import Enum
import time
import uuid


class Category(str, Enum):
    TRAPPED_INJURED = "trapped_injured"
    MEDICAL = "medical"
    STRUCTURAL_RESCUE = "structural_rescue"
    WATER_FOOD = "water_food"
    EVACUATION = "evacuation"
    INFORMATIONAL = "informational"
    UNCLASSIFIED = "unclassified"


@dataclass
class SOSMessage:
    body: str                      # free-text or templated message from the sender
    sender_id: str                 # originating node id
    lat: float                     # simulated coordinates (for map + geolocation demo)
    lon: float
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    created_at: float = field(default_factory=time.time)
    ttl: int = 8                   # max hops before the message is dropped
    hop_count: int = 0
    path: list = field(default_factory=list)   # list of node ids the message passed through

    # filled in once the classifier scores the message at the gateway
    category: str = Category.UNCLASSIFIED.value
    severity: int = None           # 1 (low) - 5 (critical)
    confidence: float = None

    def record_hop(self, node_id: str):
        self.hop_count += 1
        self.ttl -= 1
        self.path.append(node_id)

    def is_alive(self) -> bool:
        return self.ttl > 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "body": self.body,
            "sender_id": self.sender_id,
            "lat": self.lat,
            "lon": self.lon,
            "created_at": self.created_at,
            "hop_count": self.hop_count,
            "path": self.path,
            "category": self.category,
            "severity": self.severity,
            "confidence": self.confidence,
        }
