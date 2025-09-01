from dataclasses import dataclass

@dataclass
class GossipMsg:
    type: str  # "tx", "block"
    payload: dict