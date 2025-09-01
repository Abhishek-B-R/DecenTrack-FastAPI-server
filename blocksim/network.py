from typing import List, Callable
from .messages import GossipMsg

class Network:
    def __init__(self, env, mean_latency_ms: int = 100):
        self.env = env
        self.mean_latency_ms = mean_latency_ms
        self.subscribers: List[Callable[[GossipMsg], None]] = []

    def subscribe(self, handler: Callable[[GossipMsg], None]):
        self.subscribers.append(handler)

    def broadcast(self, msg: GossipMsg):
        for h in self.subscribers:
            # exponential delay with mean mean_latency_ms
            delay = max(1, int(self.env.rng.expovariate(1 / self.mean_latency_ms)))
            self.env.env.process(self._deliver(h, msg, delay))

    def _deliver(self, handler, msg, delay_ms):
        yield self.env.env.timeout(delay_ms)
        handler(msg)