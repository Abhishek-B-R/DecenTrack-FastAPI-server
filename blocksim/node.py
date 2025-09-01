import time
from collections import deque
from typing import List
from .tx import UptimeReportTx
from .messages import GossipMsg
from .state import ChainState, Validator
from ml_engine.model import MLEngine

class Node:
    def __init__(self, env, network, node_id: str, state: ChainState):
        self.env = env
        self.network = network
        self.node_id = node_id
        self.state = state
        self.mempool: deque[UptimeReportTx] = deque()
        self.ml = MLEngine()
        network.subscribe(self.on_message)

    def on_message(self, msg: GossipMsg):
        if msg.type == "tx":
            self._maybe_add_tx(msg.payload)
        elif msg.type == "block":
            self._apply_block(msg.payload)

    def submit_tx_local(self, tx: UptimeReportTx):
        if self._ml_accept(tx):
            self.mempool.append(tx)
            self.network.broadcast(GossipMsg(type="tx", payload=tx.__dict__))

    def _ml_accept(self, tx: UptimeReportTx) -> bool:
        sample = {
            "gas_used": 8_000_000,
            "gas_limit": 30_000_000,
            "transaction_count": 1,
            "difficulty": 1e12,
            "total_difficulty": 1e12,
            "latency_ms": tx.latency_ms,
        }
        return self.ml.predict_quality(sample) >= 0.3

    def _tx_weight(self, tx: UptimeReportTx) -> float:
        sample = {
            "gas_used": 8_000_000,
            "gas_limit": 30_000_000,
            "transaction_count": 1,
            "difficulty": 1e12,
            "total_difficulty": 1e12,
            "latency_ms": tx.latency_ms,
        }
        return self.ml.predict_quality(sample)

    def _maybe_add_tx(self, tx_dict: dict):
        tx = UptimeReportTx(**tx_dict)
        if self._ml_accept(tx):
            self.mempool.append(tx)

    def _apply_block(self, block: dict):
        for tx in block["txs"]:
            self.state.reports.append(tx)
            vid = tx["validator_id"]
            v = self.state.validators.get(vid) or Validator(id=vid)
            v.balance += tx["reward"]
            self.state.validators[vid] = v
        self.state.chain.append(block)

    def produce_block(self):
        batch: List[UptimeReportTx] = list(self.mempool)
        self.mempool.clear()
        weights = {}
        for tx in batch:
            w = self._tx_weight(tx)
            weights[tx.validator_id] = weights.get(tx.validator_id, 0.0) + w
        total_w = sum(weights.values()) or 1.0

        txs_out = []
        for tx in batch:
            w = self._tx_weight(tx)
            reward = int(100 * (w / total_w))
            txs_out.append({**tx.__dict__, "reward": reward})

        block = {
            "producer": self.node_id,
            "time": int(time.time()),
            "txs": txs_out,
        }
        self.network.broadcast(GossipMsg(type="block", payload=block))