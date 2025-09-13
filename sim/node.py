# sim/node.py
import time
from collections import deque
from typing import List
from .state import ChainState, Validator, Website
from ml_engine.model import MLEngine

class Node:
    def __init__(
        self,
        state: ChainState,
        block_time_s: float = 2.0,
        ml_enabled: bool = True,
        weight_rewards: bool = True,
        ml_threshold: float = 0.3,
    ):
        self.state = state
        self.block_time_s = block_time_s
        self.mempool: deque[dict] = deque()
        self.chain: List[dict] = []
        self.ml_enabled = ml_enabled
        self.weight_rewards = weight_rewards
        self.ml_threshold = ml_threshold
        self.ml = MLEngine() if ml_enabled else None

    def submit_tick(self, tick: dict) -> bool:
        sample = {
            "gas_used": 8_000_000,
            "gas_limit": 30_000_000,
            "transaction_count": 1,
            "difficulty": 1e12,
            "total_difficulty": 1e12,
            "latency_ms": tick["latency"],
        }

        if self.ml_enabled and self.ml:
            q = self.ml.predict_quality(sample)
            tick["ml_weight"] = q
            if q < self.ml_threshold:
                return False
        else:
            tick["ml_weight"] = 1.0

        self.mempool.append(tick)
        return True

    def add_website(self, url: str, contact_info: str, owner: str) -> str:
        wid = str(len(self.state.websites) + 1)
        self.state.websites[wid] = Website(
            id=wid, url=url, contact_info=contact_info, owner=owner
        )
        return wid

    def delete_website(self, website_id: str) -> bool:
        return self.state.websites.pop(website_id, None) is not None

    def register_validator(self, address: str, public_key: str, location: str):
        v = self.state.validators.get(address) or Validator(address=address)
        v.public_key = public_key
        v.location = location
        v.authenticated = True
        self.state.validators[address] = v
        return v

    def produce_block(self):
        batch = list(self.mempool)
        self.mempool.clear()

        reward_pool = 100
        weights = {}
        for tx in batch:
            vid = tx["validator"]
            w = tx.get("ml_weight", 1.0) if self.weight_rewards else 1.0
            weights[vid] = weights.get(vid, 0.0) + w

            v = self.state.validators.get(tx["validator"])
            loc = v.location if v else "sim-location"

            self.state.reports.append(
                {
                    "validator": tx["validator"],
                    "createdAt": tx.get("timestamp", int(time.time())),
                    "status": tx["status"],
                    "latency": tx["latency"],
                    "location": loc,
                    "ml_weight": tx.get("ml_weight", 1.0),
                    "website_id": tx["website_id"],
                }
            )

        total_w = sum(weights.values()) or 1.0
        for vid, w in weights.items():
            share = int(reward_pool * (w / total_w))
            v = self.state.validators.get(vid) or Validator(address=vid)
            v.balance += share
            self.state.validators[vid] = v

        self.chain.append({"time": int(time.time()), "txs": len(batch), "weights": weights})

    def run_tick(self):
        self.produce_block()

# import time
# from collections import deque
# from typing import List
# from .state import ChainState, Validator, Website
# from ml_engine.model import MLEngine

# class Node:
#     def __init__(self, state: ChainState, block_time_s: float = 2.0):
#         self.state = state
#         self.block_time_s = block_time_s
#         self.mempool: deque[dict] = deque()
#         self.chain: List[dict] = []
#         self.ml = MLEngine()

#     def submit_tick(self, tick: dict) -> bool:
#         # Call ML model before accepting
#         sample = {
#             "gas_used": 8_000_000,
#             "gas_limit": 30_000_000,
#             "transaction_count": 1,
#             "difficulty": 1e12,
#             "total_difficulty": 1e12,
#             "latency_ms": tick["latency"],
#         }
#         q = self.ml.predict_quality(sample)
#         if q < 0.3:
#             return False
#         tick["ml_weight"] = q
#         self.mempool.append(tick)
#         return True

#     def add_website(self, url: str, contact_info: str, owner: str) -> str:
#         wid = str(len(self.state.websites) + 1)
#         self.state.websites[wid] = Website(
#             id=wid, url=url, contact_info=contact_info, owner=owner
#         )
#         return wid

#     def delete_website(self, website_id: str) -> bool:
#         return self.state.websites.pop(website_id, None) is not None

#     def register_validator(self, address: str, public_key: str, location: str):
#         v = self.state.validators.get(address) or Validator(address=address)
#         v.public_key = public_key
#         v.location = location
#         v.authenticated = True
#         self.state.validators[address] = v
#         return v

#     def produce_block(self):
#         batch = list(self.mempool)
#         self.mempool.clear()

#         reward_pool = 100
#         weights = {}
#         for tx in batch:
#             vid = tx["validator"]
#             w = tx.get("ml_weight", 0.0)
#             weights[vid] = weights.get(vid, 0.0) + w
#             # ✅ always store createdAt
#             self.state.reports.append({
#                 "validator": tx["validator"],
#                 "createdAt": tx.get("timestamp", int(time.time())),
#                 "status": tx["status"],
#                 "latency": tx["latency"],
#                 "location": (
#                     self.state.validators.get(tx["validator"]).location
#                     if tx["validator"] in self.state.validators else "sim-location"
#                 ),
#                 "ml_weight": tx.get("ml_weight", 1.0),
#                 "website_id": tx["website_id"],
#             })

#         total_w = sum(weights.values()) or 1.0
#         for vid, w in weights.items():
#             share = int(reward_pool * (w / total_w))
#             v = self.state.validators.get(vid) or Validator(address=vid)
#             v.balance += share
#             self.state.validators[vid] = v

#         self.chain.append(
#             {"time": int(time.time()), "txs": len(batch), "weights": weights}
#         )

#     def run_tick(self):
#         self.produce_block()