import simpy
import random

class SimEnv:
    def __init__(self, seed: int = 42):
        self.env = simpy.Environment()
        self.rng = random.Random(seed)