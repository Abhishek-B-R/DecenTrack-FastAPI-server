class RoundRobinPoA:
    def __init__(self, node_ids, block_interval_ms=2000):
        self.node_ids = node_ids
        self.block_interval_ms = block_interval_ms
        self.index = 0

    def next_proposer(self):
        pid = self.node_ids[self.index % len(self.node_ids)]
        self.index += 1
        return pid