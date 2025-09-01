import time
from .env import SimEnv
from .network import Network
from .state import ChainState
from .node import Node
from .consensus import RoundRobinPoA
from .tx import UptimeReportTx

def main():
    env = SimEnv()
    net = Network(env, mean_latency_ms=100)
    state = ChainState()

    nodes = []
    for i in range(3):
        n = Node(env, net, f"node-{i}", state)
        nodes.append(n)

    cons = RoundRobinPoA([n.node_id for n in nodes], block_interval_ms=2000)

    def block_loop():
        while True:
            proposer_id = cons.next_proposer()
            proposer = next(n for n in nodes if n.node_id == proposer_id)
            proposer.produce_block()
            yield env.env.timeout(cons.block_interval_ms)

    env.env.process(block_loop())

    def traffic():
        t = 0
        while t < 20000:
            tx = UptimeReportTx(
                website_id="1",
                validator_id="0xvalidatorA",
                status=1,
                latency_ms=120 + (t % 80),
                timestamp=int(time.time()),
            )
            nodes[0].submit_tx_local(tx)
            yield env.env.timeout(250)
            t += 250

    env.env.process(traffic())
    env.env.run(until=20000)

    print("Blocks:", len(state.chain))
    print("Reports:", len(state.reports))
    print("Balances:", {vid: v.balance for vid, v in state.validators.items()})

if __name__ == "__main__":
    main()