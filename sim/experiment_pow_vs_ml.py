# sim/experiment_pow_vs_ml.py
import time, random
import numpy as np
import matplotlib.pyplot as plt
from .state import ChainState
from .node import Node

def gen_latency(kind: str):
    # three validator profiles
    if kind == "good":
        return max( int(np.random.normal(220, 60)), 10)
    if kind == "ok":
        return max( int(np.random.normal(600, 150)), 20)
    if kind == "noisy":
        return max( int(np.random.normal(2500, 700)), 50)
    return 200

def run_scenario(ml_enabled: bool, weight_rewards: bool, ml_threshold: float):
    state = ChainState()
    node = Node(state, ml_enabled=ml_enabled, weight_rewards=weight_rewards, ml_threshold=ml_threshold)

    # seed website and validators
    wid = node.add_website("https://github.com", "demo", "0xowner")
    node.register_validator("val-good", "pk1", "IN")
    node.register_validator("val-ok", "pk2", "US")
    node.register_validator("val-noisy", "pk3", "EU")

    profiles = {"val-good": "good", "val-ok": "ok", "val-noisy": "noisy"}

    rounds = 200
    accepted = {k: 0 for k in profiles}
    submitted = {k: 0 for k in profiles}
    latencies_acc = {k: [] for k in profiles}

    for r in range(rounds):
        ts = int(time.time())
        for vid, kind in profiles.items():
            lat = gen_latency(kind)
            submitted[vid] += 1
            ok = node.submit_tick({
                "website_id": wid,
                "validator": vid,
                "status": 0,
                "latency": lat,
                "timestamp": ts
            })
            if ok:
                accepted[vid] += 1
                latencies_acc[vid].append(lat)
        node.produce_block()

    balances = {vid: v.balance for vid, v in state.validators.items()}
    return {
        "accepted": accepted,
        "submitted": submitted,
        "balances": balances,
        "latencies_acc": {k: (np.mean(v) if v else 0) for k, v in latencies_acc.items()},
        "state": state
    }

def main():
    random.seed(42)
    np.random.seed(42)

    # Baseline PoW: no ML gating, equal rewards
    baseline = run_scenario(ml_enabled=False, weight_rewards=False, ml_threshold=0.0)

    # PoW + ML: gating and weighted rewards
    with_ml = run_scenario(ml_enabled=True, weight_rewards=True, ml_threshold=0.35)

    validators = ["val-good", "val-ok", "val-noisy"]

    # Print summary
    print("\nBaseline PoW")
    for vid in validators:
        ar = baseline["accepted"][vid] / baseline["submitted"][vid]
        print(f"{vid}: accepted {baseline['accepted'][vid]}/{baseline['submitted'][vid]} ({ar:.2%}), "
              f"avg latency {baseline['latencies_acc'][vid]:.1f} ms, balance {baseline['balances'][vid]}")

    print("\nPoW + ML")
    for vid in validators:
        ar = with_ml["accepted"][vid] / with_ml["submitted"][vid]
        print(f"{vid}: accepted {with_ml['accepted'][vid]}/{with_ml['submitted'][vid]} ({ar:.2%}), "
              f"avg latency {with_ml['latencies_acc'][vid]:.1f} ms, balance {with_ml['balances'][vid]}")

    # Plot comparison
    x = np.arange(len(validators))
    width = 0.35

    fig, axs = plt.subplots(1, 3, figsize=(14, 4))

    # Acceptance rate
    acc_base = [baseline["accepted"][v]/baseline["submitted"][v] for v in validators]
    acc_ml = [with_ml["accepted"][v]/with_ml["submitted"][v] for v in validators]
    axs[0].bar(x - width/2, acc_base, width, label="PoW")
    axs[0].bar(x + width/2, acc_ml, width, label="PoW+ML")
    axs[0].set_xticks(x, validators)
    axs[0].set_ylim(0, 1)
    axs[0].set_title("Acceptance rate")
    axs[0].legend()

    # Avg accepted latency
    lat_base = [baseline["latencies_acc"][v] for v in validators]
    lat_ml = [with_ml["latencies_acc"][v] for v in validators]
    axs[1].bar(x - width/2, lat_base, width, label="PoW")
    axs[1].bar(x + width/2, lat_ml, width, label="PoW+ML")
    axs[1].set_xticks(x, validators)
    axs[1].set_title("Avg accepted latency (ms)")

    # Validator rewards
    bal_base = [baseline["balances"][v] for v in validators]
    bal_ml = [with_ml["balances"][v] for v in validators]
    axs[2].bar(x - width/2, bal_base, width, label="PoW")
    axs[2].bar(x + width/2, bal_ml, width, label="PoW+ML")
    axs[2].set_xticks(x, validators)
    axs[2].set_title("Validator rewards")
    axs[2].legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()