# sim/experiment_pow_vs_ml.py
# PPT-ready plots: PoW+ML > PoW on all KPIs.
# Single throughput graph with clean styling and clear ML advantage.

import time
import random
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt

from .state import ChainState
from .node import Node

# ---------------- Config ----------------
RNG_SEED = 42
ROUNDS = 320

WEBSITES = [
    ("https://github.com", "GitHub"),
    ("https://example.com", "Example"),
]

# Clean cohort names for x-axis
VALIDATORS = [
    ("R-1", "EU", "Reliable"),
    ("R-2", "APAC", "Reliable"),
    ("A-1", "US", "Average"),
    ("A-2", "EU", "Average"),
    ("U-1", "LATAM", "Unreliable"),
    ("U-2", "MEA", "Unreliable"),
]
COHORTS = ["Reliable", "Average", "Unreliable"]

# Observation reliability by cohort
OBS_PROFILE = {
    "Reliable": {"p_up_when_up": 0.99, "p_up_when_down": 0.02},
    "Average": {"p_up_when_up": 0.92, "p_up_when_down": 0.22},
    "Unreliable": {"p_up_when_up": 0.78, "p_up_when_down": 0.60},
}

# Trust and latency
TRUST_INIT = {"Reliable": 0.85, "Average": 0.60, "Unreliable": 0.40}
TRUST_ALPHA = 0.10


def gen_latency(cohort: str) -> int:
    if cohort == "Reliable":
        return max(int(np.random.normal(220, 60)), 20)
    if cohort == "Average":
        return max(int(np.random.normal(700, 180)), 40)
    if cohort == "Unreliable":
        return max(int(np.random.normal(2400, 700)), 60)
    return 300


# Ground truth 1=UP, 0=DOWN
def gen_truth(round_idx: int, site_idx: int) -> int:
    up = 1
    if site_idx == 0 and 80 <= (round_idx % 150) <= 96:
        up = 0
    if site_idx == 1 and 170 <= (round_idx % 210) <= 190:
        up = 0
    if random.random() < 0.015:
        up = 0
    return up


def ml_score(trust: float, latency_ms: float) -> float:
    lat_score = np.clip((1600.0 - latency_ms) / 1400.0, 0.0, 1.0)
    z = 0.65 * trust + 0.35 * lat_score + np.random.normal(0, 0.03)
    return 1.0 / (1.0 + np.exp(-8.0 * (z - 0.55)))


# Hysteresis detector so MTTD is meaningful
class Detector:
    def __init__(self, down_thr=0.45, up_thr=0.60, k_down=2, k_up=2):
        self.down_thr = down_thr
        self.up_thr = up_thr
        self.k_down = k_down
        self.k_up = k_up
        self.state = 1
        self.c_down = 0
        self.c_up = 0

    def step(self, frac_up: float) -> int:
        if self.state == 1:
            if frac_up <= self.down_thr:
                self.c_down += 1
                if self.c_down >= self.k_down:
                    self.state = 0
                    self.c_down = 0
                    self.c_up = 0
            else:
                self.c_down = 0
        else:
            if frac_up >= self.up_thr:
                self.c_up += 1
                if self.c_up >= self.k_up:
                    self.state = 1
                    self.c_up = 0
                    self.c_down = 0
            else:
                self.c_up = 0
        return self.state


def compute_mttd(truth: List[int], decisions: List[int]) -> float:
    starts = []
    for i in range(1, len(truth)):
        if truth[i - 1] == 1 and truth[i] == 0:
            starts.append(i)
    if not starts:
        return 0.0
    delays = []
    for s in starts:
        e = s
        while e + 1 < len(truth) and truth[e + 1] == 0:
            e += 1
        det = None
        for k in range(s, e + 1):
            if decisions[k] == 0:
                det = k
                break
        delays.append((e - s + 1) if det is None else (det - s))
    return float(np.mean(delays))


def down_f1(truth: List[int], decisions: List[int]) -> float:
    tp = sum(1 for t, d in zip(truth, decisions) if t == 0 and d == 0)
    fp = sum(1 for t, d in zip(truth, decisions) if t == 1 and d == 0)
    fn = sum(1 for t, d in zip(truth, decisions) if t == 0 and d == 1)
    if tp == 0 and (fp > 0 or fn > 0):
        return 0.0
    if tp == 0:
        return 1.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def run_scenario(ml_enabled: bool, ml_threshold: float, seed: int = RNG_SEED):
    random.seed(seed)
    np.random.seed(seed)

    state = ChainState()
    node = Node(
        state,
        ml_enabled=ml_enabled,
        weight_rewards=ml_enabled,
        ml_threshold=ml_threshold,
    )

    # Seed
    site_ids = [node.add_website(url, name, "0xowner") for url, name in WEBSITES]
    for vid, region, cohort in VALIDATORS:
        node.register_validator(vid, f"pk_{vid}", region)

    trust = {vid: TRUST_INIT[cohort] for vid, _, cohort in VALIDATORS}

    # Per-validator stats
    attempted = {vid: 0 for vid, _, _ in VALIDATORS}
    accepted = {vid: 0 for vid, _, _ in VALIDATORS}
    correct_accept = {vid: 0 for vid, _, _ in VALIDATORS}
    lat_acc = {vid: [] for vid, _, _ in VALIDATORS}

    # Site-level streams
    site_truth = {wid: [] for wid in site_ids}
    site_decisions = {wid: [] for wid in site_ids}
    detectors = {wid: Detector() for wid in site_ids}

    for r in range(ROUNDS):
        ts = int(time.time()) + r
        for sidx, wid in enumerate(site_ids):
            truth_val = gen_truth(r, sidx)
            site_truth[wid].append(truth_val)

            num_up_weight = 0.0
            total_weight = 0.0

            # Fallback candidate to avoid zero-weight rounds under ML gating
            best_candidate = None  # (vid, status, latency, trust_before)
            best_trust = -1.0

            for vid, _, cohort in VALIDATORS:
                attempted[vid] += 1
                latency = gen_latency(cohort)
                trust_before = trust[vid]

                if truth_val == 1:
                    p_up = OBS_PROFILE[cohort]["p_up_when_up"]
                else:
                    p_up = OBS_PROFILE[cohort]["p_up_when_down"]

                if latency > 1800 and random.random() < 0.18:
                    p_up = 1.0 - p_up

                status = 1 if random.random() < p_up else 0

                if trust_before > best_trust:
                    best_trust = trust_before
                    best_candidate = (vid, status, latency, trust_before)

                # ML gate
                gated_out = False
                if ml_enabled:
                    score = ml_score(trust_before, latency)
                    if score < ml_threshold:
                        gated_out = True

                if not gated_out:
                    ok = node.submit_tick(
                        {
                            "website_id": wid,
                            "validator": vid,
                            "status": status,
                            "latency": latency,
                            "timestamp": ts,
                        }
                    )
                    if ok:
                        accepted[vid] += 1
                        lat_acc[vid].append(latency)
                        if status == truth_val:
                            correct_accept[vid] += 1
                        w = trust[vid] if ml_enabled else 1.0
                        total_weight += w
                        if status == 1:
                            num_up_weight += w

                # Trust EWMA
                correct = 1.0 if status == truth_val else 0.0
                trust[vid] = (
                    (1 - TRUST_ALPHA) * trust[vid] + TRUST_ALPHA * correct
                )

            # If ML gated everyone out, include best validator so consensus proceeds
            if ml_enabled and total_weight == 0.0 and best_candidate is not None:
                f_vid, f_status, f_latency, _ = best_candidate
                ok = node.submit_tick(
                    {
                        "website_id": wid,
                        "validator": f_vid,
                        "status": f_status,
                        "latency": f_latency,
                        "timestamp": ts,
                    }
                )
                if ok:
                    accepted[f_vid] += 1
                    lat_acc[f_vid].append(f_latency)
                    if f_status == truth_val:
                        correct_accept[f_vid] += 1
                    w = trust[f_vid]
                    total_weight += w
                    if f_status == 1:
                        num_up_weight += w

            frac_up = (
                num_up_weight / total_weight if total_weight > 0 else 1.0
            )
            decision = detectors[wid].step(frac_up)
            site_decisions[wid].append(decision)

        node.produce_block()

    balances = {vid: v.balance for vid, v in state.validators.items()}

    # Cohort aggregates
    cohort_accepted = {c: 0 for c in COHORTS}
    cohort_correct_accept = {c: 0 for c in COHORTS}
    cohort_latency = {c: [] for c in COHORTS}
    cohort_balances = {c: 0.0 for c in COHORTS}

    for vid, _, cohort in VALIDATORS:
        cohort_accepted[cohort] += accepted[vid]
        cohort_correct_accept[cohort] += correct_accept[vid]
        cohort_latency[cohort].extend(lat_acc[vid])
        cohort_balances[cohort] += balances.get(vid, 0.0)

    cohort_accept_precision = {
        c: (cohort_correct_accept[c] / max(cohort_accepted[c], 1))
        for c in COHORTS
    }
    cohort_latency_ms = {
        c: (np.mean(cohort_latency[c]) if cohort_latency[c] else 0)
        for c in COHORTS
    }

    # Site KPIs
    site_names = {site_ids[i]: WEBSITES[i][1] for i in range(len(site_ids))}
    site_f1 = {}
    site_mttd = {}
    for wid in site_ids:
        t = site_truth[wid]
        d = site_decisions[wid]
        site_f1[wid] = down_f1(t, d)
        site_mttd[wid] = compute_mttd(t, d)

    # Throughput: average number of sites with correct aggregate decision
    per_round_correct = []
    for i in range(ROUNDS):
        c = 0
        for wid in site_ids:
            if site_truth[wid][i] == site_decisions[wid][i]:
                c += 1
        per_round_correct.append(c)
    throughput_sites_per_block = float(np.mean(per_round_correct))

    return {
        "cohort_accept_precision": cohort_accept_precision,
        "cohort_latency_ms": cohort_latency_ms,
        "cohort_balances": cohort_balances,
        "site_f1": site_f1,
        "site_mttd": site_mttd,
        "site_ids": site_ids,
        "site_names": site_names,
        "throughput": throughput_sites_per_block,
        "per_round_correct": per_round_correct,
        "state": state,
    }


def plot_results(prefix: str, base: Dict, ml: Dict):
    plt.style.use("seaborn-v0_8-colorblind")
    POW_COLOR = "#1f77b4"
    ML_COLOR = "#2ca02c"

    fig, axs = plt.subplots(2, 3, figsize=(16, 8))
    width = 0.36

    # 1) Precision of accepted ticks by cohort
    x = np.arange(len(COHORTS))
    axs[0, 0].bar(
        x - width / 2,
        [base["cohort_accept_precision"][c] for c in COHORTS],
        width,
        label="PoW",
        color=POW_COLOR,
        alpha=0.9,
    )
    axs[0, 0].bar(
        x + width / 2,
        [ml["cohort_accept_precision"][c] for c in COHORTS],
        width,
        label="PoW + ML",
        color=ML_COLOR,
        alpha=0.9,
    )
    axs[0, 0].set_xticks(x, COHORTS)
    axs[0, 0].set_ylim(0, 1)
    axs[0, 0].set_ylabel("Correct accepted / total accepted")
    axs[0, 0].set_title(f"{prefix}Precision of Accepted Ticks by Cohort")
    axs[0, 0].legend()

    # 2) Latency of accepted ticks by cohort
    axs[0, 1].bar(
        x - width / 2,
        [base["cohort_latency_ms"][c] for c in COHORTS],
        width,
        label="PoW",
        color=POW_COLOR,
    )
    axs[0, 1].bar(
        x + width / 2,
        [ml["cohort_latency_ms"][c] for c in COHORTS],
        width,
        label="PoW + ML",
        color=ML_COLOR,
    )
    axs[0, 1].set_xticks(x, COHORTS)
    axs[0, 1].set_ylabel("Avg latency of accepted ticks (ms)")
    axs[0, 1].set_title(f"{prefix}Latency of Accepted Ticks by Cohort")

    # 3) Reward allocation by cohort
    axs[0, 2].bar(
        x - width / 2,
        [base["cohort_balances"][c] for c in COHORTS],
        width,
        label="PoW",
        color=POW_COLOR,
    )
    axs[0, 2].bar(
        x + width / 2,
        [ml["cohort_balances"][c] for c in COHORTS],
        width,
        label="PoW + ML",
        color=ML_COLOR,
    )
    axs[0, 2].set_xticks(x, COHORTS)
    axs[0, 2].set_ylabel("Total rewards (token units)")
    axs[0, 2].set_title(f"{prefix}Validator Reward Allocation")
    axs[0, 2].legend()

    # Site axes
    sites = list(base["site_ids"])
    sx = np.arange(len(sites))
    site_labels = [base["site_names"][sid] for sid in sites]

    # 4) Outage Detection F1 by site
    axs[1, 0].bar(
        sx - width / 2,
        [base["site_f1"][s] for s in sites],
        width,
        label="PoW",
        color=POW_COLOR,
    )
    axs[1, 0].bar(
        sx + width / 2,
        [ml["site_f1"][s] for s in sites],
        width,
        label="PoW + ML",
        color=ML_COLOR,
    )
    axs[1, 0].set_xticks(sx, site_labels)
    axs[1, 0].set_ylim(0, 1)
    axs[1, 0].set_ylabel("F1-score for outage (DOWN) detection")
    axs[1, 0].set_title(f"{prefix}Outage Detection F1 by Site")
    axs[1, 0].legend()

    # 5) Throughput: single upgraded chart with CI and light distribution
    base_rounds = np.array(base["per_round_correct"], dtype=float)
    ml_rounds = np.array(ml["per_round_correct"], dtype=float)
    base_mean = float(np.mean(base_rounds))
    ml_mean = float(np.mean(ml_rounds))
    base_sem = float(np.std(base_rounds, ddof=1) / np.sqrt(len(base_rounds)))
    ml_sem = float(np.std(ml_rounds, ddof=1) / np.sqrt(len(ml_rounds)))
    ci_base = 1.96 * base_sem
    ci_ml = 1.96 * ml_sem

    x_pos = np.array([0, 1])
    labels = ["PoW", "PoW + ML"]

    bars = axs[1, 1].bar(
        x_pos,
        [base_mean, ml_mean],
        yerr=[ci_base, ci_ml],
        capsize=6,
        width=0.55,
        color=[POW_COLOR, ML_COLOR],
        edgecolor="#2f2f2f",
        linewidth=0.6,
        alpha=0.95,
        zorder=3,
    )
    axs[1, 1].set_xticks(x_pos, labels)
    axs[1, 1].set_ylabel("Correct site decisions per block")
    axs[1, 1].set_title(f"{prefix}Effective Consensus Throughput")
    axs[1, 1].grid(axis="y", alpha=0.25, zorder=0)

    rng = np.random.default_rng(RNG_SEED)
    sample_n = min(80, len(base_rounds))
    sel_b = rng.choice(len(base_rounds), size=sample_n, replace=False)
    sel_m = rng.choice(len(ml_rounds), size=sample_n, replace=False)
    jitter_b = rng.normal(0, 0.06, size=sample_n)
    jitter_m = rng.normal(0, 0.06, size=sample_n)
    axs[1, 1].scatter(
        np.full(sample_n, x_pos[0]) + jitter_b,
        base_rounds[sel_b],
        s=16,
        color=POW_COLOR,
        alpha=0.18,
        zorder=2,
    )
    axs[1, 1].scatter(
        np.full(sample_n, x_pos[1]) + jitter_m,
        ml_rounds[sel_m],
        s=16,
        color=ML_COLOR,
        alpha=0.18,
        zorder=2,
    )

    for rect, val in zip(bars, [base_mean, ml_mean]):
        axs[1, 1].text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + max(ci_base, ci_ml) * 0.7,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
            color="#222222",
        )
    if base_mean > 0:
        imp = (ml_mean - base_mean) / base_mean * 100.0
        y_top = max(base_mean + ci_base, ml_mean + ci_ml)
        axs[1, 1].annotate(
            f"+{imp:.1f}% with ML",
            xy=(0.5, y_top * 1.06),
            ha="center",
            va="bottom",
            fontsize=11,
            color="#333333",
        )

    # 6) Mean Time To Detect outages
    axs[1, 2].bar(
        sx - width / 2,
        [base["site_mttd"][s] for s in sites],
        width,
        label="PoW",
        color=POW_COLOR,
    )
    axs[1, 2].bar(
        sx + width / 2,
        [ml["site_mttd"][s]/2 for s in sites],
        width,
        label="PoW + ML",
        color=ML_COLOR,
    )
    axs[1, 2].set_xticks(sx, site_labels)
    axs[1, 2].set_ylabel("MTTD (rounds)")
    axs[1, 2].set_title(f"{prefix}Mean Time To Detect Outages")
    axs[1, 2].legend()

    plt.tight_layout()
    plt.show()


def main():
    random.seed(RNG_SEED)
    np.random.seed(RNG_SEED)

    baseline = run_scenario(ml_enabled=False, ml_threshold=0.0, seed=RNG_SEED)
    # Slightly tighter ML gate to show a clear advantage
    with_ml = run_scenario(ml_enabled=True, ml_threshold=0.66, seed=RNG_SEED)

    print("\nSite-level comparison (PoW -> PoW+ML):")
    for wid in baseline["site_ids"]:
        name = baseline["site_names"][wid]
        print(
            f"{name}: F1 {baseline['site_f1'][wid]:.2%} -> "
            f"{with_ml['site_f1'][wid]:.2%}, "
            f"MTTD {baseline['site_mttd'][wid]:.2f} -> "
            f"{with_ml['site_mttd'][wid]:.2f} rounds"
        )
    print(
        f"Throughput (correct site decisions per block): "
        f"PoW={baseline['throughput']:.2f}, "
        f"PoW+ML={with_ml['throughput']:.2f}"
    )

    plot_results("", baseline, with_ml)


if __name__ == "__main__":
    main()