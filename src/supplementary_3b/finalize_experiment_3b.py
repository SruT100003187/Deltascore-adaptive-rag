"""
STEP 4 of 4 (3B generator comparison) — the four-seed confirmatory
protocol, run against the llama3.2:3b generator.

This is your original finalize_experiment_v3.py logic — same
threshold-selection rule (development-only, closest to a 60% Fast
target among thresholds with no significant dev-set drop), same
McNemar test, same 60-second outlier rule, same 50/50 development/
held-out split. The only changes are:
  (1) it reads the 3B answer/judge files instead of the 1B ones, and
  (2) it loops over all four seeds (42, 7, 123, 2024) automatically
      in one run, instead of requiring you to edit SEED and rerun by
      hand four times.

Output: final_results_3b_seed<N>.csv for each seed, plus a combined
final_results_3b_allseeds.csv summary at the end — directly comparable
to your original FINAL_results_allseeds.txt.
"""

import os
import numpy as np
import pandas as pd

try:
    from scipy.stats import binomtest
    def mcnemar_p(b, c):
        n = b + c
        return 1.0 if n == 0 else binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue
except ImportError:
    from scipy.stats import binom_test
    def mcnemar_p(b, c):
        n = b + c
        return 1.0 if n == 0 else binom_test(min(b, c), n, 0.5, alternative="two-sided")

# ---------------- settings (unchanged from your original protocol) ----------------
K_FAST = 3
DEV_FRACTION = 0.5
SEEDS = [42, 7, 123, 2024]      # same four seeds as your original confirmatory result
ALPHA = 0.05
OUTLIER_SECONDS = 60
TARGET_FAST_SHARE = 0.60
# ------------------------------------------------------------------------------------

# ---------------- 3B file names ----------------
BASELINE_CSV = "baseline_full_3b.csv"
FASTC_CSV = "fast_cache_K3_3b.csv"
JUDGE_BASE_CSV = "judge_baseline_t0_3b.csv"     # temperature-0 stabilised verdicts
JUDGE_FAST_CSV = "judge_fastK3_t0_3b.csv"
# --------------------------------------------------


def need(path):
    if not os.path.exists(path):
        raise SystemExit(f"Missing {path}. Run scripts 1-3 first (baseline, fastpath, rejudge).")
    return path


def p95_mean(vals):
    a = np.asarray(vals, float)
    clean = a[a <= OUTLIER_SECONDS]
    if len(clean) == 0:
        clean = a
    return round(np.percentile(clean, 95), 2), round(clean.mean(), 2)


def evaluate(part, tau):
    lat, adapt_v = [], []
    b = c = 0
    for _, r in part.iterrows():
        if r["delta"] >= tau:
            lat.append(r["fast_lat"]); av = int(r["fast_v"])
        else:
            lat.append(r["base_lat"]); av = int(r["base_v"])
        bv = int(r["base_v"])
        adapt_v.append(av)
        if bv == 1 and av == 0: b += 1
        elif bv == 0 and av == 1: c += 1
    p95, mean = p95_mean(lat)
    return {
        "tau": tau, "fast_n": int((part["delta"] >= tau).sum()),
        "p95": p95, "mean": mean,
        "faith": round(float(np.mean(adapt_v)), 3),
        "lost": b, "gained": c, "p": round(mcnemar_p(b, c), 4),
    }


def run_one_seed(df, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    cut = int(round(DEV_FRACTION * len(df)))
    dev = df.iloc[idx[:cut]].reset_index(drop=True)
    hold = df.iloc[idx[cut:]].reset_index(drop=True)

    TAU_ALL_DEEP = df["delta"].max() + 1.0
    dev_base = evaluate(dev, TAU_ALL_DEEP)
    hold_base = evaluate(hold, TAU_ALL_DEEP)

    cands = np.unique(np.percentile(dev["delta"], np.arange(2, 99, 2)))
    rows = [evaluate(dev, t) for t in cands]
    devsweep = pd.DataFrame(rows).sort_values("tau").reset_index(drop=True)

    ok = devsweep[devsweep["p"] >= ALPHA].copy()
    if len(ok):
        ok["fast_share"] = ok["fast_n"] / len(dev)
        ok["dist"] = (ok["fast_share"] - TARGET_FAST_SHARE).abs()
        chosen = ok.sort_values("dist").iloc[0]
    else:
        chosen = devsweep.sort_values("p", ascending=False).iloc[0]
    TAU_STAR = float(chosen["tau"])

    res = evaluate(hold, TAU_STAR)
    dP95 = round(100 * (res["p95"] - hold_base["p95"]) / hold_base["p95"], 1)
    dMean = round(100 * (res["mean"] - hold_base["mean"]) / hold_base["mean"], 1)

    print(f"\n{'='*66}\nSEED {seed}\n{'='*66}")
    print(f"  Chosen tau: {TAU_STAR:.4f}  (dev fast {int(chosen['fast_n'])}/{len(dev)})")
    print(f"  Held-out:   baseline P95 {hold_base['p95']}s -> adaptive P95 {res['p95']}s ({dP95:+.1f}%)")
    print(f"              baseline mean {hold_base['mean']}s -> adaptive mean {res['mean']}s ({dMean:+.1f}%)")
    print(f"              baseline faithfulness {hold_base['faith']} -> adaptive {res['faith']}")
    print(f"  McNemar:    lost {res['lost']}, gained {res['gained']}, p = {res['p']}")

    summary = pd.DataFrame([
        {"seed": seed, "set": "dev_baseline", "tau": "", "p95": dev_base["p95"], "mean": dev_base["mean"], "faith": dev_base["faith"], "fast_n": 0, "p": ""},
        {"seed": seed, "set": "dev_chosen", "tau": round(TAU_STAR, 4), "p95": chosen["p95"], "mean": chosen["mean"], "faith": chosen["faith"], "fast_n": int(chosen["fast_n"]), "p": chosen["p"]},
        {"seed": seed, "set": "heldout_baseline", "tau": "", "p95": hold_base["p95"], "mean": hold_base["mean"], "faith": hold_base["faith"], "fast_n": 0, "p": ""},
        {"seed": seed, "set": "heldout_adaptive", "tau": round(TAU_STAR, 4), "p95": res["p95"], "mean": res["mean"], "faith": res["faith"], "fast_n": res["fast_n"], "p": res["p"]},
    ])
    summary.to_csv(f"final_results_3b_seed{seed}.csv", index=False)

    return {
        "seed": seed, "tau": round(TAU_STAR, 4),
        "heldout_p95_base": hold_base["p95"], "heldout_p95_adapt": res["p95"], "p95_change_pct": dP95,
        "heldout_mean_base": hold_base["mean"], "heldout_mean_adapt": res["mean"], "mean_change_pct": dMean,
        "heldout_faith_base": hold_base["faith"], "heldout_faith_adapt": res["faith"],
        "fast_n": res["fast_n"], "held_n": len(hold), "mcnemar_p": res["p"],
    }


def main():
    base = pd.read_csv(need(BASELINE_CSV))
    fastc = pd.read_csv(need(FASTC_CSV))
    jb = pd.read_csv(need(JUDGE_BASE_CSV))
    jf = pd.read_csv(need(JUDGE_FAST_CSV))

    base_lat = dict(zip(base["question"], base["latency_seconds"]))
    delta    = dict(zip(base["question"], base["delta_score"]))
    fast_lat = dict(zip(fastc["question"], fastc["latency_seconds"]))
    base_v   = dict(zip(jb["question"], jb["faithful"].astype(int)))
    fast_v   = dict(zip(jf["question"], jf["faithful"].astype(int)))

    qs = [q for q in base["question"] if q in fast_lat and q in base_v and q in fast_v]
    df = pd.DataFrame({
        "question": qs,
        "delta":    [delta[q]    for q in qs],
        "base_lat": [base_lat[q] for q in qs],
        "fast_lat": [fast_lat[q] for q in qs],
        "base_v":   [base_v[q]   for q in qs],
        "fast_v":   [fast_v[q]   for q in qs],
    })
    print(f"Loaded {len(df)} queries with complete 3B records (generator=llama3.2:3b, K_FAST={K_FAST}).")

    all_results = [run_one_seed(df, seed) for seed in SEEDS]

    combined = pd.DataFrame(all_results)
    combined.to_csv("final_results_3b_allseeds.csv", index=False)

    print(f"\n{'='*66}\nSUMMARY ACROSS ALL FOUR SEEDS (3B generator)\n{'='*66}")
    print(combined.to_string(index=False))
    print(f"\nP95 reduction range: {combined['p95_change_pct'].min()}% to {combined['p95_change_pct'].max()}%")
    print(f"Mean reduction range: {combined['mean_change_pct'].min()}% to {combined['mean_change_pct'].max()}%")
    print(f"McNemar p-value range: {combined['mcnemar_p'].min()} to {combined['mcnemar_p'].max()}")
    print("\nSaved final_results_3b_seed<N>.csv for each seed and final_results_3b_allseeds.csv.")
    print("Compare these numbers directly against your original FINAL_results_allseeds.txt (1B).")


if __name__ == "__main__":
    main()
