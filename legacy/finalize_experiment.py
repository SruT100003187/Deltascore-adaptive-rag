"""
finalize_experiment.py  -  the final experiment that answers all three research questions.

WHAT IT DOES (all from files you already have, no answer generation)
  1. Loads the per-query results your sweep already produced.
  2. Splits the 400 queries into a DEVELOPMENT set and a HELD-OUT set.
  3. On the DEVELOPMENT set only, sweeps the threshold and selects the best one
     by a rule fixed in advance  ->  answers RQ3.
  4. Freezes that threshold and evaluates it ONCE on the HELD-OUT set,
     comparing the adaptive system against the static baseline  ->  answers RQ2.
  5. Builds a per-query routing confusion matrix on the held-out set,
     checking whether DeltaScore sends the deep-needed queries to the Deep Path
     ->  answers RQ1.

WHY THIS IS VALID
  The answers and their faithfulness verdicts were computed once, earlier. The
  threshold is chosen using ONLY the development partition; the held-out partition
  is used only for the final measurement and played no part in any choice. This is
  a standard and defensible way to obtain an unbiased final result from
  pre-computed predictions.

FILES IT READS (from your sweep, in the same folder)
  baseline_full.csv      question, delta_score, latency_seconds   (baseline = Deep, K=5)
  fast_cache_K3.csv      question, latency_seconds                 (Fast Path at K=3)
  judge_baseline.csv     question, faithful                        (baseline verdicts)
  judge_fastK3.csv       question, faithful                        (Fast K=3 verdicts)

HOW TO RUN
  python finalize_experiment.py
  Instant. No model calls. Writes final_results.csv and prints an RQ-by-RQ summary.

  Optional but recommended first: re-judge at temperature 0 so the verdicts are
  repeatable (ask for rejudge_temp0.py). This script then uses the steadier files.
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

# ---------------- settings you may change ----------------
K_FAST = 3               # Fast Path depth to finalise (your result says 3)
DEV_FRACTION = 0.5       # half for choosing the threshold, half held out
SEED = 7                # fixed so the split is reproducible
ALPHA = 0.05             # significance level for "no significant drop"
OUTLIER_SECONDS = 60     # same outlier rule as the rest of the thesis
# selection rule: most aggressive threshold whose DEV faithfulness drop is not
# statistically significant (McNemar p >= ALPHA). Falls back to the safest one.
# ---------------------------------------------------------

def need(path):
    if not os.path.exists(path):
        raise SystemExit(f"Missing {path}. Run this in your Implementation_RAG folder after the sweep.")
    return path

base = pd.read_csv(need("baseline_full.csv"))
fastc = pd.read_csv(need(f"fast_cache_K{K_FAST}.csv"))
jb = pd.read_csv(need("judge_baseline.csv"))
jf = pd.read_csv(need(f"judge_fastK{K_FAST}.csv"))

# assemble one row per query
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
print(f"Loaded {len(df)} queries with complete records (K_FAST={K_FAST}).\n")

# ---------------- split ----------------
rng = np.random.default_rng(SEED)
idx = rng.permutation(len(df))
cut = int(round(DEV_FRACTION * len(df)))
dev = df.iloc[idx[:cut]].reset_index(drop=True)
hold = df.iloc[idx[cut:]].reset_index(drop=True)
print(f"Development set: {len(dev)} queries.   Held-out set: {len(hold)} queries.\n")

def p95_mean(vals):
    a = np.asarray(vals, float); clean = a[a <= OUTLIER_SECONDS]
    if len(clean) == 0: clean = a
    return round(np.percentile(clean, 95), 2), round(clean.mean(), 2)

def evaluate(part, tau):
    """Apply threshold tau to a partition; return metrics and discordant counts vs baseline."""
    lat, adapt_v, base_vv = [], [], []
    b = c = 0
    for _, r in part.iterrows():
        if r["delta"] >= tau:            # Fast Path at K_FAST
            lat.append(r["fast_lat"]); av = int(r["fast_v"])
        else:                             # Deep Path = baseline
            lat.append(r["base_lat"]); av = int(r["base_v"])
        bv = int(r["base_v"])
        adapt_v.append(av); base_vv.append(bv)
        if bv == 1 and av == 0: b += 1
        elif bv == 0 and av == 1: c += 1
    p95, mean = p95_mean(lat)
    return {
        "tau": tau, "fast_n": int((part["delta"] >= tau).sum()),
        "p95": p95, "mean": mean,
        "faith": round(float(np.mean(adapt_v)), 3),
        "lost": b, "gained": c, "p": round(mcnemar_p(b, c), 4),
    }

# baseline reference on each partition (tau above max delta -> everything Deep)
TAU_ALL_DEEP = df["delta"].max() + 1.0
dev_base = evaluate(dev, TAU_ALL_DEEP)
hold_base = evaluate(hold, TAU_ALL_DEEP)

# ---------------- choose threshold on DEV ----------------
# candidate thresholds = a fine grid of DEV DeltaScore percentiles (distribution-free)
cands = np.unique(np.percentile(dev["delta"], np.arange(2, 99, 2)))
rows = [evaluate(dev, t) for t in cands]
devsweep = pd.DataFrame(rows).sort_values("tau").reset_index(drop=True)

# rule: among thresholds with no significant DEV faithfulness drop (p >= ALPHA),
# pick the one giving the lowest P95 (most speed). Fallback: highest p (safest).
ok = devsweep[devsweep["p"] >= ALPHA]
if len(ok):
    chosen = ok.sort_values("p95").iloc[0]
    rule_note = f"most aggressive threshold with no significant dev drop (p >= {ALPHA})"
else:
    chosen = devsweep.sort_values("p", ascending=False).iloc[0]
    rule_note = "no threshold avoided a significant drop on dev; chose the safest (highest p)"
TAU_STAR = float(chosen["tau"])
tau_pctile = round(float((dev["delta"] < TAU_STAR).mean() * 100), 1)

print("="*66)
print("STEP 1  -  THRESHOLD SELECTION ON THE DEVELOPMENT SET  (RQ3)")
print("="*66)
print(f"Baseline on dev:  P95={dev_base['p95']}s  faithfulness={dev_base['faith']}")
print(f"Selection rule:   {rule_note}")
print(f"Chosen threshold: tau = {TAU_STAR:.4f}   (the {tau_pctile}th percentile of dev DeltaScore)")
print(f"  on dev:  P95 {dev_base['p95']} -> {chosen['p95']} s,  faithfulness "
      f"{dev_base['faith']} -> {chosen['faith']},  fast {int(chosen['fast_n'])}/{len(dev)},  p={chosen['p']}")
print()

# ---------------- evaluate ONCE on HELD-OUT ----------------
res = evaluate(hold, TAU_STAR)
dP95 = round(100 * (res["p95"] - hold_base["p95"]) / hold_base["p95"], 1)
dMean = round(100 * (res["mean"] - hold_base["mean"]) / hold_base["mean"], 1)
print("="*66)
print("STEP 2  -  FINAL EVALUATION ON THE HELD-OUT SET  (RQ2)")
print("="*66)
print(f"Threshold frozen at tau = {TAU_STAR:.4f}. The held-out set was not used to choose it.")
print(f"  Static baseline:   P95 {hold_base['p95']} s   mean {hold_base['mean']} s   faithfulness {hold_base['faith']}")
print(f"  Adaptive (K={K_FAST}):    P95 {res['p95']} s   mean {res['mean']} s   faithfulness {res['faith']}")
print(f"  Change:            P95 {dP95:+.1f}%   mean {dMean:+.1f}%   faithfulness {res['faith']-hold_base['faith']:+.3f}")
print(f"  Fast/Deep split:   {res['fast_n']} fast / {len(hold)-res['fast_n']} deep")
print(f"  McNemar on faithfulness:  lost {res['lost']}, gained {res['gained']}, p = {res['p']}  "
      f"({'no significant drop' if res['p']>=ALPHA else 'significant drop'})")
verdict_rq2 = ("MET: meaningful P95 reduction with no statistically significant faithfulness drop"
               if (dP95 < 0 and res["p"] >= ALPHA) else
               "NOT fully met: check the P95 change and the significance above")
print(f"  RQ2 verdict:  {verdict_rq2}")
print()

# ---------------- RQ1: routing confusion matrix on HELD-OUT ----------------
# deep-needed  = baseline (Deep) supported AND Fast supported-not  -> the query needed depth
# shallow-ok   = Fast supported                                    -> shallow was enough
# both-fail    = neither supported (excluded from precision/recall, reported separately)
a=b_=c_=d_=bothfail=0
for _, r in hold.iterrows():
    routed_fast = r["delta"] >= TAU_STAR
    deep_needed = (int(r["base_v"])==1 and int(r["fast_v"])==0)
    shallow_ok  = (int(r["fast_v"])==1)
    if int(r["base_v"])==0 and int(r["fast_v"])==0:
        bothfail += 1; continue
    if deep_needed and not routed_fast: b_ += 1      # protected (good)
    elif deep_needed and routed_fast:   a += 1       # MISS (harmful error)
    elif shallow_ok and routed_fast:    c_ += 1      # fast and fine (good)
    elif shallow_ok and not routed_fast:d_ += 1      # over-cautious (safe, slower)
deep_total = a + b_
protect_recall = round(100*b_/deep_total,1) if deep_total else float("nan")
fast_routed = a + c_
deep_routed = b_ + d_
rate_fast = round(100*a/fast_routed,1) if fast_routed else float("nan")
rate_deep = round(100*b_/deep_routed,1) if deep_routed else float("nan")

print("="*66)
print("STEP 3  -  ROUTING RELIABILITY ON THE HELD-OUT SET  (RQ1)")
print("="*66)
print("Confusion matrix (rows = what the query actually needed, cols = where it was routed):")
print(f"{'':22}{'routed Fast':>14}{'routed Deep':>14}")
print(f"{'deep-needed':22}{a:>14}{b_:>14}")
print(f"{'shallow-sufficient':22}{c_:>14}{d_:>14}")
print(f"(excluded: {bothfail} queries where neither path was judged supported)")
print()
print(f"Protection recall: {protect_recall}% of deep-needed queries were correctly routed Deep.")
print(f"Deep-needed rate among Fast-routed: {rate_fast}%   among Deep-routed: {rate_deep}%")
if not np.isnan(rate_fast) and not np.isnan(rate_deep):
    verdict_rq1 = ("SUPPORTED: deep-needed queries concentrate in the Deep-routed group"
                   if rate_deep > rate_fast else
                   "WEAK: the signal does not clearly separate deep-needed queries")
    print(f"RQ1 verdict:  {verdict_rq1}")
print()

# ---------------- save ----------------
summary = pd.DataFrame([
    {"set":"dev_baseline","tau":"", "p95":dev_base["p95"],"mean":dev_base["mean"],"faith":dev_base["faith"],"fast_n":0,"p":""},
    {"set":"dev_chosen","tau":round(TAU_STAR,4),"p95":chosen["p95"],"mean":chosen["mean"],"faith":chosen["faith"],"fast_n":int(chosen["fast_n"]),"p":chosen["p"]},
    {"set":"heldout_baseline","tau":"","p95":hold_base["p95"],"mean":hold_base["mean"],"faith":hold_base["faith"],"fast_n":0,"p":""},
    {"set":"heldout_adaptive","tau":round(TAU_STAR,4),"p95":res["p95"],"mean":res["mean"],"faith":res["faith"],"fast_n":res["fast_n"],"p":res["p"]},
])
summary.to_csv("final_results.csv", index=False)
devsweep.to_csv("final_dev_sweep.csv", index=False)
print("Saved final_results.csv and final_dev_sweep.csv.")
print("\nReport TAU_STAR, the held-out numbers, and the confusion matrix in your Results chapter.")
print("Note the caveats: the faithfulness judge is noisy (re-judge at temperature 0 to firm this up),")
print("and results are on a single split (you can repeat with several SEED values for robustness).")
