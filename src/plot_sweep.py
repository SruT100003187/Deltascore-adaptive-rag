"""
plot_sweep.py  -  draws the two graphs from sweep_results.csv.

Produces:
  graph_K2.png            Fast Path K=2: faithfulness and P95 latency vs threshold
  graph_K3.png            Fast Path K=3: faithfulness and P95 latency vs threshold
  graph_faithfulness_K2_vs_K3.png   the direct comparison that answers "does K=3 recover faithfulness"

Run AFTER sweep_two_graphs.py. Instant, no model needed. Re-run any time.
If you did a smoke test, point FILE at sweep_results_smoke8.csv instead.
"""

import pandas as pd
import matplotlib.pyplot as plt

import glob, os
# Prefer the full run; fall back to a smoke-test file if that is all there is.
if os.path.exists("sweep_results.csv"):
    FILE, REF = "sweep_results.csv", "baseline_reference.csv"
else:
    cands = sorted(glob.glob("sweep_results*.csv"))
    if not cands:
        raise SystemExit("No sweep_results*.csv found. Run sweep_two_graphs.py first.")
    FILE = cands[0]
    REF = FILE.replace("sweep_results", "baseline_reference")
    print(f"Using {FILE}")
ORDER = ["mean+2sd", "mean+1sd", "mean", "mean-1sd", "mean-2sd"]

TEAL, ORANGE, GREY = "#0E7C86", "#E8912D", "#8FAFC4"

res = pd.read_csv(FILE)
ref = pd.read_csv(REF).iloc[0]
res["threshold"] = pd.Categorical(res["threshold"], ORDER, ordered=True)
res = res.sort_values(["K", "threshold"])


def one_graph(K):
    sub = res[res["K"] == K]
    fig, ax1 = plt.subplots(figsize=(8.2, 5.2))
    # faithfulness (left axis)
    ax1.plot(sub["threshold"], sub["faithfulness"], "o-", color=TEAL, lw=2.2,
             label="adaptive faithfulness")
    ax1.axhline(ref["faithfulness"], ls="--", color=TEAL, alpha=0.55,
                label=f"baseline faithfulness ({ref['faithfulness']})")
    ax1.set_ylabel("faithfulness", color=TEAL, fontsize=12)
    ax1.tick_params(axis="y", labelcolor=TEAL)
    ax1.set_xlabel("threshold  (tau)", fontsize=12)
    # latency (right axis)
    ax2 = ax1.twinx()
    ax2.plot(sub["threshold"], sub["P95_s"], "s-", color=ORANGE, lw=2.2,
             label="adaptive P95 latency")
    ax2.axhline(ref["P95_s"], ls="--", color=ORANGE, alpha=0.55,
                label=f"baseline P95 ({ref['P95_s']}s)")
    ax2.set_ylabel("P95 latency (s)", color=ORANGE, fontsize=12)
    ax2.tick_params(axis="y", labelcolor=ORANGE)

    # annotate the Fast/Deep split at each point
    for _, r in sub.iterrows():
        ax1.annotate(f"{r['fast_n']}F/{r['deep_n']}D",
                     (r["threshold"], r["faithfulness"]),
                     textcoords="offset points", xytext=(0, 9),
                     ha="center", fontsize=8, color=GREY)

    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="lower center",
               fontsize=9, ncol=2, framealpha=0.9)
    plt.title(f"Adaptive router, Fast Path K={K}\n(apple-to-apple vs static Top-5 baseline)",
              fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"graph_K{K}.png", dpi=200)
    plt.close()
    print(f"saved graph_K{K}.png")


def compare_graph():
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for K, col, mk in [(2, "#B25A0B", "o"), (3, TEAL, "s")]:
        sub = res[res["K"] == K]
        ax.plot(sub["threshold"], sub["faithfulness"], mk + "-", color=col, lw=2.2,
                label=f"adaptive faithfulness, K={K}")
    ax.axhline(ref["faithfulness"], ls="--", color=GREY, lw=1.8,
               label=f"baseline faithfulness ({ref['faithfulness']})")
    ax.set_ylabel("faithfulness", fontsize=12)
    ax.set_xlabel("threshold  (tau)", fontsize=12)
    ax.legend(fontsize=10, loc="best")
    plt.title("Does a deeper Fast Path (K=3) recover faithfulness?\nK=2 vs K=3 vs baseline",
              fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("graph_faithfulness_K2_vs_K3.png", dpi=200)
    plt.close()
    print("saved graph_faithfulness_K2_vs_K3.png")


for K in sorted(res["K"].unique()):
    one_graph(int(K))
compare_graph()
print("\nDone. Open graph_K2.png, graph_K3.png and graph_faithfulness_K2_vs_K3.png.")
