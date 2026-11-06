"""E1-lite 집계: run별 성공률, 방법별 평균(시드 3개)과 합산 95% 구간(Wilson), 같은 초기 상태끼리 짝지은 비교.
출력: results/e1_cpu_summary.csv, results/e1_cpu_summary.json, results/figures/e1_cpu_success.png
"""
import glob
import json
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import binomtest  # noqa: E402


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


rows, per = [], {}
for pk in sorted(glob.glob("results/e1_models/*.pkl")):
    meta = pickle.load(open(pk, "rb"))["meta"]
    run = meta["run"]
    chunks = sorted(glob.glob(f"results/e1_rollouts/{run}__*.json"), key=lambda p: int(p.split("__")[1][:-5]))
    s = []
    for c in chunks:
        s += json.load(open(c))["success"]
    per[run] = np.array(s, dtype=bool)
    rows.append({**{k: meta[k] for k in ("run", "method", "keep", "seed", "n_demos")},
                 "worse": meta["skill_mix"].get("worse", 0), "okay": meta["skill_mix"].get("okay", 0),
                 "better": meta["skill_mix"].get("better", 0), "n_rollouts": len(s),
                 "success_rate": float(np.mean(s)), "final_train_loss": meta["final_train_loss"]})
df = pd.DataFrame(rows)
df.to_csv("results/e1_cpu_runs.csv", index=False)

order = ["all", "oracle", "heuristic", "random", "reverse"]
summ = []
for m in order:
    d = df[df.method == m]
    if d.empty:
        continue
    k = int(sum(per[r].sum() for r in d.run)); n = int(sum(len(per[r]) for r in d.run))
    lo, hi = wilson(k, n)
    summ.append({"method": m, "keep": float(d.keep.iloc[0]), "n_seeds": len(d),
                 "mean_success": d.success_rate.mean(), "sd_across_seeds": d.success_rate.std(ddof=1),
                 "pooled_k": k, "pooled_n": n, "wilson95_lo": lo, "wilson95_hi": hi,
                 "mean_worse_in_train": d.worse.mean(), "mean_better_in_train": d.better.mean()})
S = pd.DataFrame(summ)
S.round(3).to_csv("results/e1_cpu_summary.csv", index=False)

# 짝지은 비교: 같은 시드·같은 초기 상태(롤아웃 i)끼리. 한쪽만 성공한 경우로 부호 검정(정확 이항)
def paired(a, b):
    wa = wb = 0
    for s in (0, 1, 2):
        ra = next((r for r in per if r.startswith(a) and r.endswith(f"_s{s}")), None)
        rb = next((r for r in per if r.startswith(b) and r.endswith(f"_s{s}")), None)
        if ra is None or rb is None:
            continue
        wa += int(np.sum(per[ra] & ~per[rb])); wb += int(np.sum(per[rb] & ~per[ra]))
    p = binomtest(wa, wa + wb, 0.5).pvalue if wa + wb else 1.0
    return {"a": a, "b": b, "a_only": wa, "b_only": wb, "sign_test_p": p}

pairs = [paired(a, b) for a, b in [("heuristic", "random"), ("oracle", "random"), ("heuristic", "oracle"),
                                    ("heuristic", "all"), ("random", "reverse"), ("heuristic", "reverse"),
                                    ("all", "reverse")]]
out = {"setup": "MLP BC 256-256, epochs 100, 30 rollouts x 3 seeds, horizon 300, same init states",
       "summary": S.round(4).to_dict(orient="records"), "paired": pairs}
json.dump(out, open("results/e1_cpu_summary.json", "w"), indent=2, ensure_ascii=False)

fig, ax = plt.subplots(figsize=(6.5, 4))
x = np.arange(len(S))
ax.bar(x, S.mean_success, color=["#888888", "#55A868", "#4C72B0", "#C9C9C9", "#C44E52"][:len(S)])
ax.errorbar(x, S.mean_success, yerr=[S.mean_success - S.wilson95_lo, S.wilson95_hi - S.mean_success],
            fmt="none", ecolor="black", capsize=4, lw=1)
for i, r in S.reset_index().iterrows():
    for _, rr in df[df.method == r.method].iterrows():
        ax.plot(i + np.random.uniform(-.12, .12), rr.success_rate, "k.", ms=4)
ax.set_xticks(x); ax.set_xticklabels([f"{m}\n({int(df[df.method == m].n_demos.iloc[0])} demos)" for m in S.method])
ax.set_ylim(0, 1); ax.set_ylabel("Success rate (sim rollouts)")
TASK = os.environ.get("E1_TASK", "Lift MH")
ax.set_title(f"E1-lite: which demos to keep ({TASK}, MLP BC)")
fig.text(0.01, 0.01, "bars: mean of 3 seeds; error: Wilson 95% on 90 pooled rollouts; dots: seeds",
         fontsize=7, color="gray")
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig("results/figures/e1_cpu_success.png", dpi=160)

print(df.sort_values(["method", "seed"])[["run", "worse", "okay", "better", "success_rate"]].to_string(index=False))
print(S.round(3).to_string(index=False))
for p in pairs:
    print(p)
