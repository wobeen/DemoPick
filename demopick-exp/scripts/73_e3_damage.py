"""E3 피해·회복 집계 (Lift). 사용: python scripts/73_e3_damage.py runs/e3_lift
출력: <root>/e3_damage.csv, <root>/e3_damage.json, <root>/e3_damage.png
짝지은 비교는 같은 시드의 30개 초기 상태끼리 — 롤아웃 기록이 run 단위 성공 수만 있으므로,
여기서는 시드별 성공률 차이 + 합산 Wilson 구간 + Fisher 정확 검정(합산)으로 보고한다.
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import fisher_exact  # noqa: E402

root = sys.argv[1]
rows = []
for p in glob.glob(f"{root}/*/results/e1_*_s*.jsonl"):
    cond = p.split("/")[-3]
    for line in open(p):
        r = json.loads(line)
        rows.append({"cond": cond, "method": r["method"], "seed": r["seed"], "k": r["n_success"],
                     "n": r["n_rollouts"], "rate": r["success_rate"], "n_demos": r["n_demos"]})
df = pd.DataFrame(rows).sort_values(["cond", "method", "seed"])


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


S = (df.groupby(["cond", "method"]).agg(seeds=("seed", "count"), k=("k", "sum"), n=("n", "sum"),
                                         mean=("rate", "mean"), per_seed=("rate", lambda x: list(np.round(x, 2))))
     .reset_index())
S["lo"], S["hi"] = zip(*[wilson(k, n) for k, n in zip(S.k, S.n)])
ref = S[(S.cond == "clean") & (S.method == "all")].iloc[0]
tests = []
for _, r in S.iterrows():
    if r.cond == "clean":
        continue
    _, p_vs_clean = fisher_exact([[r.k, r.n - r.k], [ref.k, ref.n - ref.k]])
    t = {"cond": r.cond, "method": r.method, "rate": r.k / r.n, "vs_clean_all_diff": r.k / r.n - ref.k / ref.n,
         "p_vs_clean_all": p_vs_clean}
    if r.method == "heuristic":
        a = S[(S.cond == r.cond) & (S.method == "all")].iloc[0]
        _, p = fisher_exact([[r.k, r.n - r.k], [a.k, a.n - a.k]])
        t.update({"vs_same_defect_all_diff": r.k / r.n - a.k / a.n, "p_vs_same_defect_all": p})
    tests.append(t)
T = pd.DataFrame(tests)
S.round(3).to_csv(f"{root}/e3_damage.csv", index=False)
json.dump({"summary": S.round(4).to_dict(orient="records"), "tests": T.round(4).to_dict(orient="records")},
          open(f"{root}/e3_damage.json", "w"), indent=1, ensure_ascii=False, default=str)

conds = ["clean", "freeze", "lag", "pause", "truncate"]
fig, ax = plt.subplots(figsize=(7.5, 4))
w = 0.38
for j, (m, col, lab) in enumerate([("all", "#C44E52", "all 300 (with defects)"),
                                   ("heuristic", "#4C72B0", "heuristic keeps 150")]):
    for i, c in enumerate(conds):
        r = S[(S.cond == c) & (S.method == m)]
        if r.empty:
            continue
        r = r.iloc[0]; x = i + (j - 0.5) * w
        ax.bar(x, r["mean"], w, color="#888888" if c == "clean" else col, label=lab if i == 1 else None)
        ax.errorbar(x, r["mean"], yerr=[[r["mean"] - r.lo], [r.hi - r["mean"]]], fmt="none", ecolor="k", capsize=3, lw=1)
ax.set_xticks(range(len(conds))); ax.set_xticklabels(conds); ax.set_ylim(0, 1)
ax.set_ylabel("Success rate"); ax.set_title("E3: 60/300 demos with injected defects (Lift MH, MLP BC)")
ax.legend(fontsize=8, loc="upper right")
fig.text(0.01, 0.01, "3 seeds x 30 rollouts, Wilson 95%; clean = no defects (gray)", fontsize=7, color="gray")
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(f"{root}/e3_damage.png", dpi=160)
pd.set_option("display.width", 200)
print(S.drop(columns=["lo", "hi"]).assign(lo=S.lo.round(2), hi=S.hi.round(2)).round(3).to_string(index=False))
print(T.round(3).to_string(index=False))
