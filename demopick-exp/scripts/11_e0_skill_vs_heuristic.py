"""E0: 값싼 지표가 숙련도 낮은 작업자의 시연을 얼마나 골라내나.

사용: python scripts/11_e0_skill_vs_heuristic.py
입력: results/features.csv, results/features_meta.json
출력: results/e0_feature_auroc.csv, results/e0_summary.json, results/figures/e0_feature_auroc.png

지표
  - 특징별 AUROC(worse vs better): '클수록 나쁨' 방향 그대로 계산 (방향은 사전 고정).
    0.5보다 작으면 그 특징은 예상과 반대로 움직였다는 뜻이다(뒤집지 않고 그대로 보고).
  - 합성 점수: 특징을 z-점수로 바꿔 평균 = '나쁨 점수'.
    A(휴리스틱 7), A+B(관측값 포함 13) 두 가지.
    · worse vs better AUROC + 부트스트랩 95% 구간(1000회)
    · 숙련도 서열(worse=0, okay=1, better=2)과의 스피어만 상관(부호를 바꿔 '좋음 점수'로)
  - 작업자별 평균 나쁨 점수(작업자 ID가 있을 때)

한계: 숙련도 라벨은 정책 성능 기여도의 대리 지표다. 이 결과만으로 '학습에 해롭다'고 쓰지 않는다.
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from common import badness  # noqa: E402

SKILL_ORDINAL = {"worse": 0, "okay": 1, "better": 2}
N_BOOT = 1000
SEED = 0


def auroc_worse_vs_better(df, score):
    m = df["skill"].isin(["worse", "better"])
    y = (df.loc[m, "skill"] == "worse").astype(int)  # 1 = worse
    return roc_auc_score(y, score[m])


def boot_ci(df, score, rng):
    m = df["skill"].isin(["worse", "better"]).values
    y = (df["skill"].values[m] == "worse").astype(int)
    s = np.asarray(score)[m]
    vals = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        vals.append(roc_auc_score(y[idx], s[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="results/features.csv")
    ap.add_argument("--outdir", default="results")
    a = ap.parse_args()

    df = pd.read_csv(a.features)
    with open(os.path.splitext(a.features)[0] + "_meta.json", encoding="utf-8") as fp:
        meta = json.load(fp)
    fa, fb = meta["features_A"], meta["features_B"]
    df = df[df["skill"].notna()].reset_index(drop=True)
    if not set(["worse", "better"]) <= set(df["skill"]):
        raise SystemExit("worse·better 라벨이 없어 E0를 계산할 수 없음 → 01_inspect 결과 확인")

    rows = []
    for c in fa + fb:
        rows.append({"feature": c, "group": "A_heuristic" if c in fa else "B_object",
                     "auroc_worse_vs_better": auroc_worse_vs_better(df, df[c]),
                     "median_worse": df.loc[df.skill == "worse", c].median(),
                     "median_okay": df.loc[df.skill == "okay", c].median(),
                     "median_better": df.loc[df.skill == "better", c].median()})
    tab = pd.DataFrame(rows)

    rng = np.random.default_rng(SEED)
    ordinal = df["skill"].map(SKILL_ORDINAL)
    summary = {"n": int(len(df)), "n_by_skill": df["skill"].value_counts().to_dict()}
    for name, cols in (("composite_A", fa), ("composite_AB", fa + fb)):
        b = badness(df, cols)
        df[name] = b
        rho, p = spearmanr(-b, ordinal)
        lo, hi = boot_ci(df, b, rng)
        summary[name] = {"features": cols,
                         "auroc_worse_vs_better": float(auroc_worse_vs_better(df, b)),
                         "auroc_95ci": [lo, hi],
                         "spearman_goodness_vs_skill": float(rho), "spearman_p": float(p)}
        tab = pd.concat([tab, pd.DataFrame([{
            "feature": name, "group": "composite",
            "auroc_worse_vs_better": summary[name]["auroc_worse_vs_better"]}])],
            ignore_index=True)

    if df["operator"].notna().any():
        summary["badness_A_by_operator"] = (
            df.groupby("operator")["composite_A"].mean().round(3).to_dict())

    os.makedirs(os.path.join(a.outdir, "figures"), exist_ok=True)
    tab.round(4).to_csv(os.path.join(a.outdir, "e0_feature_auroc.csv"), index=False, encoding="utf-8")
    with open(os.path.join(a.outdir, "e0_summary.json"), "w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)

    # 그림: 특징별 AUROC 막대 (A·B 색 구분, 0.5 기준선)
    t = tab[tab.group != "composite"].copy()
    colors = ["#4C72B0" if g == "A_heuristic" else "#DD8452" for g in t.group]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(t.feature, t.auroc_worse_vs_better, color=colors)
    ax.axhline(0.5, color="gray", lw=1, ls="--")
    for name, ls in (("composite_A", "-"), ("composite_AB", ":")):
        ax.axhline(summary[name]["auroc_worse_vs_better"], color="black", lw=1, ls=ls,
                   label=f"{name} = {summary[name]['auroc_worse_vs_better']:.2f}")
    ax.set_ylim(0, 1)
    ax.set_ylabel("AUROC (worse vs better)")
    ax.set_title("E0: cheap demo features vs operator skill")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(loc="lower right", fontsize=8)
    fig.text(0.01, 0.01, f"robomimic Lift MH, n={len(df)}; blue=heuristic, orange=object-relative; "
             "0.5 = chance", fontsize=7, color="gray")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(a.outdir, "figures", "e0_feature_auroc.png"), dpi=160)

    print(tab.round(3).to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
