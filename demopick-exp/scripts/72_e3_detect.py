"""E3 탐지: 결함 시연 60개 vs 정상 240개를 휴리스틱이 가려내는가.
사용: python scripts/72_e3_detect.py runs/e3_lift runs/e3_square
출력: <각 run 디렉터리>/e3_detect.csv, 화면 표
AUROC는 '나쁨 점수가 클수록 결함'으로 계산(0.5=우연, <0.5=결함을 오히려 좋다고 봄).
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from common import badness  # noqa: E402

rows = []
for root in sys.argv[1:]:
    ids = set(json.load(open(f"{root}/defect_ids.json"))["defect_ids"])
    for k in ["freeze", "lag", "pause", "truncate"]:
        df = pd.read_csv(f"{root}/{k}/results/features.csv")
        meta = json.load(open(f"{root}/{k}/results/features_meta.json"))
        y = df.demo_id.isin(ids).astype(int).values
        b = badness(df, meta["features_A"]).values
        keep = np.argsort(b, kind="mergesort")[: len(df) // 2]  # 휴리스틱으로 절반 남기기
        r = {"task": os.path.basename(root).replace("e3_", ""), "defect": k,
             "composite_A_auroc": roc_auc_score(y, b),
             "defects_left_after_keep50": int(y[keep].sum()), "defects_total": int(y.sum())}
        for c in meta["features_A"]:
            r[c] = roc_auc_score(y, df[c].values) if df[c].nunique() > 1 else 0.5
        rows.append(r)
out = pd.DataFrame(rows)
for root in sys.argv[1:]:
    t = os.path.basename(root).replace("e3_", "")
    out[out.task == t].round(3).to_csv(f"{root}/e3_detect.csv", index=False)
pd.set_option("display.width", 200)
print(out.round(3).to_string(index=False))
