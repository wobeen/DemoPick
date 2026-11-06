"""휴리스틱 점수를 CUPID 재학습 설정(train_config.yaml)에 방법 하나로 추가한다.

사용 (서버, CUPID 노트북 Sec 3.1로 train_config.yaml을 만든 뒤):
  python scripts/20_heuristic_to_curation.py \
      --features results/features.csv \
      --config <cupid>/configs/curation/low_dim/lift_mh/train_config.yaml

CUPID 형식 [cupid 저장소에서 확인]
  - train_config.yaml = {방법 이름: {시드: [에피소드 번호, ...]}}
  - 목록의 앞쪽부터 제거한다(나쁜 시연이 앞). filter_ratio = 학습 데이터에서 **제거하는** 비율.
  - 에피소드 번호 = hdf5의 demo_N에서 N.
  - 학습 데이터는 전체 300개 중 일부(필터 실험 기본 train_ratio=0.64)이고 시드마다 다르다.
    그래서 같은 파일의 'random' 목록(= 그 시드의 학습 에피소드 전체를 섞은 것)에서
    학습 에피소드 집합을 가져온다.
  - retrain_policies.sh는 방법 이름이 oracle/random/influence*/*similarity/dem*/*diversity/*loss
    중 하나여야 통과시킨다. 그래서 기본 이름을 'demopick_heuristic'(dem*)으로 둔다.

나쁨 점수는 시드별 학습 에피소드 안에서만 z-정규화한다(학습에 없는 시연 정보를 쓰지 않음).
원본 yaml은 .bak으로 남긴다.
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(__file__))
from common import badness, demo_index  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="results/features.csv")
    ap.add_argument("--config", required=True, help="CUPID configs/curation/<state>/<task>/train_config.yaml")
    ap.add_argument("--feature-set", choices=["A", "AB"], default="A",
                    help="A=휴리스틱 7개(E1 비교 대상), AB=관측값 포함 13개")
    ap.add_argument("--method-name", default=None)
    a = ap.parse_args()

    df = pd.read_csv(a.features)
    with open(os.path.splitext(a.features)[0] + "_meta.json", encoding="utf-8") as fp:
        meta = json.load(fp)
    cols = meta["features_A"] + (meta["features_B"] if a.feature_set == "AB" else [])
    name = a.method_name or ("demopick_heuristic" if a.feature_set == "A" else "demopick_heuristic_ab")
    if not name.startswith("dem"):
        print("[주의] 이름이 dem*으로 시작하지 않으면 retrain_policies.sh가 거부할 수 있음")
    df["ep"] = df["demo_id"].map(demo_index)
    df = df.set_index("ep")

    with open(a.config, encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp) or {}
    if "random" not in cfg:
        raise SystemExit("config에 'random' 목록이 없음 → 학습 에피소드 집합을 알 수 없음. "
                         "CUPID 노트북 Sec 3.1을 먼저 실행할 것")

    out = {}
    for seed, train_idxs in cfg["random"].items():
        train_idxs = [int(i) for i in train_idxs]
        missing = set(train_idxs) - set(df.index)
        if missing:
            raise SystemExit(f"시드 {seed}: features.csv에 없는 에피소드 {sorted(missing)[:5]}…")
        sub = df.loc[train_idxs]
        b = badness(sub, cols)
        order = b.sort_values(ascending=False, kind="mergesort").index.tolist()  # 나쁜 것 먼저
        out[seed] = [int(i) for i in order]
        worse_share = (sub.loc[order[: len(order) // 4], "skill"] == "worse").mean()
        print(f"시드 {seed}: 학습 {len(order)}개, 제거 1순위 25% 중 worse 비율 {worse_share:.2f}")

    shutil.copyfile(a.config, a.config + ".bak")
    cfg[name] = out
    with open(a.config, "w", encoding="utf-8") as fp:
        yaml.safe_dump(cfg, fp)
    print(f"'{name}' 추가 → {a.config} (원본: .bak)")
    print("retrain_policies.sh의 curation_methods에 이 이름을 넣고, "
          "curation_filter_ratios와 curation_select_ratios 배열 길이를 같게 맞출 것")


if __name__ == "__main__":
    main()
