"""시연별 특징 추출 → results/features.csv

사용: python scripts/10_extract_features.py --hdf5 <low_dim_v15.hdf5>

A. 휴리스틱 7개 (값싼 움직임 지표)
  len, jerk, hf_ratio, idle_ratio, grip_switch, path_len, t_grasp
B. 관측값(물체 상대) 6개
  d_at_grasp, n_approach, wasted_grasp, t_lift, drop_flag, post_lift_len

모든 특징은 '클수록 나쁜 시연'이 되도록 정의했다(데이터를 보기 전에 정한 방향).
임계값은 아래 상수. 바꾸면 notes/experiment_log.md에 기록한다.
"""
import argparse
import json
import os
import sys

import h5py
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

sys.path.insert(0, os.path.dirname(__file__))
from common import gripper_closed, list_demos, skill_and_operator_labels  # noqa: E402

# ---- 임계값 (사전 고정) ----
IDLE_QUANTILE = 0.10      # 전체 스텝 이동량의 하위 10% 분위수 이하 = 정체
APPROACH_PROMINENCE = 0.02  # m, 거리 곡선 국소 최솟값의 최소 깊이
WASTED_DIST = 0.05        # m, 이보다 멀리서 닫으면 헛집기
LIFT_Z = 0.84             # m(절대 높이). robosuite Lift._check_success와 같은 기준: 큐브 높이 > 테이블 0.80 + 0.04
                          # (v1: '처음 높이 +0.04'로 잘못 옮겨 148개 시연에서 들어 올림을 못 찾음 → 2026-11-03 수정)
DROP_DZ = 0.005           # m, 들어 올린 뒤 처음 높이 + 이 값 아래로 내려가면(테이블로 다시 떨어짐) '놓침'

# 과제별 설정 (Square는 2026-11-03 23:50, Square 특징값을 보기 전에 정함)
#   obj: object 벡터에서 물체 '월드 위치' 칸. Lift=[0:3], Square=[7:10]([0:3]은 그리퍼 기준 상대 위치) — 01 점검·데이터로 확인
#   lift: 'abs' = 절대 높이 LIFT_Z 초과(Lift 성공 기준), 'rel10' = 10번째 프레임 높이 + SQUARE_LIFT_DZ 초과(너트가 떨어지며 시작)
#   wasted_dist(v2, Square 결과 본 뒤 수정): 너트 위치는 중심, 잡는 곳은 약 6cm 떨어진 손잡이 → 0.05m면 정상 잡기도 헛집기로 셈. 0.10m
#   v2 수정 전 결과: runs/square/results/*_v1_beforefix.*
#   b: 이 과제에서 쓰는 관측값 특징. Square는 놓기(기둥에 끼우기)가 정상 동작이라 drop_flag·post_lift_len 제외
SQUARE_LIFT_DZ = 0.04
TASK_CFG = {
    "Lift": {"obj": (0, 3), "lift": "abs",
             "b": ["d_at_grasp", "n_approach", "wasted_grasp", "t_lift", "drop_flag", "post_lift_len"]},
    "NutAssemblySquare": {"obj": (7, 10), "lift": "rel10", "wasted_dist": 0.10,
                          "b": ["d_at_grasp", "n_approach", "wasted_grasp", "t_lift"]},
}

FEATURES_A = ["len", "jerk", "hf_ratio", "idle_ratio", "grip_switch", "path_len", "t_grasp"]
FEATURES_B = ["d_at_grasp", "n_approach", "wasted_grasp", "t_lift", "drop_flag", "post_lift_len"]


def hf_ratio(x):
    """행동 신호(T, k)의 상위 절반 주파수 에너지 비율, 축 평균."""
    vals = []
    for j in range(x.shape[1]):
        s = x[:, j] - x[:, j].mean()
        p = np.abs(np.fft.rfft(s)) ** 2
        p = p[1:]  # 직류 제외
        if p.sum() <= 0:
            continue
        half = len(p) // 2
        vals.append(p[half:].sum() / p.sum())
    return float(np.mean(vals)) if vals else 0.0


def extract_one(g, idle_thr, cfg):
    act = g["actions"][()]
    obs = g["obs"]
    eef = obs["robot0_eef_pos"][()]
    T = act.shape[0]

    step = np.linalg.norm(np.diff(eef, axis=0), axis=1)
    closed = gripper_closed(act)
    sign = np.where(closed, 1, -1)
    onsets = np.where((~closed[:-1]) & closed[1:])[0] + 1
    if closed[0]:
        onsets = np.concatenate([[0], onsets])
    first_close = int(np.argmax(closed)) if closed.any() else None

    feat = {
        "len": T,
        "jerk": float(np.mean(np.linalg.norm(np.diff(act[:, :3], n=2, axis=0), axis=1))),
        "hf_ratio": hf_ratio(act[:, :3]),
        "idle_ratio": float(np.mean(step <= idle_thr)),
        "grip_switch": int(np.sum(sign[1:] != sign[:-1])),
        "path_len": float(step.sum()),
        "t_grasp": (first_close / T) if first_close is not None else 1.0,
    }

    if "object" in obs:
        obj = obs["object"][()]
        cube = obj[:, cfg["obj"][0]:cfg["obj"][1]]
        dist = np.linalg.norm(eef - cube, axis=1)
        if cfg["lift"] == "abs":
            z0 = cube[0, 2]
            lifted = np.where(cube[:, 2] > LIFT_Z)[0]
        else:
            s0 = min(10, len(cube) - 1)
            z0 = cube[s0, 2]
            lifted = np.where(cube[s0:, 2] > z0 + SQUARE_LIFT_DZ)[0] + s0  # 수정(v2): 10프레임 이후만 검사
        lift_idx = int(lifted[0]) if len(lifted) else None
        minima, _ = find_peaks(-dist, prominence=APPROACH_PROMINENCE)
        allb = {
            "d_at_grasp": float(dist[first_close]) if first_close is not None else float(dist.min()),
            "n_approach": int(len(minima)),
            "wasted_grasp": int(np.sum(dist[onsets] > cfg.get("wasted_dist", WASTED_DIST))) if len(onsets) else 0,
            "t_lift": (lift_idx / T) if lift_idx is not None else 1.0,
            "drop_flag": int(lift_idx is not None and np.any(cube[lift_idx:, 2] < z0 + DROP_DZ)),
            "post_lift_len": (T - lift_idx) if lift_idx is not None else 0,
        }
        feat.update({k: allb[k] for k in cfg["b"]})
    return feat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--out", default="results/features.csv")
    a = ap.parse_args()

    with h5py.File(a.hdf5, "r") as f:
        demos = list_demos(f)
        skill, operator, _ = skill_and_operator_labels(f)
        env_name = json.loads(f["data"].attrs["env_args"])["env_name"]
        if env_name not in TASK_CFG:
            raise SystemExit(f"{env_name} 설정 없음 → TASK_CFG에 추가")
        cfg = TASK_CFG[env_name]

        # 1차: 전체 스텝 이동량 분포로 정체 임계값 결정
        all_steps = np.concatenate([
            np.linalg.norm(np.diff(f["data"][d]["obs"]["robot0_eef_pos"][()], axis=0), axis=1)
            for d in demos])
        idle_thr = float(np.quantile(all_steps, IDLE_QUANTILE))

        rows = []
        for d in demos:
            r = {"demo_id": d, "skill": skill.get(d), "operator": operator.get(d)}
            r.update(extract_one(f["data"][d], idle_thr, cfg))
            rows.append(r)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df.to_csv(a.out, index=False, encoding="utf-8")
    meta = {
        "hdf5": os.path.abspath(a.hdf5), "env_name": env_name, "task_cfg": cfg, "n_demos": len(df), "idle_threshold_m": idle_thr,
        "thresholds": {"IDLE_QUANTILE": IDLE_QUANTILE, "APPROACH_PROMINENCE": APPROACH_PROMINENCE,
                       "WASTED_DIST": WASTED_DIST, "LIFT_Z": LIFT_Z, "DROP_DZ": DROP_DZ,
                       "SQUARE_LIFT_DZ": SQUARE_LIFT_DZ},
        "features_A": FEATURES_A,
        "features_B": [c for c in FEATURES_B if c in df.columns],
    }
    with open(os.path.splitext(a.out)[0] + "_meta.json", "w", encoding="utf-8") as fp:
        json.dump(meta, fp, ensure_ascii=False, indent=2)

    print(f"시연 {len(df)}개 → {a.out}")
    print(df.groupby("skill")[meta["features_A"] + meta["features_B"]].median().round(3).T)
    never = df["t_lift"].eq(1.0).sum() if "t_lift" in df else 0
    if never:
        print(f"\n[주의] 큐브를 들어 올린 순간을 못 찾은 시연 {never}개 → LIFT_Z 또는 object 해석 확인")


if __name__ == "__main__":
    main()
