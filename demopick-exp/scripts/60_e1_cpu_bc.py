"""E1-lite (CPU): 시연 선별 방법별로 BC 정책을 학습하고 robosuite 시뮬레이터에서 성공률을 잰다.

GPU·CUPID 없이 '선별이 정책 성능을 바꾸는가'를 직접 확인하는 축소판이다.
  정책: scikit-learn MLP(256-256) 행동 복제(관측 19차원 → 행동 7차원, MSE)
  관측: robot0_eef_pos(3) + robot0_eef_quat(4) + robot0_gripper_qpos(2) + object(10)
  선별 방법(남기는 비율 --keep):
    all        전체 300개
    random     무작위
    heuristic  E0 composite_A(휴리스틱 7개 나쁨 점수)가 낮은 순 — 숙련도 라벨 사용 안 함
    oracle     숙련도 라벨(better > okay > worse, 같은 등급 안에서는 무작위)
    reverse    heuristic의 반대(나쁜 시연만 남김) — 선별이 효과가 있는지 확인용 대조군
  평가: 같은 시드 목록으로 초기 상태를 고정해 모든 방법이 같은 큐브 위치에서 시험받는다.
  한계: CUPID의 확산 정책이 아니라 단순 MLP BC. 결과는 '이 정책 기준'으로만 쓴다.

사용: python scripts/60_e1_cpu_bc.py --hdf5 <lift_mh low_dim> --method heuristic --keep 0.5 --seed 0
결과는 results/e1_cpu.jsonl 에 한 줄씩 추가된다.
"""
import argparse
import json
import os
import sys
import time

import h5py
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from common import badness, list_demos  # noqa: E402

OBS_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]
ENV_OBS_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object-state"]
SKILL_RANK = {"better": 0, "okay": 1, "worse": 2}


def select(df, method, keep, seed):
    n = len(df) if method == "all" else int(round(len(df) * keep))
    rng = np.random.default_rng(seed)
    if method == "all":
        return df["demo_id"].tolist()
    if method == "random":
        return df["demo_id"].values[rng.permutation(len(df))[:n]].tolist()
    if method in ("heuristic", "reverse"):
        b = badness(df, json.load(open(FEATURES_META, encoding="utf-8"))["features_A"])
        order = np.argsort(b.values, kind="mergesort")
        if method == "reverse":
            order = order[::-1]
        return df["demo_id"].values[order[:n]].tolist()
    if method == "oracle":
        tie = rng.permutation(len(df))
        key = df["skill"].map(SKILL_RANK).values * 10000 + tie
        return df["demo_id"].values[np.argsort(key)[:n]].tolist()
    raise ValueError(method)


def load_xy(f, demos, chunk=1):
    """chunk>1이면 다음 chunk개 행동을 이어 붙여 정답으로 쓴다(끝은 마지막 행동으로 채움)."""
    X, Y = [], []
    for d in demos:
        g = f[f"data/{d}"]
        X.append(np.concatenate([g["obs"][k][()] for k in OBS_KEYS], axis=1))
        a = g["actions"][()]
        if chunk > 1:
            pad = np.concatenate([a, np.repeat(a[-1:], chunk - 1, axis=0)])
            a = np.concatenate([pad[i:i + len(a)] for i in range(chunk)], axis=1)
        Y.append(a)
    return np.concatenate(X), np.concatenate(Y)


def rollout(env, policy, scaler, n, horizon, base_seed, chunk=1, execute=1):
    succ = []
    lengths = []
    for i in range(n):
        np.random.seed(base_seed + i)
        o = env.reset()
        ok = False
        plan = []
        for t in range(horizon):
            if not plan:
                x = np.concatenate([o[k] for k in ENV_OBS_KEYS])[None]
                y = np.clip(policy.predict(scaler.transform(x))[0], -1, 1).reshape(chunk, -1)
                plan = list(y[:execute])
            a = plan.pop(0)
            o, _, _, _ = env.step(a)
            if env._check_success():
                ok = True
                break
        succ.append(ok)
        lengths.append(t + 1)
    return succ, lengths


def main():
    global FEATURES_META
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--features", default="results/features.csv")
    ap.add_argument("--method", required=True, choices=["all", "random", "heuristic", "oracle", "reverse"])
    ap.add_argument("--keep", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--n-rollouts", type=int, default=30)
    ap.add_argument("--horizon", type=int, default=400)
    ap.add_argument("--eval-seed", type=int, default=10000)
    ap.add_argument("--out", default="results/e1_cpu.jsonl")
    ap.add_argument("--chunk", type=int, default=1, help="한 번에 예측할 행동 수 (1=기존)")
    ap.add_argument("--execute", type=int, default=1, help="예측한 행동 중 실행할 수")
    a = ap.parse_args()
    FEATURES_META = os.path.splitext(a.features)[0] + "_meta.json"

    df = pd.read_csv(a.features)
    chosen = select(df, a.method, a.keep, a.seed)
    mix = df.set_index("demo_id").loc[chosen, "skill"].value_counts().to_dict()

    t0 = time.time()
    with h5py.File(a.hdf5, "r") as f:
        X, Y = load_xy(f, chosen, a.chunk)
        env_args = json.loads(f["data"].attrs["env_args"])
    scaler = StandardScaler().fit(X)
    pol = MLPRegressor(hidden_layer_sizes=(256, 256), batch_size=256, learning_rate_init=1e-3,
                       max_iter=a.epochs, random_state=a.seed, shuffle=True)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pol.fit(scaler.transform(X), Y)
    t_train = time.time() - t0

    import robosuite as suite
    env = suite.make(env_args["env_name"], **env_args["env_kwargs"])
    t1 = time.time()
    succ, lengths = rollout(env, pol, scaler, a.n_rollouts, a.horizon, a.eval_seed, a.chunk, a.execute)
    t_eval = time.time() - t1

    rec = {"method": a.method, "keep": 1.0 if a.method == "all" else a.keep, "seed": a.seed,
           "n_demos": len(chosen), "n_samples": int(len(X)), "skill_mix": mix,
           "epochs": a.epochs, "chunk": a.chunk, "execute": a.execute, "final_train_loss": float(pol.loss_),
           "n_rollouts": a.n_rollouts, "horizon": a.horizon, "eval_seed": a.eval_seed,
           "success_rate": float(np.mean(succ)), "n_success": int(np.sum(succ)),
           "mean_len_success": float(np.mean([l for l, s in zip(lengths, succ) if s])) if any(succ) else None,
           "train_sec": round(t_train, 1), "eval_sec": round(t_eval, 1),
           "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(json.dumps(rec, ensure_ascii=False))


if __name__ == "__main__":
    main()
