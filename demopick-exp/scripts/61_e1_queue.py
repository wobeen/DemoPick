"""E1-lite 대기열 실행기: 작업 공간 명령 한 번(최대 ~3분) 안에 끝나는 단위로 나눠 이어서 실행한다.

단위: 학습(run당 1회, 모델을 results/e1_models/<run>.pkl로 저장) → 평가(롤아웃 CHUNK개씩, results/e1_rollouts/<run>__<start>.json)
같은 명령을 반복해서 부르면 끝난 단위는 건너뛰고 남은 것만 한다. 모두 끝나면 'QUEUE_DONE'.

사전 고정(2026-11-03 23:35, 결과 보기 전):
  RUNS: keep 0.5 × {all, heuristic, oracle, random, reverse} × seed {0,1,2}
  EPOCHS 100, N_ROLLOUTS 30, HORIZON 300, EVAL_SEED 10000 (모든 run 같은 초기 상태)
사용: python scripts/61_e1_queue.py --hdf5 <lift_mh> [--budget 150]
"""
import argparse
import glob
import importlib.util
import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EPOCHS, N_ROLLOUTS, HORIZON, EVAL_SEED, CHUNK = 100, 30, 300, 10000, 5  # 기본값 = Lift 사전 고정값
ACT_CHUNK, ACT_EXEC = 1, 1  # 행동 묶음(기본 1 = Lift와 같음)
METHODS = ["all", "heuristic", "oracle", "random", "reverse"]
SEEDS = [0, 1, 2]
KEEPS = [0.5]
MODEL_DIR, ROLL_DIR = "results/e1_models", "results/e1_rollouts"


def bc():
    spec = importlib.util.spec_from_file_location("bc", os.path.join(HERE, "60_e1_cpu_bc.py"))
    m = importlib.util.module_from_spec(spec)
    sys.path.insert(0, HERE)
    spec.loader.exec_module(m)
    m.FEATURES_META = "results/features_meta.json"
    return m


def runs():
    out = []
    for seed in SEEDS:
        for keep in KEEPS:
            for m in METHODS:
                out.append((f"{m}_k{keep}_s{seed}" if m != "all" else f"all_s{seed}", m, keep, seed))
    seen, uniq = set(), []
    for r in out:
        if r[0] not in seen:
            seen.add(r[0]); uniq.append(r)
    return uniq


def train_unit(hdf5, run, method, keep, seed):
    import warnings
    import h5py
    import pandas as pd
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    m = bc()
    df = pd.read_csv("results/features.csv")
    chosen = m.select(df, method, keep, seed)
    mix = df.set_index("demo_id").loc[chosen, "skill"].value_counts().to_dict()
    t0 = time.time()
    with h5py.File(hdf5, "r") as f:
        X, Y = m.load_xy(f, chosen, ACT_CHUNK)
    sc = StandardScaler().fit(X)
    pol = MLPRegressor(hidden_layer_sizes=(256, 256), batch_size=256, learning_rate_init=1e-3,
                       max_iter=EPOCHS, random_state=seed, shuffle=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pol.fit(sc.transform(X), Y)
    meta = {"run": run, "method": method, "keep": 1.0 if method == "all" else keep, "seed": seed,
            "n_demos": len(chosen), "n_samples": int(len(X)), "skill_mix": mix,
            "final_train_loss": float(pol.loss_), "train_sec": round(time.time() - t0, 1),
            "epochs": EPOCHS, "act_chunk": ACT_CHUNK, "act_exec": ACT_EXEC}
    with open(f"{MODEL_DIR}/{run}.pkl.tmp", "wb") as fp:
        pickle.dump({"policy": pol, "scaler": sc, "meta": meta}, fp)
    os.replace(f"{MODEL_DIR}/{run}.pkl.tmp", f"{MODEL_DIR}/{run}.pkl")
    return f"train {run} {meta['train_sec']}s"


def eval_unit(hdf5, run, start):
    import h5py
    import robosuite as suite
    m = bc()
    with open(f"{MODEL_DIR}/{run}.pkl", "rb") as fp:
        obj = pickle.load(fp)
    with h5py.File(hdf5, "r") as f:
        ea = json.loads(f["data"].attrs["env_args"])
    env = suite.make(ea["env_name"], **ea["env_kwargs"])
    t0 = time.time()
    succ, lengths = m.rollout(env, obj["policy"], obj["scaler"], CHUNK, HORIZON, EVAL_SEED + start,
                              ACT_CHUNK, ACT_EXEC)
    rec = {"run": run, "start": start, "success": [bool(s) for s in succ], "lengths": lengths,
           "sec": round(time.time() - t0, 1)}
    with open(f"{ROLL_DIR}/{run}__{start}.json", "w") as fp:
        json.dump(rec, fp)
    return f"eval {run} [{start}:{start + CHUNK}] {sum(succ)}/{CHUNK} {rec['sec']}s"


def pending(hdf5):
    units = []
    for run, method, keep, seed in runs():
        if not os.path.exists(f"{MODEL_DIR}/{run}.pkl"):
            units.append(("train", (hdf5, run, method, keep, seed), 120 if method == "all" else 50))
            continue
        for start in range(0, N_ROLLOUTS, CHUNK):
            if not os.path.exists(f"{ROLL_DIR}/{run}__{start}.json"):
                units.append(("eval", (hdf5, run, start), 45))
    return units


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--budget", type=float, default=150)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--keeps", type=float, nargs="+", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--n-rollouts", type=int, default=None)
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--act-chunk", type=int, default=None)
    ap.add_argument("--act-exec", type=int, default=None)
    a = ap.parse_args()
    global KEEPS, EPOCHS, N_ROLLOUTS, HORIZON, ACT_CHUNK, ACT_EXEC
    KEEPS = a.keeps or KEEPS
    EPOCHS = a.epochs or EPOCHS
    N_ROLLOUTS = a.n_rollouts or N_ROLLOUTS
    HORIZON = a.horizon or HORIZON
    ACT_CHUNK = a.act_chunk or ACT_CHUNK
    ACT_EXEC = a.act_exec or ACT_EXEC
    cfg = {"keeps": KEEPS, "epochs": EPOCHS, "n_rollouts": N_ROLLOUTS, "horizon": HORIZON,
           "act_chunk": ACT_CHUNK, "act_exec": ACT_EXEC, "eval_seed": EVAL_SEED,
           "methods": METHODS, "seeds": SEEDS, "hdf5": a.hdf5}
    os.makedirs("results", exist_ok=True)
    if os.path.exists("results/e1_queue_config.json"):
        old = json.load(open("results/e1_queue_config.json"))
        if old != cfg:
            raise SystemExit(f"설정이 처음 실행과 다름 → 중단. 처음: {old}")
    else:
        json.dump(cfg, open("results/e1_queue_config.json", "w"), indent=1)
    os.makedirs(MODEL_DIR, exist_ok=True); os.makedirs(ROLL_DIR, exist_ok=True)
    t0 = time.time()
    inflight = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        while True:
            busy = {k for k in inflight}
            todo = [u for u in pending(a.hdf5) if (u[0], u[1][1], u[1][2] if u[0] == "eval" else None) not in busy]
            # 학습이 끝나지 않은 run의 평가는 pending에 안 나오므로 순서는 자동으로 맞는다
            while todo and len(inflight) < a.workers:
                kind, args, cost = todo.pop(0)
                if time.time() - t0 + cost > a.budget:
                    todo = []
                    break
                fn = train_unit if kind == "train" else eval_unit
                key = (kind, args[1], args[2] if kind == "eval" else None)
                inflight[key] = ex.submit(fn, *args)
            if not inflight:
                break
            done = [k for k, fu in inflight.items() if fu.done()]
            for k in done:
                try:
                    print(inflight.pop(k).result(), flush=True)
                except Exception as e:  # 실패도 기록하고 계속
                    inflight.pop(k, None)
                    print("FAIL", k, repr(e), flush=True)
            time.sleep(1)
    left = pending(a.hdf5)
    print(f"elapsed {time.time() - t0:.0f}s, remaining units {len(left)}")
    if not left:
        print("QUEUE_DONE")


if __name__ == "__main__":
    main()
