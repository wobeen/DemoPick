"""E3: 시연 데이터에 결함을 주입한 hdf5를 만든다. 설계는 notes/e3_defect_prereg.md (사전 고정).

사용: python scripts/70_inject_defects.py --hdf5 <원본> --outdir runs/e3_<task>
출력: <outdir>/<결함>/data.hdf5, <outdir>/defect_ids.json
"""
import argparse
import json
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import list_demos, skill_and_operator_labels  # noqa: E402

DEFECTS = ["freeze", "lag", "pause", "truncate"]
OBS_KEEP = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]
N_PER_SKILL = 20
SEED = 7
LAG = 5


def inject(kind, obs, act, rng):
    T = len(act)
    obs = {k: v.copy() for k, v in obs.items()}
    act = act.copy()
    if kind == "freeze":
        L = max(1, round(0.3 * T)); s = int(rng.integers(int(0.2 * T), max(int(0.2 * T) + 1, int(0.5 * T))))
        e = min(T, s + L)
        for k in obs:
            obs[k][s:e] = obs[k][s - 1] if s > 0 else obs[k][0]
    elif kind == "lag":
        idx = np.minimum(np.arange(T) + LAG, T - 1)
        act = act[idx]
    elif kind == "pause":
        P = max(1, round(0.5 * T)); s = int(rng.integers(int(0.2 * T), max(int(0.2 * T) + 1, int(0.6 * T))))
        ins_act = np.zeros((P, act.shape[1]))
        ins_act[:, :3] = np.clip(rng.normal(0, 0.02, (P, 3)), -1, 1)
        ins_act[:, -1] = act[s, -1]
        act = np.concatenate([act[:s], ins_act, act[s:]])
        for k in obs:
            obs[k] = np.concatenate([obs[k][:s], np.repeat(obs[k][s:s + 1], P, axis=0), obs[k][s:]])
    elif kind == "truncate":
        closed = act[:, -1] > 0
        cut = int(np.argmax(closed)) + 3 if closed.any() else T // 2
        cut = max(5, min(cut, T))
        act = act[:cut]
        obs = {k: v[:cut] for k, v in obs.items()}
    return obs, act


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    with h5py.File(a.hdf5, "r") as f:
        demos = list_demos(f)
        skill, _, _ = skill_and_operator_labels(f)
        rng = np.random.default_rng(SEED)
        chosen = []
        for s in ("worse", "okay", "better"):
            pool = sorted([d for d in demos if skill.get(d) == s], key=lambda x: int(x.split("_")[1]))
            chosen += [pool[i] for i in sorted(rng.choice(len(pool), N_PER_SKILL, replace=False))]
        json.dump({"defect_ids": chosen, "n": len(chosen), "seed": SEED}, open(f"{a.outdir}/defect_ids.json", "w"),
                  indent=1)
        cs = set(chosen)
        for kind in DEFECTS:
            os.makedirs(f"{a.outdir}/{kind}/results", exist_ok=True)
            krng = np.random.default_rng(SEED + DEFECTS.index(kind) + 1)
            with h5py.File(f"{a.outdir}/{kind}/data.hdf5", "w") as g:
                gd = g.create_group("data")
                for k, v in f["data"].attrs.items():
                    gd.attrs[k] = v
                for d in demos:
                    src = f["data"][d]
                    obs = {k: src["obs"][k][()] for k in OBS_KEEP}
                    act = src["actions"][()]
                    if d in cs:
                        obs, act = inject(kind, obs, act, krng)
                    dd = gd.create_group(d)
                    dd.create_dataset("actions", data=act)
                    og = dd.create_group("obs")
                    for k, v in obs.items():
                        og.create_dataset(k, data=v)
                f.copy(f["mask"], g, "mask")
            print(f"{kind}: {len(cs)}개 주입 → {a.outdir}/{kind}/data.hdf5")


if __name__ == "__main__":
    main()
