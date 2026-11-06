"""스크립트 동작 검증용 가짜 데이터 (실제 결과로 쓰지 말 것).

robomimic Lift MH low_dim과 같은 키 구조로 300개 시연을 만든다.
worse 작업자일수록 느리고 흔들리고 재시도가 많도록 일부러 만들었으므로,
여기서 나온 AUROC는 '스크립트가 돈다'는 뜻일 뿐 아무 의미가 없다.
"""
import sys

import h5py
import numpy as np

rng = np.random.default_rng(1)
SK = {"worse": 2.0, "okay": 1.0, "better": 0.5}


def demo(level):
    T = int(rng.normal(120 + 60 * level, 15))
    cube0 = np.array([rng.uniform(-.03, .03), rng.uniform(-.03, .03), 0.82])
    eef = np.zeros((T, 3)); cube = np.tile(cube0, (T, 1))
    start = np.array([0, 0, 1.0]); grasp_t = int(T * 0.5); lift_t = int(T * 0.7)
    for t in range(T):
        if t < grasp_t:
            a = t / grasp_t
            eef[t] = start * (1 - a) + cube0 * a
            eef[t] += rng.normal(0, 0.004 * level, 3)
            if level > 1.5 and grasp_t * 0.4 < t < grasp_t * 0.6:
                eef[t, 2] += 0.05  # 재접근
        else:
            eef[t] = cube0 + np.array([0, 0, 0.12 * max(0, (t - lift_t) / (T - lift_t))])
            if t >= lift_t:
                cube[t] = eef[t]
    act = np.zeros((T, 7))
    act[:-1, :3] = np.diff(eef, axis=0) * 50 + rng.normal(0, 0.05 * level, (T - 1, 3))
    act[:, -1] = -1
    act[grasp_t:, -1] = 1
    if level > 1.5:
        act[int(grasp_t * 0.5):int(grasp_t * 0.55), -1] = 1  # 헛집기
    width = np.where(act[:, -1] > 0, 0.01, 0.08)
    width = np.convolve(width, np.ones(5) / 5, mode="same")
    qpos = np.stack([width / 2, -width / 2], 1)
    obj = np.concatenate([cube, np.tile([0, 0, 0, 1], (T, 1)), eef - cube], 1)
    return act, eef, qpos, obj


def main(path):
    with h5py.File(path, "w") as f:
        data = f.create_group("data"); mask = f.create_group("mask")
        idx = 0; groups = {}
        for s in ("worse", "okay", "better"):
            for op in (1, 2):
                names = []
                for _ in range(50):
                    act, eef, qpos, obj = demo(SK[s])
                    g = data.create_group(f"demo_{idx}")
                    g.create_dataset("actions", data=act)
                    o = g.create_group("obs")
                    o.create_dataset("robot0_eef_pos", data=eef)
                    o.create_dataset("robot0_gripper_qpos", data=qpos)
                    o.create_dataset("object", data=obj)
                    names.append(f"demo_{idx}"); idx += 1
                mask.create_dataset(f"{s}_operator_{op}", data=np.array(names, dtype="S"))
                groups.setdefault(s, []).extend(names)
        for s, n in groups.items():
            mask.create_dataset(s, data=np.array(n, dtype="S"))
    print("fake ->", path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tests/fake_lift_mh.hdf5")
