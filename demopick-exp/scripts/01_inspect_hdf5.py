"""데이터 구조 확인 (가장 먼저 실행).

사용: python scripts/01_inspect_hdf5.py --hdf5 <low_dim_v15.hdf5 경로>
출력: 화면 요약 + results/data_info.json

확인하는 것
  1) 시연 개수, 관측 키와 모양, 행동 차원
  2) mask(filter key) 목록과 개수 → 숙련도·작업자 라벨을 쓸 수 있는지
  3) 그리퍼 행동 부호: 행동 > 0일 때 손가락 간격이 줄어드는지(= 닫기)
  4) object 벡터 해석: object[0:3]이 큐브 위치인지(첫 스텝 높이, 성공 시 상승)
"""
import argparse
import json
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import list_demos, read_mask, skill_and_operator_labels  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--out", default="results/data_info.json")
    a = ap.parse_args()

    info = {"hdf5": os.path.abspath(a.hdf5)}
    with h5py.File(a.hdf5, "r") as f:
        demos = list_demos(f)
        info["n_demos"] = len(demos)
        d0 = f["data"][demos[0]]
        info["obs_keys"] = {k: list(d0["obs"][k].shape) for k in d0["obs"].keys()}
        info["action_shape"] = list(d0["actions"].shape)
        if "env_args" in f["data"].attrs:
            try:
                info["env_name"] = json.loads(f["data"].attrs["env_args"])["env_name"]
            except Exception:
                pass

        masks = read_mask(f)
        info["mask_counts"] = {k: len(v) for k, v in sorted(masks.items())}
        skill, operator, _ = skill_and_operator_labels(f)
        info["n_with_skill"] = len(skill)
        info["skill_counts"] = {s: sum(1 for v in skill.values() if v == s)
                                for s in ("worse", "okay", "better")}
        info["n_with_operator"] = len(operator)
        info["operators"] = sorted(set(operator.values()))

        lengths = [f["data"][d]["actions"].shape[0] for d in demos]
        info["length_min_median_max"] = [int(np.min(lengths)), int(np.median(lengths)),
                                         int(np.max(lengths))]

        # 그리퍼 부호 점검: 행동 부호와 다음 스텝 손가락 간격 변화의 관계
        corr = []
        # 행동 부호가 음→양으로 바뀐 순간부터 K스텝 뒤 손가락 간격 변화
        K = 10
        if "robot0_gripper_qpos" in d0["obs"]:
            for d in demos[:30]:
                g = f["data"][d]
                act = g["actions"][:, -1]
                q = g["obs"]["robot0_gripper_qpos"][()]
                width = np.abs(q[:, 0] - q[:, -1]) if q.shape[1] > 1 else q[:, 0]
                pos = act > 0
                for t in np.where((~pos[:-1]) & pos[1:])[0] + 1:
                    # 닫기 명령이 K스텝 이상 유지된 경우만 (짧게 눌렀다 뗀 경우 제외)
                    if t + K < len(width) and pos[t:t + K].all():
                        corr.append(width[t + K] - width[t])
        info["gripper_width_change_after_positive_action_m"] = (
            float(np.median(corr)) if corr else None)
        info["gripper_note"] = (
            "음수(간격 감소)면 '행동>0 = 닫기'(robosuite 관례)와 일치. "
            "양수면 common.gripper_closed의 부등호를 반대로 바꾼다.")

        # object 벡터 점검: [0:3]이 큐브 위치라면 끝 높이 > 처음 높이 (Lift 성공 시연)
        if "object" in d0["obs"]:
            rises, rel_err = [], []
            for d in demos[:30]:
                g = f["data"][d]["obs"]
                obj = g["object"][()]
                rises.append(float(obj[-1, 2] - obj[0, 2]))
                if obj.shape[1] >= 10 and "robot0_eef_pos" in g:
                    eef = g["robot0_eef_pos"][()]
                    a1 = np.linalg.norm(obj[:, 7:10], axis=1)
                    a2 = np.linalg.norm(eef - obj[:, 0:3], axis=1)
                    rel_err.append(float(np.median(np.abs(a1 - a2))))
            info["object_z_rise_median_m"] = float(np.median(rises))
            info["object_rel_vs_eef_dist_median_abs_diff_m"] = (
                float(np.median(rel_err)) if rel_err else None)
            info["object_note"] = (
                "z 상승 중앙값이 수 cm 이상이면 object[0:3]=큐브 위치로 본다. "
                "rel 차이가 작으면 object[7:10]=그리퍼-큐브 상대 위치로 본다(작업 중 그리퍼 기준점 차이로 1~2cm는 날 수 있음).")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fp:
        json.dump(info, fp, ensure_ascii=False, indent=2)
    print(json.dumps(info, ensure_ascii=False, indent=2))

    problems = []
    if info["n_with_skill"] < info["n_demos"]:
        problems.append(f"숙련도 라벨 없는 시연 {info['n_demos'] - info['n_with_skill']}개")
    if info["n_with_operator"] == 0:
        problems.append("작업자 ID 없음 → E2-b는 숙련도 그룹 단위로 대신")
    print("\n[점검] " + ("문제 없음" if not problems else " / ".join(problems)))


if __name__ == "__main__":
    main()
