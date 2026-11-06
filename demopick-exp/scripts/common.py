"""데모픽 실험 공통 함수: robomimic hdf5 읽기, 숙련도·작업자 라벨 찾기.

robomimic v1.5 low_dim 형식을 기준으로 한다.
  data/demo_N/actions            (T, 7)  위치 3 + 자세 3 + 그리퍼 1
  data/demo_N/obs/robot0_eef_pos (T, 3)
  data/demo_N/obs/object         (T, 10) Lift: 큐브 위치 3 + 자세 4 + 그리퍼→큐브 3
  mask/<filter_key>              시연 이름 목록 (MH: worse/okay/better 등)
실제 키 이름은 01_inspect_hdf5.py로 확인한다.
"""
import re

import numpy as np

SKILLS = ("worse", "okay", "better")
SKILL_ORDINAL = {"worse": 0, "okay": 1, "better": 2}


def demo_sort_key(name):
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else name


def list_demos(f):
    return sorted(f["data"].keys(), key=demo_sort_key)


def read_mask(f):
    """mask 그룹을 {filter_key: set(demo 이름)}으로 읽는다. 없으면 빈 dict."""
    out = {}
    if "mask" not in f:
        return out
    for k in f["mask"].keys():
        names = f["mask"][k][()]
        out[k] = {n.decode() if isinstance(n, bytes) else str(n) for n in names}
    return out


def skill_and_operator_labels(f):
    """시연별 숙련도(worse/okay/better)와 작업자 ID를 mask에서 찾는다.

    - 숙련도: 이름이 정확히 worse/okay/better인 filter key
    - 작업자: '<skill>_operator_<n>' 형태의 filter key (있을 때만)
    찾지 못한 값은 None. 겹치면(한 시연이 두 숙련도에 속하면) 오류를 낸다.
    """
    masks = read_mask(f)
    skill, operator = {}, {}
    for s in SKILLS:
        for d in masks.get(s, ()):
            if d in skill and skill[d] != s:
                raise ValueError(f"{d}가 숙련도 {skill[d]}와 {s}에 모두 속함")
            skill[d] = s
    op_pat = re.compile(r"^(worse|okay|better)_operator_(\d+)$")
    for k, names in masks.items():
        m = op_pat.match(k)
        if not m:
            continue
        for d in names:
            operator[d] = f"{m.group(1)}_{m.group(2)}"
    return skill, operator, masks


def gripper_closed(actions):
    """robosuite 관례: 그리퍼 행동 > 0 이면 닫기, < 0 이면 열기.
    부호가 반대인 데이터라면 01_inspect의 점검 결과를 보고 SIGN을 바꾼다."""
    return actions[:, -1] > 0


# 사전 고정: 모든 특징은 '클수록 나쁜 시연'으로 정의돼 있다 (10_extract_features.py 참고)
def badness(df, cols):
    """특징을 z-점수로 바꿔 평균한 '나쁨 점수'. 클수록 나쁜 시연."""
    import numpy as _np
    z = (df[cols] - df[cols].mean()) / df[cols].std(ddof=0).replace(0, _np.nan)
    return z.fillna(0).mean(axis=1)


def demo_index(name):
    """'demo_17' → 17. CUPID는 이 번호를 에피소드 인덱스로 쓴다."""
    return int(str(name).split("_")[-1])
