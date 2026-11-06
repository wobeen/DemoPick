# demopick-exp — 초기 작업 메모 (2026-11-03)

> 실험 시작 시점의 메모를 그대로 보존한 문서입니다. 최신 안내는 저장소 맨 위 `README.md`를 보세요.
> 아래 `30_`, `40_`, `50_` 스크립트는 작성하지 않았습니다(CUPID·E2를 실행하지 않음). 아래의 상대 경로는 이 메모가 `demopick-exp/`에 있던 때 기준입니다.

계획: `../12_demopick_dev_plan.md` · 기록: `notes/experiment_log.md`

## 지금 할 일 (E0, GPU 불필요, 로컬 PC)

### 1. 데이터 받기 (브라우저로 직접)
robomimic 공식 문서의 Lift MH low_dim 파일 (작업자 6명 × 50개 = 300개 시연):

https://huggingface.co/datasets/amandlek/robomimic/resolve/main/v1.5/lift/mh/low_dim_v15.hdf5?download=true

- 출처: robomimic `docs/datasets/robomimic_v0.1.md` 다운로드 표 (MH · lift · low_dim)
- **OneDrive 폴더 말고** 영문 로컬 경로에 저장한다. 예: `C:\data\robomimic\lift_mh_low_dim_v15.hdf5`
- 서버에서 CUPID를 돌린다면, CUPID가 쓰는 파일(diffusion_policy zip 안의 lift/mh low_dim hdf5)로도
  10번 스크립트를 한 번 더 돌려 E2에서 같은 파일 기준으로 맞춘다.

### 2. 환경 (한 번만)
```powershell
cd demopick-exp
uv sync                      # uv가 없으면: pip install numpy pandas h5py scipy scikit-learn matplotlib pyyaml
```

### 3. 실행
```powershell
uv run python scripts/01_inspect_hdf5.py --hdf5 C:\data\robomimic\lift_mh_low_dim_v15.hdf5
uv run python scripts/10_extract_features.py --hdf5 C:\data\robomimic\lift_mh_low_dim_v15.hdf5
uv run python scripts/11_e0_skill_vs_heuristic.py
```
또는 한 번에: `.\run_e0.ps1 C:\data\robomimic\lift_mh_low_dim_v15.hdf5` (Mac·Linux: `bash run_e0.sh <경로>`)

### 4. 결과 확인
| 파일 | 내용 |
|---|---|
| `results/data_info.json` | 데이터 구조 점검. **`[점검]` 줄과 그리퍼·object 점검 값을 먼저 본다** |
| `results/features.csv` | 시연 300줄 × 특징 13개 |
| `results/e0_feature_auroc.csv` | 특징별 AUROC(worse vs better) |
| `results/e0_summary.json` | 합성 점수 AUROC·95% 구간·숙련도 순위 상관 |
| `results/figures/e0_feature_auroc.png` | 덱 7장용 그림 |

`01_inspect` 점검에서 볼 것:
- `skill_counts`가 100/100/100인가 (아니면 mask 이름이 다른 것 → `common.py` 수정)
- `gripper_width_change_after_positive_action_m`가 음수인가 (양수면 `common.gripper_closed` 부등호 반대로)
- `object_z_rise_median_m`가 수 cm 이상인가 (아니면 object 벡터 해석이 다름 → B 특징 보류)

## 스크립트

| 스크립트 | 어디서 | 하는 일 | 상태 |
|---|---|---|---|
| `01_inspect_hdf5.py` | 로컬 | 데이터 구조·라벨·그리퍼 부호·object 해석 점검 | 가짜 데이터로 검증 완료 |
| `10_extract_features.py` | 로컬 | 휴리스틱 7 + 관측값 6 특징 | 가짜 데이터로 검증 완료 |
| `11_e0_skill_vs_heuristic.py` | 로컬 | E0 표·그림 | 가짜 데이터로 검증 완료 |
| `20_heuristic_to_curation.py` | 서버 | 휴리스틱 순위를 CUPID `train_config.yaml`에 `demopick_heuristic`으로 추가 | 가짜 config로 검증 완료 |
| `30_export_cupid_scores.py` | 서버 | CUPID 시연별 점수 → CSV | 다음 단계 |
| `40_e2_light_scorer.py` | 로컬 | E2 | 다음 단계 |
| `50_figures.py` | 로컬 | E1 그래프·E2 표 | 다음 단계 |

`tests/make_fake_hdf5.py`는 **동작 검증용 가짜 데이터**다. 여기서 나온 수치는 어디에도 쓰지 않는다.

## CUPID 저장소에서 확인한 사실 (2026-11-03, github.com/agiachris/cupid)
- `filter_ratio` = 학습 데이터에서 **제거하는** 비율. 설정 파일의 목록 앞쪽(나쁜 시연)부터 제거 (`diffusion_policy/common/sampler.py`)
- 필터 실험 기본 분할: 학습 64%(=192개), 검증 4%, 나머지 holdout. 숙련도별 균등 추출(`uniform_quality=true`). **CUPID 점수는 학습 192개에만 생긴다**
- 에피소드 번호 = `demo_N`의 N
- `retrain_policies.sh` 기본 방법: `influence_sum_official`, `influence_quality_official`, `oracle`, `random`. State/Action Diversity·Policy Loss는 저자들이 "성능이 낮아 제외"로 주석 처리
- 방법 이름은 `oracle`/`random`/`influence*`/`*similarity`/`dem*`/`*diversity`/`*loss`만 통과 → 우리 이름은 `demopick_heuristic`
- `curation_filter_ratios`와 `curation_select_ratios`는 같은 위치끼리 짝 → 줄일 때 두 배열을 같이 줄인다
- 에폭: Lift MH low_dim 기본 학습 1001, 재학습 filter 0.50은 2301 (0.25는 별도 값 없음 → 1001)
- **노트북 주석: "Demo-SCORE는 Lift MH에서 작동하지 않는다. 정책이 실패를 보이지 않기 때문"** → Lift MH는 기본 정책 성공률이 매우 높을 가능성. E1 천장 효과 위험이 실제로 있다
