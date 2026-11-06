#!/usr/bin/env bash
# E1-lite 전체 실행 (사전 고정: epochs 100, rollouts 50, horizon 400, eval_seed 10000)
# 사용: bash run_e1_cpu.sh <hdf5> [python]
H5="$1"; PY="${2:-python}"
cd "$(dirname "$0")"
export MUJOCO_GL=disable OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
jobs=()
for seed in 0 1 2; do
  jobs+=("all 1.0 $seed")
  for keep in 0.5 0.33; do
    for m in heuristic oracle random reverse; do jobs+=("$m $keep $seed"); done
  done
done
printf '%s\n' "${jobs[@]}" | xargs -P 2 -L 1 bash -c \
  "$PY scripts/60_e1_cpu_bc.py --hdf5 '$H5' --method \$0 --keep \$1 --seed \$2 --epochs 100 --n-rollouts 50 --horizon 400 --out results/e1_cpu.jsonl 2>&1 | grep -v WARNING"
echo ALL_DONE
