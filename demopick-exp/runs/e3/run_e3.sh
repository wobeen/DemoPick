#!/usr/bin/env bash
cd "$(dirname "$0")/runs/e3_lift"
export MUJOCO_GL=disable OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/home/claude/venv/bin/python; S=/home/claude/cloud-exp/scripts/60_e1_cpu_bc.py
{ for s in 0 1 2; do echo "clean all 1.0 $s"; for k in freeze lag pause truncate; do echo "$k all 1.0 $s"; echo "$k heuristic 0.5 $s"; done; done; } | \
xargs -P 2 -L 1 bash -c 'cd $0 && '"$PY $S"' --hdf5 data.hdf5 --method $1 --keep $2 --seed $3 --epochs 100 --n-rollouts 30 --horizon 300 --eval-seed 10000 --out results/e1_$1_s$3.jsonl 2>&1 | grep -v WARNING | tail -1 | cut -c1-160; echo "DONE $0 $1 s$3 $(date +%H:%M:%S)"'
echo ALL_DONE
