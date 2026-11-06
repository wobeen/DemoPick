#!/usr/bin/env bash
# E0 한 번에 실행 (Mac·Linux). 사용: bash run_e0.sh /path/to/lift_mh_low_dim_v15.hdf5
set -euo pipefail
cd "$(dirname "$0")"
H5="$1"
uv run python scripts/01_inspect_hdf5.py --hdf5 "$H5"
uv run python scripts/10_extract_features.py --hdf5 "$H5"
uv run python scripts/11_e0_skill_vs_heuristic.py
