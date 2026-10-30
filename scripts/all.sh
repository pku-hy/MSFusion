#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export DATASET_DIR="${DATASET_DIR:-$SCRIPT_DIR/../dataset}"
export RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)_$$}"

# Check the complete benchmark data layout before starting any training.
files=(ETT-small/ETTh1.csv ETT-small/ETTh2.csv ETT-small/ETTm1.csv
       ETT-small/ETTm2.csv weather/weather.csv electricity/electricity.csv
       traffic/traffic.csv solar/solar_AL.txt)
for file in "${files[@]}"; do
  if [ ! -f "$DATASET_DIR/$file" ]; then
    printf 'Dataset not found: %s/%s\n' "$DATASET_DIR" "$file" >&2
    exit 2
  fi
done

for dataset in ETTh1 ETTh2 ETTm1 ETTm2 Weather Electricity Traffic Solar; do
  bash "$SCRIPT_DIR/run_dataset.sh" "$dataset" "$@"
done
