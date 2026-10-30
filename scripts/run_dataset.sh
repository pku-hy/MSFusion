#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${1:?Usage: bash scripts/run_dataset.sh DATASET [96 192 336 720]}"
shift

case "$DATASET" in
  ETTh1|ETTh2|ETTm1|ETTm2|Weather|Electricity|Traffic|Solar) ;;
  *) printf 'Unsupported dataset: %s\n' "$DATASET" >&2; exit 2 ;;
esac

PYTHON="${PYTHON:-python}"
DATASET_DIR="${DATASET_DIR:-$ROOT_DIR/dataset}"
DEVICE="${DEVICE:-auto}"
GPU="${GPU:-0}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EPOCHS="${EPOCHS:-100}"
SEQ_LEN="${SEQ_LEN:-336}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)_$$}"

# With no horizon arguments, run all four benchmark horizons sequentially.
if [ "$#" -eq 0 ]; then
  set -- 96 192 336 720
fi
for horizon in "$@"; do
  case "$horizon" in
    96|192|336|720) ;;
    *) printf 'Unsupported prediction length: %s\n' "$horizon" >&2; exit 2 ;;
  esac
done

mkdir -p -- "$LOG_DIR"
for horizon in "$@"; do
  run_id="${DATASET}_${RUN_TAG}_pl${horizon}"
  log_file="$LOG_DIR/${run_id}.log"
  printf 'Starting %s, horizon=%s, run_id=%s\n' "$DATASET" "$horizon" "$run_id"
  "$PYTHON" -u "$ROOT_DIR/train.py" \
    --dataset "$DATASET" \
    --dataset-dir "$DATASET_DIR" \
    --device "$DEVICE" \
    --gpu "$GPU" \
    --seq-len "$SEQ_LEN" \
    --pred-len "$horizon" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --run-id "$run_id" 2>&1 | tee -- "$log_file"
done
