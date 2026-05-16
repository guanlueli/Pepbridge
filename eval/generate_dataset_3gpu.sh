#!/usr/bin/env bash
#
# Launch generate_dataset.py in parallel across N GPUs by sharding the complex
# list via --shard / --num-shards. Each shard is its own Python process so the
# model is replicated on its assigned GPU with zero inter-GPU communication.
#
# Idempotent: generate_dataset.py skips per-complex output dirs that already
# contain the expected pdbs, so you can ctrl-C and re-run.
#
# Required env vars (no portable default exists):
#   DATASET_ROOT      Per-complex peptide.pdb + pocket.pdb tree
#   LMDB_DIR          Directory holding <LMDB_NAME>_structure_cache.lmdb
#   CHECKPOINT        Path to the .pt model checkpoint
#
# Common overrides:
#   GPUS="1,2,3"      Comma-separated GPU ids
#   OUTPUT_ROOT=...   Default: ./outputs/designs
#   NUM_SAMPLES=50    Designs per complex (default 4)
#   NUM_STEPS=200     Bridge steps (default 1000)
#   MINI_BATCH_SIZE=10
#   ONLY_FROM=/path/to/ids.txt
#   EXTRA_ARGS="--overwrite --limit 20"
#   PY=/path/to/python
#   DRY_RUN=1         Print commands, do not exec
#
# Logs land at $OUTPUT_ROOT/_logs/shard_<i>_gpu_<n>.log.

set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)
SCRIPT=${SCRIPT:-$REPO_ROOT/eval/generate_dataset.py}
CONFIG=${CONFIG:-$REPO_ROOT/configs/learn_surf_angle.yaml}
PY=${PY:-python}

: "${DATASET_ROOT:?DATASET_ROOT is required — point at the per-complex peptide.pdb tree}"
: "${LMDB_DIR:?LMDB_DIR is required — directory containing <LMDB_NAME>_structure_cache.lmdb}"
: "${CHECKPOINT:?CHECKPOINT is required — path to the .pt model checkpoint}"

OUTPUT_ROOT=${OUTPUT_ROOT:-./outputs/designs}
LMDB_NAME=${LMDB_NAME:-pep_pocket_test_surf}

NUM_SAMPLES=${NUM_SAMPLES:-4}
NUM_STEPS=${NUM_STEPS:-1000}
MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-$NUM_SAMPLES}
SAMPLE_SURF=${SAMPLE_SURF:-True}
SAMPLE_BB=${SAMPLE_BB:-True}
SAMPLE_ANG=${SAMPLE_ANG:-True}
SAMPLE_SEQ=${SAMPLE_SEQ:-True}
SEED=${SEED:-114514}

ONLY_FROM=${ONLY_FROM:-}
EXTRA_ARGS=${EXTRA_ARGS:-}

GPUS=${GPUS:-1,2,3}
IFS=',' read -r -a GPU_ARR <<< "$GPUS"
NUM_SHARDS=${#GPU_ARR[@]}

LOG_DIR=${LOG_DIR:-$OUTPUT_ROOT/_logs}
mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_ROOT"

COMMON=(
  --dataset-root  "$DATASET_ROOT"
  --output-root   "$OUTPUT_ROOT"
  --checkpoint    "$CHECKPOINT"
  --config        "$CONFIG"
  --lmdb-dir      "$LMDB_DIR"
  --lmdb-name     "$LMDB_NAME"
  --num-samples   "$NUM_SAMPLES"
  --num-steps     "$NUM_STEPS"
  --mini-batch-size "$MINI_BATCH_SIZE"
  --sample-surf   "$SAMPLE_SURF"
  --sample-bb     "$SAMPLE_BB"
  --sample-ang    "$SAMPLE_ANG"
  --sample-seq    "$SAMPLE_SEQ"
  --seed          "$SEED"
  --num-shards    "$NUM_SHARDS"
)
[[ -n "$ONLY_FROM" ]] && COMMON+=(--only-from "$ONLY_FROM")
if [[ -n "$EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARR=($EXTRA_ARGS)
  COMMON+=("${EXTRA_ARR[@]}")
fi

echo "=== PepBridge multi-GPU generation ==="
echo "  repo:        $REPO_ROOT"
echo "  python:      $PY"
echo "  dataset:     $DATASET_ROOT"
echo "  lmdb:        $LMDB_DIR/${LMDB_NAME}_structure_cache.lmdb"
echo "  ckpt:        $CHECKPOINT"
echo "  output:      $OUTPUT_ROOT"
echo "  GPUs:        $GPUS  ($NUM_SHARDS shards)"
echo "  N samples:   $NUM_SAMPLES   N steps: $NUM_STEPS   mini batch: $MINI_BATCH_SIZE"
echo "  log dir:     $LOG_DIR"
[[ -n "$ONLY_FROM" ]] && echo "  only-from:   $ONLY_FROM"
[[ -n "$EXTRA_ARGS" ]] && echo "  extra args:  $EXTRA_ARGS"
echo

PIDS=()
LOGS=()
for i in "${!GPU_ARR[@]}"; do
  GPU=${GPU_ARR[$i]}
  LOG="$LOG_DIR/shard_${i}_gpu_${GPU}.log"
  LOGS+=("$LOG")
  CMD=("$PY" "$SCRIPT" "${COMMON[@]}" --shard "$i" --device "cuda:$GPU")

  echo "[shard $i / gpu $GPU] log: $LOG"
  if [[ "${DRY_RUN:-}" == "1" ]]; then
    echo "  DRY_RUN: ${CMD[*]}"
    continue
  fi

  "${CMD[@]}" >"$LOG" 2>&1 &
  PIDS+=($!)
done

if [[ "${DRY_RUN:-}" == "1" ]]; then
  echo
  echo "DRY_RUN=1: not launching."
  exit 0
fi

echo
echo "Launched ${#PIDS[@]} shards (pids: ${PIDS[*]}). Tailing logs..."
echo "  (ctrl-C to detach; shards continue running. Kill ${PIDS[*]} to abort.)"
echo

trap 'echo; echo "Detaching from tail. Shards still running (pids: ${PIDS[*]})."; exit 0' INT

# Wait briefly for at least one log to materialise, then use tail's multi-file
# follow mode — it prints "==> file <==" headers when the active file changes.
for _ in {1..50}; do
  for L in "${LOGS[@]}"; do
    [[ -f "$L" ]] && break 2
  done
  sleep 0.2
done

tail -n +1 -F "${LOGS[@]}" 2>/dev/null &
TAIL_PID=$!

RC=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    RC=1
  fi
done

sleep 1
kill "$TAIL_PID" 2>/dev/null || true
wait "$TAIL_PID" 2>/dev/null || true

echo
echo "=== all shards finished (rc=$RC) ==="
for i in "${!LOGS[@]}"; do
  GPU=${GPU_ARR[$i]}
  LOG="${LOGS[$i]}"
  TAIL=$(tail -n 3 "$LOG" 2>/dev/null | tr '\n' ' ' | tr -s ' ')
  echo "  shard $i / gpu $GPU: $TAIL"
done

N_DESIGN=$(find "$OUTPUT_ROOT" -type f -name "sample_*.pdb" -not -path "*/refold_cif/*" 2>/dev/null | wc -l)
N_REFOLD=$(find "$OUTPUT_ROOT" -type f -path "*/refold_cif/sample_*.pdb" 2>/dev/null | wc -l)
echo
echo "Output summary:"
echo "  design pdbs:  $N_DESIGN"
echo "  refold pdbs:  $N_REFOLD"
echo "  output root:  $OUTPUT_ROOT"

exit "$RC"
