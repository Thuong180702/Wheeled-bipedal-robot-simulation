#!/usr/bin/env bash
# Post-fix WBC campaign: the full 225-scenario three-arm batch, re-run on the
# tree that contains the F1-F4 audit fixes (commit 433160f), sharded across
# cores. Serial cost is ~69 h; 8 workers bring it to ~9 h wall clock.
#
# Each shard is the unmodified suite runner over a subset of
# (suite x height x seed) writing to its own directory, so shards never
# contend for the JSONL. scripts/merge_wbc_shards.py stitches them back.
#
#   bash scripts/run_wbc_postfix_campaign.sh          # 8 workers
#   WORKERS=6 bash scripts/run_wbc_postfix_campaign.sh
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
WORKERS="${WORKERS:-8}"
SHARD_ROOT="$ROOT/outputs/wbc_postfix_shards"
LOG_ROOT="$SHARD_ROOT/_logs"
mkdir -p "$LOG_ROOT"

# One process per shard; keep BLAS/JAX single-threaded so 8 shards do not
# oversubscribe the 8 performance cores.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 JAX_PLATFORMS=cpu

HEIGHTS=(nominal low_tiny high_tiny low_small high_small)
TASKS="$SHARD_ROOT/_shards.txt"
: > "$TASKS"

# Longest shards first so the tail of the run is not one straggler.
for h in "${HEIGHTS[@]}"; do
  for g in "201,202,203,204,205" "206,207,208,209,210" \
           "211,212,213,214,215" "216,217,218,219,220"; do
    echo "random_push__${h}__s${g%%,*}|random_push|$h|$g" >> "$TASKS"
  done
done
for h in "${HEIGHTS[@]}"; do
  for s in 42 113 999 77 201; do
    echo "single_push__${h}__s${s}|single_push|$h|$s" >> "$TASKS"
  done
done
for h in "${HEIGHTS[@]}"; do
  for s in 42 113 999; do
    echo "step_d__${h}__s${s}|step_d|$h|$s" >> "$TASKS"
  done
done
for h in "${HEIGHTS[@]}"; do echo "step_e__${h}|step_e|$h|"  >> "$TASKS"; done
for h in "${HEIGHTS[@]}"; do echo "step_c__${h}|step_c|$h|"  >> "$TASKS"; done

echo "Shards: $(wc -l < "$TASKS")   workers: $WORKERS"

# V3 truth check once for the whole campaign (same commit for every shard).
echo "Running V3 baseline truth check once..."
"$PY" scripts/phase3d_v3_baseline_truth_check.py \
  > "$LOG_ROOT/v3_truth_check.log" 2>&1
if grep -q "Baseline Truth Check: PASS" "$LOG_ROOT/v3_truth_check.log"; then
  echo "V3 truth check: PASS"
else
  echo "FATAL: V3 truth check did not pass; see $LOG_ROOT/v3_truth_check.log"
  exit 1
fi

run_shard() {
  IFS='|' read -r tag suite height seeds <<< "$1"
  # Already finished on an earlier invocation? Skip without paying startup.
  if [ -f "$SHARD_ROOT/$tag/full_batch_report.md" ]; then
    echo "SKIP (done): $tag"; return 0
  fi
  local t0=$SECONDS
  # --steps 2000 matches the protocol the superseded pre-fix campaign
  # documented ("225 scenarios x 2000 steps",
  # docs/validation/V3_vs_V3_Assist_comparison_report.md), so the post-fix
  # numbers differ from it only by the F1-F4 repairs. Push scenarios ignore
  # this: their length is push_warmup + push_duration + post_push = 2155.
  local args=(--tag "$tag" --suite "$suite" --height "$height" --steps 2000)
  [ -n "$seeds" ] && args+=(--seeds "$seeds")
  "$PY" scripts/run_wbc_shard.py "${args[@]}" > "$LOG_ROOT/$tag.log" 2>&1
  local rc=$?
  echo "$( [ $rc -eq 0 ] && echo OK || echo "FAIL rc=$rc" )  $tag  ($((SECONDS-t0))s)"
}
export -f run_shard
export ROOT PY SHARD_ROOT LOG_ROOT

xargs -P "$WORKERS" -I{} bash -c 'run_shard "$@"' _ {} < "$TASKS"

echo "All shards finished. Merge with:"
echo "  $PY scripts/merge_wbc_shards.py"
