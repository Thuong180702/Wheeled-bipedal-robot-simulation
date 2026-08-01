#!/usr/bin/env bash
# WBC task-weight ablation: steel-man the baseline before comparing it to ACC.
#
# The main post-fix campaign runs ONE weight mode (balanced_default). A reviewer
# will reasonably ask whether WBC was tuned at all, or whether an arbitrary
# weighting was picked and then declared a loser. This sweeps the four
# non-degenerate modes in wheeled_biped/wbc/offline_task_stack.py so the paper
# can report the BEST WBC configuration rather than an arbitrary one:
#
#   balanced_default  w_torso=3   w_com=3   w_posture=1.5   <- main campaign
#   torso_priority    w_torso=10  w_com=1   w_posture=1     <- direct answer to
#                                                              "you under-weighted
#                                                               orientation"
#   posture_priority  w_torso=1   w_com=1   w_posture=10
#   com_priority      w_torso=1   w_com=10  w_posture=1
#
# feasibility_only is excluded on purpose: all task weights are zero, so it is a
# degenerate null controller, not a tuning candidate. (It is also the mode that
# bug F2 hardcoded.)
#
# Scenario subset: random_push at the nominal height, seeds 201-205. That is the
# suite where WBC fails hardest and most reproducibly in the main campaign
# (97.4% of every episode in a fallen state at all five heights), so it is the
# cheapest discriminative test. If no weighting rescues WBC here, none will.
#
# balanced_default is NOT re-run: the main campaign's
# outputs/wbc_postfix_shards/random_push__nominal__s201 shard is exactly this
# scenario set at that mode, so it is reused as the ablation's baseline arm.
# That leaves 3 shards -> 3 workers, which also runs cooler than the campaign.
#
#   bash scripts/run_wbc_taskweight_ablation.sh
#
# Run this only AFTER the main campaign finishes; it competes for the same cores.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
WORKERS="${WORKERS:-3}"
SHARD_ROOT="$ROOT/outputs/wbc_postfix_shards"
LOG_ROOT="$SHARD_ROOT/_logs"
BASELINE_SHARD="$SHARD_ROOT/random_push__nominal__s201"
mkdir -p "$LOG_ROOT"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 JAX_PLATFORMS=cpu

HEIGHT=nominal
SEEDS="201,202,203,204,205"
MODES=(torso_priority posture_priority com_priority)

if [ ! -f "$BASELINE_SHARD/full_batch_report.md" ]; then
  echo "FATAL: baseline shard not finished: $BASELINE_SHARD"
  echo "Run scripts/run_wbc_postfix_campaign.sh to completion first."
  exit 1
fi

TASKS="$SHARD_ROOT/_ablation_shards.txt"
: > "$TASKS"
for m in "${MODES[@]}"; do echo "$m" >> "$TASKS"; done

echo "Task-weight ablation: ${#MODES[@]} modes (+ balanced_default reused)"
echo "Scenario set: random_push, height=$HEIGHT, seeds=$SEEDS"
echo "Workers: $WORKERS"

run_mode() {
  local mode="$1"
  local tag="ablation__${mode}"
  if [ -f "$SHARD_ROOT/$tag/full_batch_report.md" ]; then
    echo "SKIP (done): $tag"; return 0
  fi
  local t0=$SECONDS
  # --steps 2000 matches the main campaign; push scenarios ignore it anyway
  # (their length is fixed at 2155), but keeping the flag identical means the
  # ablation and the campaign differ ONLY in task_mode.
  "$PY" scripts/run_wbc_shard.py \
      --tag "$tag" --suite random_push --height "$HEIGHT" --seeds "$SEEDS" \
      --task-mode "$mode" --steps 2000 \
      > "$LOG_ROOT/$tag.log" 2>&1
  local rc=$?
  echo "$( [ $rc -eq 0 ] && echo OK || echo "FAIL rc=$rc" )  $tag  ($((SECONDS-t0))s)"
}
export -f run_mode
export ROOT PY SHARD_ROOT LOG_ROOT HEIGHT SEEDS

xargs -P "$WORKERS" -I{} bash -c 'run_mode "$@"' _ {} < "$TASKS"

echo
echo "=== Ablation summary (best WBC across weight modes) ==="
"$PY" scripts/summarize_wbc_ablation.py
