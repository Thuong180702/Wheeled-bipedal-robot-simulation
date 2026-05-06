# Experiment Plan

**Date:** 2026-05-06  
**Status:** Ready for execution after paper review  
**Purpose:** Document exact commands for final training and evaluation

---

## Prerequisites

1. All Phase A-C code complete
2. All tests passing
3. Paper draft reviewed and approved
4. GPU machine provisioned

---

## Training Experiments

### 1. Short Residual Sanity Run (1M steps)

**Purpose:** Quick verification that residual training works end-to-end.

```bash
python scripts/train.py single \
  --stage balance_residual \
  --steps 1000000 \
  --seed 42 \
  --num-envs 2048 \
  --output-dir outputs/residual_sanity_1M
```

**Expected duration:** ~30 minutes on A100  
**Success criteria:** No NaN, curriculum advances, residual metrics logged

---

### 2. Medium Residual Check (10M steps)

**Purpose:** Verify training stability and curriculum progression.

```bash
python scripts/train.py single \
  --stage balance_residual \
  --steps 10000000 \
  --seed 42 \
  --num-envs 4096 \
  --output-dir outputs/residual_medium_10M
```

**Expected duration:** ~4 hours on A100  
**Success criteria:** Survival rate > 50%, height RMSE < 0.05m, no training collapse

---

### 3. Main Residual Training (50M steps, 3 seeds)

**Purpose:** Primary results for paper.

```bash
# Seed 42
python scripts/train.py single \
  --stage balance_residual \
  --steps 50000000 \
  --seed 42 \
  --num-envs 4096 \
  --output-dir outputs/residual_main_50M/seed42

# Seed 113
python scripts/train.py single \
  --stage balance_residual \
  --steps 50000000 \
  --seed 113 \
  --num-envs 4096 \
  --output-dir outputs/residual_main_50M/seed113

# Seed 999
python scripts/train.py single \
  --stage balance_residual \
  --steps 50000000 \
  --seed 999 \
  --num-envs 4096 \
  --output-dir outputs/residual_main_50M/seed999
```

**Expected duration:** ~20 hours per seed on A100  
**Total GPU time:** ~60 hours  
**Success criteria:** Mean survival rate > 80%, height RMSE < 0.03m across 3 seeds

---

### 4. Extended Training (200M steps, optional)

**Purpose:** If 50M is insufficient, extend to 200M.

```bash
# Resume from 50M checkpoint
python scripts/train.py single \
  --stage balance_residual \
  --steps 200000000 \
  --seed 42 \
  --num-envs 4096 \
  --resume outputs/residual_main_50M/seed42/checkpoints/final \
  --output-dir outputs/residual_extended_200M/seed42
```

**Expected duration:** ~80 hours per seed on A100  
**Decision point:** Only run if 50M results are promising but incomplete

---

### 5. Pure PPO Reference Baseline (50M steps, 1 seed)

**Purpose:** Fair comparison baseline.

```bash
python scripts/train.py single \
  --stage balance \
  --steps 50000000 \
  --seed 42 \
  --num-envs 4096 \
  --output-dir outputs/pure_ppo_baseline_50M/seed42
```

**Expected duration:** ~20 hours on A100  
**Note:** Use existing balance.yaml config, not balance_residual.yaml

---

## Evaluation Experiments

### 1. LQR/IK Prior Baseline (already done in Phase B.5)

**Purpose:** Establish prior-only performance.

```bash
# Fixed-height sweep
python scripts/eval_classical_priors.py \
  --variants geometric_lqr_ik \
  --scenarios fixed_height_sweep \
  --episodes 20 \
  --output-dir outputs/lqr_ik_baseline
```

**Status:** DONE (see docs/phase_b5_classical_prior_variants_report.md)

---

### 2. Residual PPO Evaluation (all scenarios)

**Purpose:** Main paper results.

```bash
# For each seed checkpoint
CKPT=outputs/residual_main_50M/seed42/checkpoints/final

# Nominal standing
python scripts/eval_balance.py \
  --checkpoint $CKPT \
  --stage balance_residual \
  --scenarios nominal \
  --num-episodes 100 \
  --output-dir $CKPT/eval/nominal

# Fixed-height sweep
python scripts/eval_balance.py \
  --checkpoint $CKPT \
  --stage balance_residual \
  --scenarios fixed_height_sweep \
  --heights 0.70 0.65 0.60 0.55 0.50 0.45 0.40 \
  --num-episodes 50 \
  --output-dir $CKPT/eval/fixed_height

# Random-height balance
python scripts/eval_balance.py \
  --checkpoint $CKPT \
  --stage balance_residual \
  --scenarios random_height \
  --num-episodes 100 \
  --output-dir $CKPT/eval/random_height

# Height transitions
python scripts/eval_balance.py \
  --checkpoint $CKPT \
  --stage balance_residual \
  --scenarios height_transition \
  --num-episodes 50 \
  --output-dir $CKPT/eval/height_transition

# Push recovery
python scripts/eval_balance.py \
  --checkpoint $CKPT \
  --stage balance_residual \
  --scenarios push_recovery \
  --push-magnitudes 20 40 60 80 100 120 140 160 180 200 \
  --num-episodes 20 \
  --output-dir $CKPT/eval/push_recovery

# Robustness sweep
python scripts/eval_balance.py \
  --checkpoint $CKPT \
  --stage balance_residual \
  --scenarios robustness \
  --num-episodes 50 \
  --output-dir $CKPT/eval/robustness
```

**Expected duration:** ~2 hours per seed for all scenarios  
**Total eval time:** ~6 hours for 3 seeds

---

### 3. Pure PPO Evaluation (same scenarios)

**Purpose:** Comparison baseline.

```bash
CKPT=outputs/pure_ppo_baseline_50M/seed42/checkpoints/final

# Run same eval scenarios as residual PPO
# (use --stage balance instead of balance_residual)
```

---

### 4. Residual Action Analysis

**Purpose:** Understand residual behavior.

```bash
python scripts/analyze_residual.py \
  --checkpoint outputs/residual_main_50M/seed42/checkpoints/final \
  --scenarios nominal random_height push_recovery \
  --num-episodes 50 \
  --output-dir outputs/residual_analysis/seed42
```

**Outputs:**
- base_action_abs RMS by joint group
- residual_action RMS by joint group
- residual_to_base_ratio
- residual_saturation_rate
- residual distribution histograms

---

### 5. Ablation Studies

**Purpose:** Validate design choices.

```bash
# Ablation 1: No base_action in obs (train from scratch)
python scripts/train.py single \
  --stage balance_residual_ablation_no_base_obs \
  --steps 50000000 \
  --seed 42 \
  --output-dir outputs/ablation/no_base_obs

# Ablation 2: Different residual_scale
python scripts/train.py single \
  --stage balance_residual_ablation_scale_0.5x \
  --steps 50000000 \
  --seed 42 \
  --output-dir outputs/ablation/scale_0.5x

# Ablation 3: No residual penalty
python scripts/train.py single \
  --stage balance_residual_ablation_no_penalty \
  --steps 50000000 \
  --seed 42 \
  --output-dir outputs/ablation/no_penalty
```

**Note:** Create ablation configs before running

---

## Export Results for Paper

### 1. Aggregate Multi-Seed Results

```bash
python scripts/compare_controllers.py \
  --lqr-ik outputs/lqr_ik_baseline \
  --pure-ppo outputs/pure_ppo_baseline_50M/seed42/eval \
  --residual-ppo outputs/residual_main_50M/seed*/eval \
  --output-dir outputs/paper_results
```

**Outputs:**
- `controller_comparison.csv`
- `mean_std_tables.csv`
- `scenario_breakdown.csv`

---

### 2. Export LaTeX Tables

```bash
python scripts/export_results.py \
  --input outputs/paper_results \
  --format latex \
  --output paper/tables
```

**Outputs:**
- `table_random_height.tex`
- `table_fixed_height.tex`
- `table_height_transition.tex`
- `table_push_recovery.tex`
- `table_robustness.tex`
- `table_residual_analysis.tex`
- `table_ablation.tex`

---

### 3. Generate Figures

```bash
python scripts/plot_results.py \
  --input outputs/paper_results \
  --output paper/figures
```

**Outputs:**
- `training_curves.pdf`
- `height_tracking.pdf`
- `push_recovery.pdf`
- `residual_distribution.pdf`
- `robustness_sweep.pdf`

---

## tmux Session Management (for GPU machine)

### Start Training Sessions

```bash
# Create tmux session for each seed
tmux new-session -d -s residual_seed42
tmux send-keys -t residual_seed42 "cd /path/to/repo && source .venv/bin/activate" C-m
tmux send-keys -t residual_seed42 "python scripts/train.py single --stage balance_residual --steps 50000000 --seed 42 --num-envs 4096 --output-dir outputs/residual_main_50M/seed42" C-m

tmux new-session -d -s residual_seed113
tmux send-keys -t residual_seed113 "cd /path/to/repo && source .venv/bin/activate" C-m
tmux send-keys -t residual_seed113 "python scripts/train.py single --stage balance_residual --steps 50000000 --seed 113 --num-envs 4096 --output-dir outputs/residual_main_50M/seed113" C-m

tmux new-session -d -s residual_seed999
tmux send-keys -t residual_seed999 "cd /path/to/repo && source .venv/bin/activate" C-m
tmux send-keys -t residual_seed999 "python scripts/train.py single --stage balance_residual --steps 50000000 --seed 999 --num-envs 4096 --output-dir outputs/residual_main_50M/seed999" C-m
```

### Monitor Training

```bash
# List sessions
tmux ls

# Attach to session
tmux attach -t residual_seed42

# Detach: Ctrl+b, then d

# Kill session
tmux kill-session -t residual_seed42
```

### Monitor GPU Usage

```bash
# Watch GPU utilization
watch -n 1 nvidia-smi

# Check training logs
tail -f outputs/residual_main_50M/seed42/train.log
```

---

## Expected Output Folder Structure

```
outputs/
├── residual_sanity_1M/
│   └── balance_residual/rl/seed42/
├── residual_medium_10M/
│   └── balance_residual/rl/seed42/
├── residual_main_50M/
│   ├── seed42/
│   │   ├── checkpoints/
│   │   │   ├── best_eval_success/
│   │   │   ├── best_eval_per_step/
│   │   │   └── final/
│   │   ├── eval/
│   │   │   ├── nominal/
│   │   │   ├── fixed_height/
│   │   │   ├── random_height/
│   │   │   ├── height_transition/
│   │   │   ├── push_recovery/
│   │   │   └── robustness/
│   │   └── train.log
│   ├── seed113/
│   └── seed999/
├── pure_ppo_baseline_50M/
│   └── seed42/
├── ablation/
│   ├── no_base_obs/
│   ├── scale_0.5x/
│   └── no_penalty/
├── residual_analysis/
│   ├── seed42/
│   ├── seed113/
│   └── seed999/
└── paper_results/
    ├── controller_comparison.csv
    ├── mean_std_tables.csv
    └── scenario_breakdown.csv
```

---

## Checkpoint Validation

**IMPORTANT:** `validate_checkpoint.py` validates checkpoint metadata and compatibility only.
It does NOT perform full residual environment rollout with LQR/IK prior composition.
For full residual evaluation with action decomposition and residual-specific metrics,
use `eval_balance.py` instead (see Evaluation Experiments section above).

Before evaluation, validate each checkpoint:

```bash
# Pure PPO checkpoint validation
python scripts/validate_checkpoint.py \
  --checkpoint outputs/pure_ppo_baseline_50M/seed42/checkpoints/final \
  --stage balance

# Residual PPO checkpoint validation
python scripts/validate_checkpoint.py \
  --checkpoint outputs/residual_main_50M/seed42/checkpoints/final \
  --stage balance_residual
```

**Expected output for residual checkpoint:**
```
Checkpoint type : residual_lqr_ik
Obs dim         : 52
Base obs dim    : 42
Base action dim : 10
Action type     : bounded_residual
Residual scale  : [0.10, 0.05, 0.15, 0.15, 0.30, ...]
Validation      : PASS
```

**What validate_checkpoint.py checks:**
- Checkpoint metadata (policy_type, action_mode, obs_dim, action_dim)
- Residual scale shape and values
- Obs size compatibility with lin_vel_mode
- Benchmark metrics (nominal scenario)
- Standing quality signals (wheel spin, drift, oscillation)

**What validate_checkpoint.py does NOT check:**
- Full residual action decomposition (base + residual → final)
- Residual-specific metrics (residual_to_base_ratio, saturation_rate)
- Multi-scenario evaluation (push recovery, robustness, etc.)
- Per-joint residual analysis

For comprehensive residual evaluation, use `eval_balance.py` with residual checkpoints.

---

## Troubleshooting

### Training NaN

```bash
# Check for NaN in checkpoint
python -c "import pickle; ckpt = pickle.load(open('checkpoint.pkl', 'rb')); import jax.numpy as jnp; print('NaN in params:', jnp.any(jnp.isnan(jax.tree_util.tree_flatten(ckpt['params'])[0][0])))"

# Resume from earlier checkpoint
python scripts/train.py single --resume outputs/.../checkpoints/ckpt_best/...
```

### Evaluation Crash

```bash
# Run with smaller batch size
python scripts/eval_balance.py --num-envs 256 ...

# Run with noise disabled
python scripts/eval_balance.py --no-noise ...
```

### Disk Space

```bash
# Check disk usage
du -sh outputs/*

# Clean old checkpoints (keep only best and final)
find outputs -name "ckpt_step_*" -type d -exec rm -rf {} +
```

---

## Timeline Estimate

| Phase | Duration | GPU Hours |
|-------|----------|-----------|
| Sanity run (1M) | 30 min | 0.5 |
| Medium check (10M) | 4 hours | 4 |
| Main training (50M × 3 seeds) | 60 hours | 60 |
| Pure PPO baseline (50M × 1 seed) | 20 hours | 20 |
| All evaluations | 8 hours | 8 |
| Ablations (optional) | 60 hours | 60 |
| **Total (without ablations)** | **92 hours** | **92.5** |
| **Total (with ablations)** | **152 hours** | **152.5** |

**Cost estimate (A100 @ $2/hour):** $185 (without ablations), $305 (with ablations)

---

## Acceptance Criteria

Before filling paper results:

- [ ] All 3 seeds complete 50M steps without NaN
- [ ] Mean survival rate > 80% across seeds
- [ ] Height RMSE < 0.03m across seeds
- [ ] Push recovery > 100N across seeds
- [ ] All evaluation scenarios complete
- [ ] Residual analysis complete
- [ ] LaTeX tables generated
- [ ] Figures generated
- [ ] Pure PPO baseline complete for comparison

---

## Next Steps After Experiments

1. Fill TODO placeholders in paper/main.tex
2. Update docs/results_todo_checklist.md with DONE status
3. Generate final figures and tables
4. Review paper with complete results
5. Submit to conference/journal
