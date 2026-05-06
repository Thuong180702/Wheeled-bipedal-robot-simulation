# Results TODO Checklist

**Date:** 2026-05-06  
**Purpose:** Track every result needed for paper/main.tex  
**Status:** Training configs ready, awaiting experiments

**Training Configs:** READY
- `configs/training/balance_residual.yaml` — main residual training config
- `configs/training/balance_residual_robust.yaml` — future push recovery fine-tuning
- Both configs registered in `scripts/train.py` stage mapping

---

## A. LQR/IK Prior Baseline Results

### A.1. Fixed-Height Baseline Table

**Command:**
```bash
python scripts/eval_classical_priors.py \
  --variants geometric_lqr_ik \
  --scenarios fixed_height_sweep \
  --episodes 20 \
  --output-dir outputs/lqr_ik_baseline
```

**Output:** `outputs/lqr_ik_baseline/fixed_height_sweep_results.csv`  
**Fills:** Table IV — Nominal LQR/IK Prior Verification  
**Status:** DONE (Phase B.5 completed, see docs/phase_b5_classical_prior_variants_report.md)

### A.2. Classical Prior Variants Table

**Command:** Same as A.1  
**Output:** `outputs/lqr_ik_baseline/controller_comparison.csv`  
**Fills:** Table IV — Nominal LQR/IK Prior Verification  
**Status:** DONE (Phase B.5 completed)

---

## B. Residual PPO Training Results

### B.1. Training Curves (3 seeds)

**Command:**
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

**Output:**
- `outputs/residual_main_50M/seed42/train.log`
- `outputs/residual_main_50M/seed113/train.log`
- `outputs/residual_main_50M/seed999/train.log`

**Fills:** Figure 4 — Training curves over 3 seeds  
**Status:** TODO

### B.2. Curriculum Progression

**Command:** Extracted from training logs above  
**Output:** `outputs/residual_main_50M/seed*/curriculum_progress.csv`  
**Fills:** Figure 4 — Training curves (curriculum subplot)  
**Status:** TODO

### B.3. Residual Metrics During Training

**Command:** Extracted from training logs above  
**Output:** `outputs/residual_main_50M/seed*/residual_metrics.csv`  
**Fills:** Figure 4 — Training curves (residual RMS subplot)  
**Status:** TODO

---

## C. Random-Height Balance Results

### C.1. Residual PPO Evaluation (3 seeds)

**Command:**
```bash
# For each seed
for SEED in 42 113 999; do
  CKPT=outputs/residual_main_50M/seed${SEED}/checkpoints/final
  python scripts/eval_balance.py \
    --checkpoint $CKPT \
    --stage balance_residual \
    --scenarios random_height \
    --num-episodes 100 \
    --output-dir $CKPT/eval/random_height
done
```

**Output:**
- `outputs/residual_main_50M/seed42/eval/random_height/eval_results.json`
- `outputs/residual_main_50M/seed113/eval/random_height/eval_results.json`
- `outputs/residual_main_50M/seed999/eval/random_height/eval_results.json`

**Fills:** Table V — Main Random-Height Balance Results  
**Status:** TODO

### C.2. Pure PPO Baseline

**Command:**
```bash
CKPT=outputs/pure_ppo_baseline_50M/seed42/checkpoints/final
python scripts/eval_balance.py \
  --checkpoint $CKPT \
  --stage balance \
  --scenarios random_height \
  --num-episodes 100 \
  --output-dir $CKPT/eval/random_height
```

**Output:** `outputs/pure_ppo_baseline_50M/seed42/eval/random_height/eval_results.json`  
**Fills:** Table V — Main Random-Height Balance Results (baseline column)  
**Status:** TODO

### C.3. Random-Height Time Series Figure

**Command:**
```bash
python scripts/plot_results.py \
  --input outputs/paper_results \
  --output paper/figures \
  --plot-type height_tracking
```

**Output:** `paper/figures/height_tracking.pdf`  
**Fills:** Figure 5 — Random-height tracking time series  
**Status:** TODO

---

## D. Fixed-Height Sweep Results

### D.1. Residual PPO Fixed-Height Sweep (3 seeds)

**Command:**
```bash
# For each seed
for SEED in 42 113 999; do
  CKPT=outputs/residual_main_50M/seed${SEED}/checkpoints/final
  python scripts/eval_balance.py \
    --checkpoint $CKPT \
    --stage balance_residual \
    --scenarios fixed_height_sweep \
    --heights 0.70 0.65 0.60 0.55 0.50 0.45 0.40 \
    --num-episodes 50 \
    --output-dir $CKPT/eval/fixed_height
done
```

**Output:**
- `outputs/residual_main_50M/seed42/eval/fixed_height/eval_results.json`
- `outputs/residual_main_50M/seed113/eval/fixed_height/eval_results.json`
- `outputs/residual_main_50M/seed999/eval/fixed_height/eval_results.json`

**Fills:** Table VI — Fixed-Height Balance Sweep  
**Status:** TODO

### D.2. Pure PPO Fixed-Height Sweep

**Command:**
```bash
CKPT=outputs/pure_ppo_baseline_50M/seed42/checkpoints/final
python scripts/eval_balance.py \
  --checkpoint $CKPT \
  --stage balance \
  --scenarios fixed_height_sweep \
  --heights 0.70 0.65 0.60 0.55 0.50 0.45 0.40 \
  --num-episodes 50 \
  --output-dir $CKPT/eval/fixed_height
```

**Output:** `outputs/pure_ppo_baseline_50M/seed42/eval/fixed_height/eval_results.json`  
**Fills:** Table VI — Fixed-Height Balance Sweep (baseline column)  
**Status:** TODO

---

## E. Height Transition Results

### E.1. Residual PPO Height Transitions (3 seeds)

**Command:**
```bash
# For each seed
for SEED in 42 113 999; do
  CKPT=outputs/residual_main_50M/seed${SEED}/checkpoints/final
  python scripts/eval_balance.py \
    --checkpoint $CKPT \
    --stage balance_residual \
    --scenarios height_transition \
    --num-episodes 50 \
    --output-dir $CKPT/eval/height_transition
done
```

**Output:**
- `outputs/residual_main_50M/seed42/eval/height_transition/eval_results.json`
- `outputs/residual_main_50M/seed113/eval/height_transition/eval_results.json`
- `outputs/residual_main_50M/seed999/eval/height_transition/eval_results.json`

**Fills:** Table VII — Commanded Height-Transition Performance  
**Status:** TODO

### E.2. Height Transition Time Series Figure

**Command:**
```bash
python scripts/plot_results.py \
  --input outputs/paper_results \
  --output paper/figures \
  --plot-type height_transition
```

**Output:** `paper/figures/height_transition.pdf`  
**Fills:** Figure 6 — Commanded height transition time series  
**Status:** TODO

---

## F. Push Recovery Results

### F.1. Residual PPO Push Recovery (3 seeds)

**Command:**
```bash
# For each seed
for SEED in 42 113 999; do
  CKPT=outputs/residual_main_50M/seed${SEED}/checkpoints/final
  python scripts/eval_balance.py \
    --checkpoint $CKPT \
    --stage balance_residual \
    --scenarios push_recovery \
    --push-magnitudes 20 40 60 80 100 120 140 160 180 200 \
    --num-episodes 20 \
    --output-dir $CKPT/eval/push_recovery
done
```

**Output:**
- `outputs/residual_main_50M/seed42/eval/push_recovery/eval_results.json`
- `outputs/residual_main_50M/seed113/eval/push_recovery/eval_results.json`
- `outputs/residual_main_50M/seed999/eval/push_recovery/eval_results.json`

**Fills:** Table VIII — Push-Disturbance Recovery  
**Status:** TODO

### F.2. Pure PPO Push Recovery

**Command:**
```bash
CKPT=outputs/pure_ppo_baseline_50M/seed42/checkpoints/final
python scripts/eval_balance.py \
  --checkpoint $CKPT \
  --stage balance \
  --scenarios push_recovery \
  --push-magnitudes 20 40 60 80 100 120 140 160 180 200 \
  --num-episodes 20 \
  --output-dir $CKPT/eval/push_recovery
```

**Output:** `outputs/pure_ppo_baseline_50M/seed42/eval/push_recovery/eval_results.json`  
**Fills:** Table VIII — Push-Disturbance Recovery (baseline column)  
**Status:** TODO

### F.3. Push Recovery Time Series Figure

**Command:**
```bash
python scripts/plot_results.py \
  --input outputs/paper_results \
  --output paper/figures \
  --plot-type push_recovery
```

**Output:** `paper/figures/push_recovery.pdf`  
**Fills:** Figure 7 — Push recovery time series  
**Status:** TODO

### F.4. Push Magnitude Sweep Figure

**Command:**
```bash
python scripts/plot_results.py \
  --input outputs/paper_results \
  --output paper/figures \
  --plot-type push_sweep
```

**Output:** `paper/figures/push_sweep.pdf`  
**Fills:** Figure 8 — Push magnitude sweep  
**Status:** TODO

---

## G. Robustness Results

### G.1. Residual PPO Robustness Sweep (3 seeds)

**Command:**
```bash
# For each seed
for SEED in 42 113 999; do
  CKPT=outputs/residual_main_50M/seed${SEED}/checkpoints/final
  python scripts/eval_balance.py \
    --checkpoint $CKPT \
    --stage balance_residual \
    --scenarios robustness \
    --num-episodes 50 \
    --output-dir $CKPT/eval/robustness
done
```

**Output:**
- `outputs/residual_main_50M/seed42/eval/robustness/eval_results.json`
- `outputs/residual_main_50M/seed113/eval/robustness/eval_results.json`
- `outputs/residual_main_50M/seed999/eval/robustness/eval_results.json`

**Fills:** Table IX — Robustness to Model Uncertainty  
**Status:** TODO

### G.2. Robustness Sweep Figure

**Command:**
```bash
python scripts/plot_results.py \
  --input outputs/paper_results \
  --output paper/figures \
  --plot-type robustness
```

**Output:** `paper/figures/robustness_sweep.pdf`  
**Fills:** Figure 9 — Robustness sweep  
**Status:** TODO

---

## H. Residual Action Analysis

### H.1. Residual Action Metrics (3 seeds)

**Command:**
```bash
# For each seed
for SEED in 42 113 999; do
  python scripts/analyze_residual.py \
    --checkpoint outputs/residual_main_50M/seed${SEED}/checkpoints/final \
    --scenarios nominal random_height push_recovery \
    --num-episodes 50 \
    --output-dir outputs/residual_analysis/seed${SEED}
done
```

**Output:**
- `outputs/residual_analysis/seed42/residual_metrics.csv`
- `outputs/residual_analysis/seed113/residual_metrics.csv`
- `outputs/residual_analysis/seed999/residual_metrics.csv`

**Fills:** Table X — Residual Action Analysis  
**Status:** SCRIPT READY (scripts/analyze_residual.py implemented, awaits training)

### H.2. Residual Distribution Figure

**Command:**
```bash
python scripts/plot_results.py \
  --input outputs/residual_analysis \
  --output paper/figures \
  --plot-type residual_distribution
```

**Output:** `paper/figures/residual_distribution.pdf`  
**Fills:** Figure 10 — Residual action distribution  
**Status:** SCRIPT READY (scripts/plot_results.py implemented, awaits training)

---

## I. Ablation Study Results

### I.1. Ablation: No Base Action in Obs

**Command:**
```bash
python scripts/train.py single \
  --stage balance_residual_ablation_no_base_obs \
  --steps 50000000 \
  --seed 42 \
  --output-dir outputs/ablation/no_base_obs

# Evaluate
CKPT=outputs/ablation/no_base_obs/checkpoints/final
python scripts/eval_balance.py \
  --checkpoint $CKPT \
  --stage balance_residual_ablation_no_base_obs \
  --scenarios random_height \
  --num-episodes 100 \
  --output-dir $CKPT/eval/random_height
```

**Output:** `outputs/ablation/no_base_obs/eval/random_height/eval_results.json`  
**Fills:** Table XI — Ablation Study  
**Status:** TODO (requires ablation config creation first)

### I.2. Ablation: Different Residual Scale (0.5x)

**Command:**
```bash
python scripts/train.py single \
  --stage balance_residual_ablation_scale_0.5x \
  --steps 50000000 \
  --seed 42 \
  --output-dir outputs/ablation/scale_0.5x

# Evaluate
CKPT=outputs/ablation/scale_0.5x/checkpoints/final
python scripts/eval_balance.py \
  --checkpoint $CKPT \
  --stage balance_residual_ablation_scale_0.5x \
  --scenarios random_height \
  --num-episodes 100 \
  --output-dir $CKPT/eval/random_height
```

**Output:** `outputs/ablation/scale_0.5x/eval/random_height/eval_results.json`  
**Fills:** Table XI — Ablation Study  
**Status:** TODO (requires ablation config creation first)

### I.3. Ablation: No Residual Penalty

**Command:**
```bash
python scripts/train.py single \
  --stage balance_residual_ablation_no_penalty \
  --steps 50000000 \
  --seed 42 \
  --output-dir outputs/ablation/no_penalty

# Evaluate
CKPT=outputs/ablation/no_penalty/checkpoints/final
python scripts/eval_balance.py \
  --checkpoint $CKPT \
  --stage balance_residual_ablation_no_penalty \
  --scenarios random_height \
  --num-episodes 100 \
  --output-dir $CKPT/eval/random_height
```

**Output:** `outputs/ablation/no_penalty/eval/random_height/eval_results.json`  
**Fills:** Table XI — Ablation Study  
**Status:** TODO (requires ablation config creation first)

---

## J. Pure PPO Baseline Training

### J.1. Pure PPO Training (50M steps, 1 seed)

**Command:**
```bash
python scripts/train.py single \
  --stage balance \
  --steps 50000000 \
  --seed 42 \
  --num-envs 4096 \
  --output-dir outputs/pure_ppo_baseline_50M/seed42
```

**Output:** `outputs/pure_ppo_baseline_50M/seed42/train.log`  
**Fills:** Table XII — Pure PPO Baseline Discussion  
**Status:** TODO

### J.2. Pure PPO Nominal Evaluation

**Command:**
```bash
CKPT=outputs/pure_ppo_baseline_50M/seed42/checkpoints/final
python scripts/eval_balance.py \
  --checkpoint $CKPT \
  --stage balance \
  --scenarios nominal \
  --num-episodes 100 \
  --output-dir $CKPT/eval/nominal
```

**Output:** `outputs/pure_ppo_baseline_50M/seed42/eval/nominal/eval_results.json`  
**Fills:** Table XII — Pure PPO Baseline Discussion  
**Status:** TODO

---

## K. Aggregated Results for Paper

### K.1. Controller Comparison Table

**Command:**
```bash
python scripts/compare_controllers.py \
  --lqr-ik outputs/lqr_ik_baseline \
  --pure-ppo outputs/pure_ppo_baseline_50M/seed42/eval \
  --residual-ppo outputs/residual_main_50M/seed*/eval \
  --output-dir outputs/paper_results
```

**Output:** `outputs/paper_results/controller_comparison.csv`  
**Fills:** Table II — Controller Comparison  
**Status:** SCRIPT READY (scripts/compare_controllers.py implemented, awaits training)

### K.2. Mean ± Std Tables

**Command:** Same as K.1  
**Output:** `outputs/paper_results/mean_std_tables.csv`  
**Fills:** Tables V-XI (mean ± std across 3 seeds)  
**Status:** SCRIPT READY (scripts/compare_controllers.py implemented, awaits training)

### K.3. Scenario Breakdown

**Command:** Same as K.1  
**Output:** `outputs/paper_results/scenario_breakdown.csv`  
**Fills:** Tables V-XI (per-scenario metrics)  
**Status:** SCRIPT READY (scripts/compare_controllers.py implemented, awaits training)

---

## L. LaTeX Table Export

### L.1. Export All Tables

**Command:**
```bash
python scripts/export_results.py \
  --input outputs/paper_results \
  --format latex \
  --output paper/tables
```

**Output:**
- `paper/tables/table_random_height.tex`
- `paper/tables/table_fixed_height.tex`
- `paper/tables/table_height_transition.tex`
- `paper/tables/table_push_recovery.tex`
- `paper/tables/table_robustness.tex`
- `paper/tables/table_residual_analysis.tex`
- `paper/tables/table_ablation.tex`

**Fills:** Tables V-XI  
**Status:** TODO

---

## M. Figure Generation

### M.1. Generate All Figures

**Command:**
```bash
python scripts/plot_results.py \
  --input outputs/paper_results \
  --output paper/figures
```

**Output:**
- `paper/figures/training_curves.pdf`
- `paper/figures/height_tracking.pdf`
- `paper/figures/height_transition.pdf`
- `paper/figures/push_recovery.pdf`
- `paper/figures/push_sweep.pdf`
- `paper/figures/robustness_sweep.pdf`
- `paper/figures/residual_distribution.pdf`

**Fills:** Figures 4-10  
**Status:** SCRIPT READY (scripts/plot_results.py implemented, awaits training)

---

## Summary Status

| Category | Items | Status |
|----------|-------|--------|
| A. LQR/IK Prior | 2 | DONE |
| B. Training | 3 | TODO |
| C. Random-Height | 3 | TODO |
| D. Fixed-Height | 2 | TODO |
| E. Height Transition | 2 | TODO |
| F. Push Recovery | 4 | TODO |
| G. Robustness | 2 | TODO |
| H. Residual Analysis | 2 | SCRIPTS READY |
| I. Ablations | 3 | TODO (configs needed) |
| J. Pure PPO | 2 | TODO |
| K. Aggregation | 3 | SCRIPTS READY |
| L. LaTeX Export | 1 | TODO |
| M. Figures | 1 | SCRIPTS READY |
| **Total** | **30** | **2 DONE, 3 SCRIPTS READY, 25 TODO** |

**Note:** "SCRIPTS READY" means evaluation/analysis scripts are implemented and ready to use once training completes. Results still require actual training runs.

---

## Execution Order

1. **Sanity checks** (1M, 10M) — verify training works
2. **Main training** (50M × 3 seeds residual + 50M × 1 seed pure PPO) — ~80 GPU hours
3. **All evaluations** (C-G) — ~8 GPU hours
4. **Residual analysis** (H) — ~2 GPU hours
5. **Ablations** (I, optional) — ~60 GPU hours
6. **Aggregation** (K) — ~1 hour
7. **Export** (L, M) — ~1 hour

**Total time (without ablations):** ~92 GPU hours  
**Total time (with ablations):** ~152 GPU hours

---

## Acceptance Criteria

Before filling paper results, verify:

- [ ] All 3 residual PPO seeds complete 50M steps without NaN
- [ ] Mean survival rate > 80% across seeds
- [ ] Height RMSE < 0.03m across seeds
- [ ] Push recovery > 100N across seeds
- [ ] All evaluation scenarios complete (C-G)
- [ ] Residual analysis complete (H)
- [ ] LaTeX tables generated (L)
- [ ] Figures generated (M)
- [ ] Pure PPO baseline complete (J)
- [ ] Ablations complete (I, optional)

---

## Notes

- LQR/IK prior baseline (A) is already DONE from Phase B.5
- All other results require training runs to complete
- Ablation configs must be created before running ablation experiments
- Do not fill paper with fabricated numbers
- Leave TODO placeholders until experiments complete
- After experiments, run aggregation (K) → export (L, M) → fill paper
