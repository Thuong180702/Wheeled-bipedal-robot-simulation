# Phase B.6: Stronger Classical Prior Evaluation

**Date:** 2026-05-07  
**Status:** Implementation complete, evaluation pending  
**Goal:** Build the strongest practical classical prior before residual PPO training

## Motivation

Phase B.5 evaluation showed that all classical prior variants (geometric_lqr_ik, pitch_bias_lqr_ik, com_feedback_lqr_ik) failed as standalone controllers with 85-100% fall rates across the height range. The root cause was the fixed CoM height assumption in the TWIP model.

Phase B.6 aims to build a stronger classical prior by:
1. Using a 6D LQR state vector that includes forward velocity/position and CoM error
2. Height-scheduled gain interpolation
3. Wheel command filtering for smoother control
4. Integrating CoM feedback directly into the LQR formulation (not as additive correction)

**Decision criterion:** Must beat geometric_lqr_ik by 20% in survival time, pitch RMS, or fall rate by 10 percentage points to justify use as the main residual prior.

## Implementation

### Controller: height_scheduled_dynamic_lqr_ik

**Config:** `configs/controllers/height_scheduled_dynamic_lqr.yaml`

**Key features:**
- **6D LQR state vector:** `[pitch_error, pitch_rate, fwd_vel, fwd_pos, com_y_error, com_y_error_rate]`
  - Replaces the 4D state `[pitch, pitch_rate, wheel_pos, wheel_vel]` from geometric_lqr_ik
  - Forward velocity/position replace wheel states for better sagittal dynamics
  - CoM error states integrated into LQR (not additive correction)

- **Height-scheduled gains:** Linear interpolation across 7 height points (0.40-0.70 m)
  - `k_pitch`, `k_pitch_rate`, `k_fwd_vel`, `k_fwd_pos`, `k_com`, `k_com_rate`
  - Gains increase at lower heights (more unstable)

- **Wheel command filtering:**
  - Exponential smoothing: `cmd_filtered = alpha * cmd_prev + (1-alpha) * cmd_raw`
  - Max delta constraint: limits change per step to avoid abrupt commands
  - Default: `alpha=0.7`, `max_delta=2.0 rad/s`

- **CoM feedback:** Uses simulator CoM oracle (MuJoCo `subtree_com`)
  - **Hardware deployment note:** Requires CoM estimator (kinematic or sensor-based)
  - Not hardware-ready without CoM estimation

### Code changes

**Files modified:**
- `wheeled_biped/controllers/lqr_ik_prior.py` (lines 103-110, 112-163, 295-327, 474-625, 677-689)
  - Added wheel filter config fields to `LQRIKConfig`
  - Updated config loading to detect controller type from metadata
  - Implemented 6D state vector construction in height-scheduled mode
  - Added wheel command filtering with exponential smoothing and max delta
  - Updated `reset()` to clear filter state

**Files created:**
- `configs/controllers/height_scheduled_dynamic_lqr.yaml` (180 lines)
- `scripts/tune_stronger_classical_prior.py` (tuning script)
- `scripts/eval_stronger_classical_prior.py` (evaluation script)
- `tests/test_lqr_ik_prior.py` (added `TestHeightScheduledDynamicLQR` class)

### Tuning and evaluation scripts

**Tuning:** `scripts/tune_stronger_classical_prior.py`
- Grid search over LQR gains and wheel filter parameters
- Coarse search (samples every 2nd value) to reduce compute
- Evaluates at nominal height (0.55 m) for speed
- Outputs tuned config to `outputs/phase_b6_tuning/tuned_config.yaml`

**Evaluation:** `scripts/eval_stronger_classical_prior.py`
- Compares height_scheduled_dynamic_lqr_ik vs geometric_lqr_ik baseline
- Evaluates across all 7 height points (0.40-0.70 m)
- Computes aggregate metrics and improvement percentages
- Decision logic: ADOPT if meets any criterion, REJECT otherwise
- Outputs comparison to `outputs/phase_b6_eval/phase_b6_comparison.json`

## Expected outcomes

### If ADOPT (meets 20% improvement threshold):
1. Update `configs/training/balance_residual.yaml`:
   ```yaml
   prior_config: configs/controllers/height_scheduled_dynamic_lqr.yaml
   ```
2. Update `configs/training/balance_residual_robust.yaml` (same change)
3. Proceed to Phase D: residual PPO training with stronger prior

### If REJECT (does not meet threshold):
1. Keep `prior_config: configs/controllers/gain_scheduled_lqr.yaml` (geometric_lqr_ik)
2. Document findings: even with 6D state, CoM feedback, and filtering, classical prior remains limited
3. Proceed to Phase D with existing prior
4. Emphasize in paper that residual learning is necessary even with sophisticated priors

## Known limitations

1. **Simulator dependency:** Uses MuJoCo `subtree_com` oracle
   - Hardware deployment requires CoM estimator
   - Kinematic CoM from body masses is feasible but adds estimation error

2. **Height-scheduled gains:** Require per-height tuning
   - Current gains are conservative initial estimates
   - Optimal gains may vary with robot mass/inertia

3. **Wheel filtering trade-off:** Smoothness vs responsiveness
   - Higher alpha → smoother but slower response
   - Lower alpha → faster but more oscillatory

4. **Still a limited prior:** Even with improvements, standalone performance expected to be poor
   - Residual PPO still necessary for robust balance
   - Prior provides structure, not standalone solution

## Paper implications

### If ADOPT:
- Stronger baseline makes residual learning contribution clearer
- Can claim "even with sophisticated classical prior (6D LQR + CoM feedback + filtering), residual learning provides X% improvement"
- Justifies residual approach over pure classical control

### If REJECT:
- Demonstrates that classical priors remain limited even with enhancements
- Strengthens motivation for residual learning
- Shows that fixed-model assumptions (TWIP, linearization) are fundamental bottleneck

### Either outcome:
- Phase B.6 provides quantitative evidence for prior limitations
- Establishes that residual learning is necessary, not just convenient
- Supports claim that prior is "limited structured action prior" not "near-optimal baseline"

## Next steps

1. **Run tuning:** `python scripts/tune_stronger_classical_prior.py --episodes 20`
2. **Run evaluation:** `python scripts/eval_stronger_classical_prior.py --episodes 20`
3. **Make decision:** Based on evaluation results (ADOPT or REJECT)
4. **Update configs:** If ADOPT, update balance_residual.yaml
5. **Proceed to Phase D:** Residual PPO training (1M+ steps, 3 seeds)

## Testing

Added comprehensive tests in `tests/test_lqr_ik_prior.py`:
- Config loading and metadata validation
- 6D state vector construction
- Height-scheduled gain interpolation
- Wheel command filtering (exponential smoothing, max delta)
- CoM feedback integration
- Filter state reset
- Comparison with geometric baseline

Run tests:
```bash
pytest tests/test_lqr_ik_prior.py::TestHeightScheduledDynamicLQR -v
```

## References

- Phase B.4 report: `paper/notes/phase_b4_lqr_ik_prior_validation.md`
- Phase B.5 report: `paper/notes/phase_b5_classical_prior_variants.md`
- TWIP model: Two-Wheeled Inverted Pendulum with variable CoM height
- LQR: Linear Quadratic Regulator with height-dependent linearization
