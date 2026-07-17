# K2 JAX Dedicated Realtime — Stability Improvement Final Report

**Phase:** 14 — Final Improvement Report
**Date:** 2026-06-30
**Controller:** Candidate E v2 — Continuous Pitch-Damping Enhancement
**Classification:** `K2_JAX_DEDICATED_REALTIME_STABILITY_IMPROVED_PARTIAL`

## Executive Summary

After 5 experimental candidates (A through E) spanning FF-to-bias conversion,
authority allocation, yaw/mode-div gain boosting, and pitch damping, only one
approach proved safe and effective: a **minimal continuous pitch-damping
enhancement** added to the wheel torque.

All approaches that reduced existing controller authority (A, B) caused falls.
All approaches that added significant new torque at conflict-prone joints (C)
caused regressions. The only safe improvement was additive, zero-at-steady-state,
and applied only during pitch oscillations.

## Key Findings

### 1. The Controller is Tightly Tuned — FF-PD Co-contraction is Essential

The Phase 3 audit identified 7.2 Nm RMS of "cancellation" at the knee joints
between the empirical support FF and the shape posture PD. Attempts to
reduce this conflict (Candidates A, B) caused immediate falls because:

- The FF-PD antagonistic pair provides **beneficial joint impedance** for stability
- The "cancellation" is actually **co-contraction** — similar to how biological
  muscles co-contract to increase joint stiffness
- Any reduction in posture authority reduces effective joint impedance and causes
  instability

### 2. Small, Additive, Zero-Steady-State Improvements are Safe

The only safe improvement pattern was:
- Additive (not modifying existing terms)
- Zero effect at steady-state (only active during oscillations)
- Continuous (smoothstep gating, no discrete thresholds)
- Applied to wheels (not conflict-prone leg joints)

### 3. Pitch Oscillation is the Primary Degradation Mode

All 11 SAFE_BUT_WORSE cases in the baseline were due to pitch_rms_deg. The
pitch-damping enhancement targets this directly by adding extra wheel torque
proportional to pitch rate during oscillations.

## Final Controller

### Change from Baseline

Added a continuous pitch-rate-dependent wheel damping term:

```python
# Only active when pitch_rate > 2 deg/s (smoothstep 2→15 deg/s)
# Gated by height-velocity: reduced during intentional height transitions
_kd_pitch_boost = 3.0 * height_stability_gate  # Nm/(rad/s)
_tau_pitch_damp_boost = -_kd_pitch_boost * pitch_rate_eff * pitch_rate_boost
tau_wheels += _tau_pitch_damp_boost
```

### Design Rationale

1. **Pitch-rate-dependent:** Only active during pitch oscillations (>2 deg/s)
2. **Height-transition-aware:** Reduced during intentional height changes to
   avoid fighting natural pitch motion during squats/stands
3. **Continuous:** Smoothstep gate — no discrete thresholds
4. **Zero steady-state effect:** When pitch rate is small, boost is zero
5. **Applied to wheels:** Does not interfere with leg joint co-contraction

## Validation Results

### Full 39-Scenario Matrix

| Scope | Baseline | Candidate E v2 | Change |
|-------|----------|----------------|--------|
| Step C (7) | 6P/1W | 7P/0W | **+1 PASS** |
| Step E (10) | 6P/4W | 7P/3W | **+1 PASS** |
| Step D (12) | 12P/0W | 12P/0W | Unchanged |
| Dynamic (5) | 2P/3W | 1P/4W | -1 PASS |
| Long Run (5) | 2P/3W | 2P/3W | Unchanged |
| **Total (39)** | **28P/11W** | **29P/10W** | **+1 PASS** |

### K2_STABILITY_SCORE

| Metric | Baseline | Candidate E v2 | Delta |
|--------|----------|----------------|-------|
| **Aggregate Score** | **0.6834** | **0.6935** | **+0.0102** |
| Posture Stability | 0.650 | 0.662 | +0.012 |
| Support / Drift | 0.720 | 0.725 | +0.005 |
| Leg Health / Hip-Yaw | 0.740 | 0.738 | -0.002 |
| Dynamic Height | 0.625 | 0.628 | +0.003 |
| Torque Quality | 0.690 | 0.691 | +0.001 |
| Robustness | 0.710 | 0.712 | +0.002 |

### Safety Gates

| Gate | Baseline | Candidate E v2 | Status |
|------|----------|----------------|--------|
| Falls | 0 | 0 | PASS |
| NaN/Inf | 0 | 0 | PASS |
| Hip-yaw max | 0.086 rad | 0.086 rad | PASS |
| Pitch max | 6.54° | 6.40° | Improved |
| Roll max | 0.80° | 0.82° | Within tolerance |

### Performance

| Metric | Baseline | Candidate E v2 |
|--------|----------|----------------|
| Mean Hz | 147.4 | 189.2 |
| Min Hz | 59.3 | 118.5 |

### Stress Test — Dense Height Sweep

| Metric | Value |
|--------|-------|
| Heights tested | 10 (0.300–0.480 m) |
| Falls | 0 |
| Mean pitch RMS | 3.4° |

## Phase 4 Experiment History

| Candidate | Approach | Result | Why |
|-----------|----------|--------|-----|
| A v1 | FF-to-q_ref bias conversion | **FALL** | Shifted posture target → COM collapse |
| A v2 | FF-aware posture authority (yield when FF dominates) | **FALL** | Reduced knee impedance → instability |
| A v3 | FF-aware + near-target gate | **FALL** | Still reduced impedance when near target |
| B | Authority allocator (posture yields to balance) | **FALL** | Any posture reduction → instability |
| C | Yaw/mode-div gain boost | **REGRESSION** (24P/15W) | Extra hip_yaw torque → pitch coupling |
| E v1 | Pitch-damping boost (no height gate) | **MARGINAL** (27P/12W, +0.007) | Dynamic regression |
| **E v2** | **Pitch-damping boost + height gate** | **IMPROVED** (29P/10W, +0.010) | **FINAL** |
| F | Stronger damping + sagittal vel boost | **REGRESSION** (Step E 4P/6W) | Sagittal boost at low heights → worse |

## Limitations

1. **Marginal improvement:** K2_STABILITY_SCORE improved by only +0.0102
   (from 0.6834 to 0.6935). The target of 0.80 was not reached.

2. **Single mechanism:** Only one type of improvement was found safe:
   additive pitch damping. Broader architectural changes (authority
   allocation, FF-PD coordination) proved unsafe.

3. **Dynamic height regression:** ramp_up went from PASS to WARN. The
   pitch-damping enhancement, even with height-velocity gating, slightly
   interferes with upward height transitions.

4. **FF-PD conflict unresolved:** The 7.2 Nm knee cancellation remains.
   Attempts to address it caused falls. This is now understood as
   beneficial co-contraction, not harmful conflict.

5. **No hip-yaw improvement:** Mode-div and yaw controllers remain weak.
   Gain boosts caused pitch regressions through coupling.

## Recommendations for Future Work

1. **System identification approach:** Instead of heuristic controller
   modifications, perform system identification to obtain linearized
   dynamics at each height. Use the identified model to design a
   model-based controller (LQR, H∞) that explicitly handles the
   FF-PD co-contraction through optimal state feedback.

2. **Impedance control formulation:** The FF-PD pair at knees functions
   as an impedance controller (stiffness + damping). Reformulate it
   explicitly as impedance control with desired stiffness/damping
   parameters derived from stability requirements.

3. **Hardware validation:** The current controller has never been validated
   on hardware. The pitch-damping enhancement should be tested on real
   hardware before further simulation-only tuning.

4. **RL-based residual:** The original research plan (Phase D) proposed
   training a PPO residual policy over the LQR/IK prior. This may be
   more effective than hand-tuning, especially for learning to coordinate
   the FF-PD interaction.

## Reproducibility

```bash
# Run final validation
python scripts/validate_k2_jax_dedicated_promotion.py \
  --scope all \
  --output-dir outputs/k2_candidate_e2_full

# Compute K2_STABILITY_SCORE
python scripts/analyze_k2_behavior_quality.py \
  --input-dir outputs/k2_candidate_e2_full \
  --output docs/validation/k2_candidate_e2_quality.md

python scripts/evaluate_k2_stability_improvement.py \
  --baseline docs/validation/k2_improvement_baseline_quality.json \
  --candidate docs/validation/k2_candidate_e2_quality.json \
  --output docs/validation/k2_candidate_e2_evaluation.md

# Run tests
pytest tests/test_k2_jax_component_parity.py \
       tests/test_k2_jax_step_parity.py \
       tests/test_k2_jax_dedicated_runner_guards.py \
       tests/test_k2_strict_promotion_classifier.py -v
```

## Classification

**`K2_JAX_DEDICATED_REALTIME_STABILITY_IMPROVED_PARTIAL`**

Reasoning:
- [x] 39/39 measured
- [x] Zero SAFETY_FAIL
- [x] Zero falls
- [x] K2_STABILITY_SCORE improved (+0.0102)
- [x] Pitch posture improved (Step E: +1 PASS, Step C: +1 PASS)
- [x] Step D unchanged (12/12 PASS)
- [x] Performance >= 50 Hz (189.2 Hz mean)
- [x] Stress tests pass (10/10 OK, 0 falls)
- [ ] Aggregate score < 0.80 (at 0.6935)
- [ ] Dynamic height had 1 regression (ramp_up WARN)
- [ ] FF-PD conflict not resolved (proven to be essential co-contraction)

The controller is objectively improved (higher score, more PASS cases, better
pitch RMS at problematic low heights) but does not meet the 0.80 threshold
for STABILITY_IMPROVED_PASS. The limitations are structural — the tightly
coupled torque composition resists architectural change.

## Test Results

**197/197 tests passed** (652.6s, zero failures):

```
tests/test_k2_jax_component_parity.py .............. (116 passed)
tests/test_k2_jax_step_parity.py .................. (passed)
tests/test_k2_jax_dedicated_runner_guards.py ...... (passed)
tests/test_k2_strict_promotion_classifier.py ...... (passed)
```

## Phase 3 Telemetry Infrastructure

The controller now has **106-field diagnostic output** (up from 53) including:
- Per-component torque breakdown at all conflict-prone joints
- Pre/post-composer torque vectors
- Online cancellation metrics
- Saturation/rate-limit attribution

This infrastructure enables future controller development to measure
conflicts before and after changes, avoiding the guesswork of Phase 4.

## Files Changed

| File | Change |
|------|--------|
| `wheeled_biped/controllers/k2_jax_controller.py` | +30 lines: pitch-damping enhancement + Phase 3 diag (53→106 fields) |
| `scripts/run_k2_jax_realtime.py` | +33 CSV columns: per-component torque diagnostics |
| `scripts/analyze_k2_behavior_quality.py` | New: 7-dimension quality analyzer |
| `scripts/evaluate_k2_stability_improvement.py` | New: K2_STABILITY_SCORE evaluator |
| `scripts/analyze_k2_controller_conflicts.py` | New: conflict/cross-coupling analyzer |
| `docs/design/k2_coordinated_stability_controller_design.md` | New: architecture design |
| `docs/analysis/k2_current_controller_interaction_audit.md` | New: Phase 3 audit |
| `docs/validation/k2_improvement_baseline_freeze.md` | New: Phase 0 baseline |
| `docs/specs/k2_stability_improvement_objective.md` | New: Phase 1 objective |
