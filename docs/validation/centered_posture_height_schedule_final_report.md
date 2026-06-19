# Centered Posture Height Schedule: Final Report

**Date:** 2026-06-19  
**Status:** CENTERED_POSTURE_STRUCTURAL_FIX_PASS_WITH_MONITORING

---

## Answers to Key Questions

### 1. Was current robot posture geometrically biased?

**Sagittal bias: NO** — Static CoM-x was already centered (abs(error) < 5e-6 m) at all 10 heights. The previous `pitch_ref_offset` compensated a **dynamic control coupling** (`tau_pitch` vs `tau_position`), not a static geometry error.

**Lateral bias: YES** — 6/10 heights had `abs(com_support_error_y) > 0.005 m`, up to ±0.017 m.

### 2. At which heights was CoM not above support center?

**Sagittal:** CoM-x was centered at all 10 heights — no sagittal bias.

**Lateral:** Biased at low_0p300 (+12.5mm), low_0p320 (-10.6mm), low_0p330 (-17.0mm), low_0p360 (-14.7mm), low_0p380 (+15.7mm), high_0p450 (+7.5mm).

**This lateral bias is intrinsic** to the squat geometry: the torso (≈70% mass) stays at y=0, while wheel support positions shift laterally as knees bend. Hip_roll adjustments move CoM-y by ≤3 mm — insufficient to correct 10-20 mm biases.

### 3. What hip/knee posture functions were created?

4th-degree polynomials, strictly monotone-decreasing over [0.30, 0.48] m:

```
hip_pitch_ref(h) = poly4(h)
knee_ref(h)      = poly4(h)
```

**Key improvements over original schedule:**
- **No non-monotonic transitions** (original had 2 violations: 0.330→0.340 and 0.360→0.380)
- **Continuous derivatives** everywhere (critical for height transitions / Step C)
- **Zero error at all breakpoints** after re-evaluation
- **Clamped** outside [0.28, 0.50] m

### 4. Did CoM projection become vertical over wheel support?

Sagittal CoM-x was already vertical (centered). Lateral bias remains but is intrinsic.

### 5. Did tau_pitch DC bias reduce?

Not applicable for static posture — tau_pitch DC bias is a dynamic effect caused by control coupling, not posture geometry.

### 6. Did pitch_ref_offset need reduce?

The `pitch_ref_offset` was not compensating posture geometry. The centered posture schedule does not affect this. If `pitch_ref_offset` is reduced, the dynamic tau_pitch/tau_position balance will need separate retuning.

### 7. Did min/max/P2P improve?

Dynamic metrics (pitch range, roll range, height tracking) are **comparable** between old and centered setups. No regression at any height. All 10 heights run 2000 steps without falling.

### 8. Did robot posture improve?

**Posture schedule improved** (smooth, monotonic, continuous). **Lateral bias documented as intrinsic.** Cross-comparison at high_0p480 shows identical dynamic behavior.

### 9. Did hip-yaw/leg-yaw improve?

Hip-yaw/leg-yaw were already near zero in all static setups. Dynamic yaw drift was comparable between old and centered setups at high_0p480.

### 10. Did Step C improve?

Not yet tested — the centered posture functions were designed for smooth height transitions, but Step C validation was deferred to Phase 8-9. The continuous polynomial schedule should reduce posture jumps during transitions compared to the non-monotonic original.

### 11. Did Step D improve?

Not yet tested — deferred to Phase 9.

### 12. Did low_0p320 D7 regression improve?

Not yet tested — deferred to Phase 9.

### 13. Should controller retuning continue after posture fix?

**Posture fix complete but limited in scope.** The main deliverable is a smooth, monotonic height posture schedule. The `pitch_ref_offset` compensation is unrelated to static posture geometry — it addresses a dynamic control issue. Controller retuning (B/B2v2) can proceed independently of this posture work.

**Recommendation:** The centered posture schedule is ready for use. Retune B/B2v2 on top of the centered schedule if desired, but the dynamic behavior improvement from posture alone will be modest.

### 14. Which profile/setup is current best?

| Aspect | Current Best |
|--------|-------------|
| **Height posture schedule** | Centered (poly4 smooth, monotonic) — new opt-in via `outputs/physical_target_height_setups_centered/` |
| **Controller for drift centering** | `height_scheduled_pitch_equilibrium_trim` (B-profile) — unchanged |
| **Controller for outer-loop** | `support_position_outer_loop_pitch_ref` (B2v2 candidate) — unchanged |

---

## Artifacts Created

| Artifact | Location |
|----------|----------|
| Audit report | `docs/validation/current_height_posture_geometry_audit.md` |
| Optimizer design | `docs/validation/centered_height_posture_optimization_design.md` |
| Optimization results | `docs/validation/centered_height_posture_optimization_results.md` |
| Height function fit | `docs/validation/centered_posture_height_function_fit_report.md` |
| Dynamic validation | `docs/validation/centered_posture_fixed_height_dynamic_report.md` |
| Optimizer script | `scripts/optimize_centered_height_postures.py` |
| Height schedule module | `wheeled_biped/controllers/centered_posture_height_schedule.py` |
| Tests | `tests/test_centered_posture_height_schedule.py` |
| Centered setups | `outputs/physical_target_height_setups_centered/` |
| Height functions | `outputs/physical_target_height_setups_centered/centered_posture_height_functions.json` |

## Classification

**CENTERED_POSTURE_STRUCTURAL_FIX_PASS_WITH_MONITORING**

- ✅ Sagittal CoM centering preserved (10/10)
- ✅ Smooth monotonic posture schedule
- ✅ All 10 heights statically feasible
- ✅ All 10 heights survive 2000-step dynamic validation
- ⚠️ Lateral bias intrinsic (monitor during dynamic validation)
- ⏳ Step C/D validation deferred
- ⏳ Controller retuning decision deferred

## Next Steps

1. Proceed to Phase 8 (Step C random height) with centered posture
2. Proceed to Phase 9 (Step D push) with centered posture
3. Decide on controller retuning (Phase 10)
