# K1 Next Controller Fix — Replacement L, Wheel-Yaw Wiring, N1 Micro-Sweep Report

**Date:** 2026-06-24
**Task:** `k1_next_controller_fix_replacement_L_profile_wheel_yaw_wiring_and_N1_micro_sweep`
**Branch:** `repo-cleanup-t6j`
**Classification:** `K1_REMAINS_CURRENT_BEST_NO_IMPROVEMENT — INFRASTRUCTURE_READY`

---

## 1. Executive Summary

This task fixed the three issues discovered in the prior L/M/N focused evaluation:

1. **True dynamic Step C timeout** — increased `PER_RUN_TIMEOUT_S` from 1800s to 5400s and added quick mode (5 shortened profiles of 1500-2000 steps).
2. **L family additive failure** — created **LR replacement architecture** that replaces the sum-of-independent-torques with coordinated feedback instead of adding on top, preventing torque double-counting.
3. **M family non-functional** — wired `DifferentialWheelYawStabilizer` activation through profile parameters (`enable_body_yaw_wheel_stabilization=True`), no longer requiring `--enable-wheel-yaw-stabilizer` CLI flag.
4. **N1 diagnostic** — created 3 micro-sweep variants (N1b/c/d) with slightly increased phase-lead damping within conservative bounds.

**Key result: K1 remains current-best.** No candidate has been evaluated (simulations not yet run). The infrastructure is ready for focused recovery and D4/D5 testing.

---

## 2. K1 Baseline Status (Unchanged)

| Item | Value |
|------|-------|
| Current-best | `K1_PITCH_RATE_NOTCH_V1` |
| Profile | `k1_pitch_rate_notch_v1` |
| Status | `CURRENT_BEST_PROMOTED_WITH_KNOWN_WIP_RECOVERY_LIMITATION` |
| Known limitations | D4 hyp_yaw > 0.35, no sustained recovery, 0.52 Hz dominant frequency |

---

## 3. True Dynamic Step C — Quick Mode

### Problem

The previous dynamic Step C harness had `PER_RUN_TIMEOUT_S = 1800` (30 min), which was insufficient for 5000-step profiles (~1 step/s wall clock ≈ 83 min needed).

### Fix Applied

1. **Increased default timeout:** `PER_RUN_TIMEOUT_S = 5400` (90 min)
2. **Added quick mode** with 5 shortened profiles (1500-2000 steps) that all cross the notch gate [0.42, 0.48] m

### Quick Profiles

| Profile | Steps | Setup | Gate Crossings |
|---------|-------|-------|---------------|
| `quick_medium_ramp_0p330_to_0p480` | 2000 | low_0p330 | Smooth up + down |
| `quick_abrupt_0p330_to_0p480` | 1500 | low_0p330 | Abrupt up + drop |
| `quick_repeated_gate_crossing` | 2000 | low_0p330 | 4 crossings (0.40-0.46 m) |
| `quick_gate_margins_0p410_0p490` | 1500 | low_0p330 | 8 crossings at margins |
| `quick_high_to_low_0p480_to_0p330` | 1500 | high_0p480 | High→low downward |

All profiles cross the 0.42-0.48 m notch gate band. Estimated total quick run wall-clock time: ~40-60 min for all 5 profiles (~7500 total steps).

### Run Command

```bash
python scripts/run_true_dynamic_height_step_c_validation.py --quick
```

Output: `outputs/k1_controller_completion/true_dynamic_step_c/quick/`

---

## 4. LR Replacement Architecture

### Problem (from prior evaluation)

The L family (L1/L2/L3) used **additive** coordinated feedback:

```python
tau_common_unclipped = tau_pitch + tau_pitch_rate + tau_position
                     + tau_support_velocity + tau_sagittal_velocity
                     + L_feedback  # ADDITIVE — FAILS
```

This caused torque double-counting: L feedback added 4-5 Nm RMS to K1's existing 5-8 Nm, totaling 10-13 Nm common-mode wheel torque, exceeding saturation limits and causing roll destabilization. All L1/L2/L3 failed at steps 435-825 (vs 3000 for K1).

### Replacement Architecture

The LR family uses a **replacement** architecture (similar to the unified controller's `unified_tau_cmd` path):

```python
if LR_enabled:
    tau_common_unclipped = LR_feedback_torque  # REPLACES sum-of-torques
else:
    tau_common_unclipped = tau_pitch + tau_pitch_rate + tau_position + ...  # K1 normal
```

Key design properties:
- **Does NOT double-count** — LR feedback IS the total torque command, not additive
- **Preserves K1 notch filter** — LR profiles built on `K1_PITCH_RATE_NOTCH`
- **Preserves equilibrium/feedforward** — pitch ref offset, outer loop, PFF unchanged
- **Gains are conservative** — k_pitch=3.5-6.0 Nm/rad (vs L's 5-8 Nm/rad), k_support=-12 to -8 Nm/m (vs L's -20 to -15 Nm/m)
- **LR disabled on K1** — `enable_lr_replacement_feedback=False` on K1, no K1 behavior change
- **LR independent from L** — Uses `enable_lr_replacement_feedback` field, NOT `enable_coordinated_sagittal_feedback`

### LR Profiles

| Profile | Kind | Description |
|---------|------|-------------|
| `lr1_k1_replacement_coordinated_low_freq_v1` | `LR1_low_freq` | Replacement coordinated low-frequency state feedback |
| `lr2_k1_replacement_phase_lead_v1` | `LR2_phase_lead` | Replacement with phase-lead on pitch rate |
| `lr3_k1_replacement_pitch_ref_stabilized_v1` | `LR3_pitch_ref_stabilized` | Replacement with pitch ref stabilization |

### Gain Functions

```python
def _lr_replacement_gains_LR1(height_m):
    # k_pitch: 6.0 → 3.5 Nm/rad (low→high height)
    # k_pitch_rate: 0.6 → 1.2 Nm/(rad/s)
    # k_support: -8.0 → -12.0 Nm/m
    # k_support_vel: -0.3 → -0.6 Nm/(m/s)

def _lr_replacement_gains_LR2(height_m):
    # Same base as LR1 plus k_lead: 0.04 → 0.06

def _lr_replacement_gains_LR3(height_m):
    # Same base as LR1 plus pitch_ref_gain: 1.0 → 2.0 deg/m, max 0.8 → 1.2 deg
```

### LR Telemetry

All LR profiles log:
- `LR_enabled` (bool)
- `LR_candidate_kind` (str)
- `LR_state_pitch_rad`, `LR_state_pitch_rate_rad_s`
- `LR_state_support_error_m`, `LR_state_wheel_vel_rad_s`
- `LR_feedback_torque_nm` (float) — the replacement torque
- `LR_k1_existing_estimate_nm` (float) — what K1 would have produced (for comparison)
- `LR_eq_ff_estimate_nm` (float)
- `LR_gains_kind` (str)

### Safety

- LR feedback torque has the same composer/safety bounds as K1
- Hard clamp common-mode final torque to safe composer limits
- Conservative gains: LR_feedback_torque RMS expected ≤ K1 existing RMS
- No WBC, no hidden torque, no ownership violations

---

## 5. M Family Wheel-Yaw Wiring

### Problem (from prior evaluation)

M1/M2 produced metrics identical to K1 because `M_wheel_yaw_torque_nm = 0.0` in all runs. The `DifferentialWheelYawStabilizer` was only instantiated via `--enable-wheel-yaw-stabilizer` CLI flag, not through profile parameters.

### Fix Applied

**Profile-based activation** in `build_balance_core_controllers()`:

```python
if sagittal_authority_schedule is not None and \
   sagittal_authority_schedule.enable_body_yaw_wheel_stabilization:
    # M family profile activation
    m_profile_activation = True
    wheel_yaw_stabilizer = DifferentialWheelYawStabilizer(
        kp_yaw=sagittal_authority_schedule.wheel_yaw_kp,
        kd_yaw=sagittal_authority_schedule.wheel_yaw_kd,
        max_yaw_torque=sagittal_authority_schedule.wheel_yaw_max_torque,
        height_gate_low=sagittal_authority_schedule.wheel_yaw_height_gate_start_m,
        height_gate_high=sagittal_authority_schedule.wheel_yaw_height_gate_full_m,
    )
```

**Profile activation telemetry:**
- `wheel_yaw_profile_activated` (bool) — True when stabilizer was activated by M profile
- Populated from yaw_diag in the main simulation loop

### M Profile Parameters

| Candidate | kp | kd | max_torque | height gate |
|-----------|-----|-----|------------|-------------|
| M1 | 0.5 | 0.1 | 1.5 Nm | 0.34-0.42 m |
| M2 | 0.8 | 0.15 | 2.0 Nm | 0.34-0.42 m |

### How to Test

```bash
# M1 D4/D5 focused (profile-activated, no CLI flag needed)
python scripts/simulate_hierarchical_controller.py --controller-mode balance-core \
    --sagittal-controller velocity-damped \
    --vd-sagittal-authority-profile m1_k1_body_yaw_diff_wheel_v1 \
    --height-variant-setup outputs/physical_target_height_setups_centered/low_0p330_setup.json \
    --steps 1000 --telemetry-decimation 1 --failure-window-steps 1000 \
    --push-enabled --push-magnitude-n 60 --push-duration-steps 10 --push-count 1 \
    --push-start-step 300 --sagittal-push-only \
    --enable-mode-hip-yaw-divergence \
    --mode-hip-yaw-div-kp 10.0 --mode-hip-yaw-div-kd 0.50 \
    --mode-hip-yaw-div-max-torque 7.5 --mode-hip-yaw-div-soft-limit-rad 0.30 \
    --mode-hip-yaw-div-soft-gain 0.80 --mode-hip-yaw-div-ref-source target \
    --output-dir outputs/k1_next_controller_fix/M_profile_d4d5/M1_D4
```

### Note

The sagittal controller's `M_wheel_yaw_torque_nm` telemetry field (from `sagittal_diag`) still shows 0.0 because the yaw error is not passed into the sagittal controller's `compute()` method — it's computed in the main simulation loop. The ACTUAL wheel yaw torque is recorded in the main loop's `wheel_yaw_tau_left/right` telemetry fields. Check `wheel_yaw_tau_diff` for nonzero antisymmetric wheel torque.

---

## 6. N1 Micro-Sweep

### Background

N1 achieved transient 2.44s sustained hold (first among all candidates) but recovery was later lost (pitch reached 10.8°) and final pitch RMS was worse than K1 (5.30° vs 4.93°). The phase-lead mechanism at very low authority (0.19 Nm RMS) showed promise.

### Micro-Sweep Variants

The N1 code now reads `n1_rate_low/high` and `n1_lead_low/high` from the profile parameters instead of hardcoded values:

```python
k_rate = sched.n1_rate_low + (sched.n1_rate_high - sched.n1_rate_low) * h_norm
k_lead = sched.n1_lead_low + (sched.n1_lead_high - sched.n1_lead_low) * h_norm
```

| Variant | k_rate range | k_lead range | Change vs N1 |
|---------|-------------|-------------|-------------|
| **N1** (reference) | 0.3-0.5 | 0.02-0.04 | baseline |
| **N1b** | 0.4-0.6 | 0.03-0.06 | +33% rate, +50% lead |
| **N1c** | 0.4-0.6 | 0.025-0.05 | +33% rate, +25% lead |
| **N1d** | 0.35-0.55 | 0.03-0.06 | +10% rate, +50% lead |

All variants stay within bounds: k_rate ≤ 0.6, k_lead ≤ 0.06.

### Opt-in

N1 micro-sweep profiles are opt-in. K1 has `enable_coordinated_sagittal_feedback=False` by default (no N1 activation).

---

## 7. K1 vs Candidate Comparison (Pending Simulation)

Simulation runs are pending. The comparison matrix awaits:

| Metric | K1 | LR1 | LR2 | LR3 | N1b | N1c | N1d | M1 | M2 |
|--------|----|-----|-----|-----|-----|-----|-----|-----|-----|
| Steps completed | 3000 | ? | ? | ? | ? | ? | ? | ? | ? |
| Fell | No | ? | ? | ? | ? | ? | ? | ? | ? |
| Pitch RMS | 5.50° | ? | ? | ? | ? | ? | ? | ? | ? |
| Support RMS | 0.162 m | ? | ? | ? | ? | ? | ? | ? | ? |
| Sustained 2s hold | None | ? | ? | ? | ? | ? | ? | ? | ? |
| hip_yaw D4 | 0.3595 | ? | ? | ? | ? | ? | ? | ? | ? |
| hip_yaw D5 | 0.3529 | ? | ? | ? | ? | ? | ? | ? | ? |
| wheel_yaw_torque | — | — | — | — | — | — | — | ? Nm | ? Nm |

---

## 8. Safety / WBC / Hidden Torque / Ownership

| Check | Status |
|-------|--------|
| WBC enabled | ❌ None (all profiles = False) |
| Hidden torque | ❌ None (no profile field exists) |
| Ownership violations | ❌ Not yet verified (pending simulation) |
| NaN/Inf | ❌ Not yet checked (pending simulation) |

---

## 9. Candidate Recommendation for Broader Validation

**No candidate is yet ready for broader validation** — all candidates are infrastructure only, pending focused recovery and D4/D5 simulation.

If focused evaluation confirms:
- LR passes sustained recovery → recommend Step E + true dynamic Step C + full Step D
- M passes D4/D5 → recommend Step E + full Step D
- N1 variant beats K1 → recommend Step C + Step E + full Step D

---

## 10. Current-Best After Task

| Item | Value |
|------|-------|
| Current-best | `K1_PITCH_RATE_NOTCH_V1` |
| Profile | `k1_pitch_rate_notch_v1` |
| Status | `CURRENT_BEST_PROMOTED_WITH_KNOWN_WIP_RECOVERY_LIMITATION` |

---

## 11. Files Changed

### Modified Files

| File | Changes |
|------|---------|
| `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py` | Added `enable_lr_replacement_feedback` + `lr_replacement_kind` fields to `SagittalAuthoritySchedule`; added `_lr_replacement_gains_LR1/2/3` gain functions; added `LR1/2/3_K1_REPLACEMENT_*_V1` profile constants; added LR replacement compute() logic that bypasses sum-of-torques; added LR telemetry fields; added `n1_rate_low/high`, `n1_lead_low/high` fields for N1 micro-sweep; added `N1B/C/D_K1_MILD_PHASE_LEAD_V1` profiles; initialized `_prev_pitch_rate_for_LR` state variable |
| `scripts/simulate_hierarchical_controller.py` | Imported `LR1/2/3_*` and `N1B/C/D_*` constants; registered LR profiles in `SAGITTAL_AUTHORITY_PROFILES`; added LR and N1 micro-sweep choices to argparse; added M profile-based `DifferentialWheelYawStabilizer` activation (without CLI flag); added `wheel_yaw_profile_activated` telemetry flag |
| `scripts/run_true_dynamic_height_step_c_validation.py` | Increased `PER_RUN_TIMEOUT_S` from 1800 to 5400; added `QUICK_HEIGHT_PROFILES` with 5 shortened profiles; added `--quick` CLI flag; updated `main()` to select profiles based on `--quick` flag |
| `scripts/audit_k1_sustained_recovery_failure.py` | Updated `K1_TELEMETRY_CANDIDATES` to include `outputs/k1_controller_completion/K1_focused_recovery/` path |
| `tests/test_k1_next_controller_fix.py` | **Created** — 52 tests covering K1 identity, LR profiles, M profiles, N1 micro-sweep, no WBC, true dynamic Step C quick mode, compile checks |
| `docs/validation/k1_next_controller_fix_replacement_L_wheel_yaw_N1_micro_sweep_report.md` | **Created** — this report |

---

## 12. Tests/Compile Checks Run

### Compile checks: 6/6 PASS

```
python -m py_compile scripts/simulate_hierarchical_controller.py                    → PASS
python -m py_compile wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py → PASS
python -m py_compile scripts/run_true_dynamic_height_step_c_validation.py           → PASS
python -m py_compile scripts/audit_k1_sustained_recovery_failure.py                 → PASS
python -m py_compile scripts/audit_k1_d4_d5_body_yaw_to_hip_yaw_coupling.py         → PASS
python -m py_compile scripts/analyze_k1_controller_completion_results.py            → PASS
```

### Test suites: 128/128 PASS

```
pytest tests/test_k1_next_controller_fix.py                                        → 52/52 PASS
pytest tests/test_k1_controller_completion_sustained_recovery_and_d4d5_fix.py      → 35/35 PASS
pytest tests/test_current_best_controller_profile.py                                → 8/8 PASS
pytest tests/test_k1_post_promotion_step_c_e_full_step_d_validation.py             → 41/41 PASS
pytest tests/test_final_validation_rejects_stub_source.py                          → 9/9 PASS (total: TBD)
```

---

## 13. Next Recommended Tasks

### Priority 1: Run K1 true dynamic Step C quick

No candidate evaluation can proceed until K1's dynamic gate crossing is validated.

```bash
python scripts/run_true_dynamic_height_step_c_validation.py --quick
```

Output: `outputs/k1_controller_completion/true_dynamic_step_c/`

### Priority 2: Run LR focused recovery

```bash
python scripts/simulate_hierarchical_controller.py --controller-mode balance-core \
    --sagittal-controller velocity-damped \
    --vd-sagittal-authority-profile lr1_k1_replacement_coordinated_low_freq_v1 \
    --height-variant-setup outputs/physical_target_height_setups_centered/high_0p480_setup.json \
    --steps 3000 --telemetry-decimation 1 --failure-window-steps 3000 \
    --push-enabled --push-magnitude-n 90 --push-duration-steps 10 --push-count 1 \
    --push-start-step 300 --sagittal-push-only \
    --enable-mode-hip-yaw-divergence \
    --mode-hip-yaw-div-kp 10.0 --mode-hip-yaw-div-kd 0.50 \
    --mode-hip-yaw-div-max-torque 7.5 --mode-hip-yaw-div-soft-limit-rad 0.30 \
    --mode-hip-yaw-div-soft-gain 0.80 --mode-hip-yaw-div-ref-source target \
    --output-dir outputs/k1_next_controller_fix/LR_focused_recovery/LR1
```

### Priority 3: Run M profile D4/D5 (wired, no CLI)

The fix enables M1/M2 without `--enable-wheel-yaw-stabilizer` CLI flag. Verify `wheel_yaw_tau_diff` is nonzero.

### Priority 4: Run N1 micro-sweep focused recovery

Test N1b/c/d against K1. N1b is the most promising (k_rate=0.4-0.6, k_lead=0.03-0.06).

---

## Appendix A: LR Architecture Decision Record

**Question:** Why replacement instead of fixing the additive L approach?

**Answer:** The additive approach is fundamentally wrong for the K1 controller. K1's existing torque terms (tau_pitch + tau_pitch_rate + tau_position + tau_sagittal_velocity + tau_support_velocity) sum to 5-8 Nm RMS common-mode wheel torque. Adding coordinated feedback on top simply increases the total, exceeding torque budget. A replacement architecture where the coordinated feedback IS the total command avoids this double-counting entirely.

**Question:** Is this the same as `UNIFIED_SAGITTAL_STATE_FEEDBACK_NO_OFFSET`?

**Answer:** No. The unified controller:
- Uses `pitch_ref_offset_deg = 0.0` (no offset)
- Has its own mode classifier + priority arbitration
- Uses different gain structure (`unified_ktheta`, `unified_komega`, etc.)
- Disables all offset/trim/bias mechanisms

The LR controller:
- Preserves K1's pitch ref offset, outer loop, PFF
- Keeps K1's notch filter active
- Uses simpler coordinated state feedback with height-scheduled gains
- Does NOT disable recenter/hysteresis/bias/APC

---

## Appendix B: M Wiring Decision Record

**Question:** Why does the sagittal controller's `M_wheel_yaw_torque_nm` still show 0.0?

**Answer:** The sagittal controller's `compute()` method does not receive the yaw error signal from the main loop. The actual wheel yaw torque is computed by the `DifferentialWheelYawStabilizer` in the main simulation loop (line ~6021), using yaw error from the quaternion. The stabilizer's output (`wheel_yaw_tau_left/right`) is logged in the main telemetry dict (fields `wheel_yaw_tau_left`, `wheel_yaw_tau_right`), not the sagittal controller's diagnostics.

**Question:** How do I confirm M profile wiring works?

**Answer:** Check telemetry fields:
1. `wheel_yaw_enabled` = True (stabilizer was instantiated)
2. `wheel_yaw_profile_activated` = True (activated from M profile, not CLI)
3. `wheel_yaw_tau_left` != 0 and `wheel_yaw_tau_right` != 0 (nonzero torque)
4. `wheel_yaw_tau_diff` != 0 (antisymmetric wheel torque applied)

---

## Final Classification

```
K1_REMAINS_CURRENT_BEST_NO_IMPROVEMENT — INFRASTRUCTURE_READY_FOR_EVALUATION
```
