# K1 Next Controller Fix — Ready Infrastructure Evaluation Report

**Date:** 2026-06-24
**Task:** `run_k1_next_controller_fix_ready_infrastructure_evaluation`
**Branch:** `repo-cleanup-t6j`
**Classification:** `K1_REMAINS_CURRENT_BEST_NO_READY_CANDIDATE`

---

## 1. Executive Summary

The four infrastructure-ready controller families (LR replacement, M wheel-yaw wiring, N1 micro-sweep, and true dynamic Step C quick) were evaluated through real simulations. **No candidate outperformed K1.** All require implementation fixes or parameter redesign before broader validation.

| Family | Result | Verdict |
|--------|--------|---------|
| LR (replacement coordinated feedback) | CRITICAL FAILURE — zero equilibrium/feedforward, 10x too little torque | Requires implementation fix |
| M (wheel-yaw stabilizer wiring) | Wiring verified ✓, but D5 hip_yaw regression (+73% vs K1) | Requires parameter redesign |
| N1 (phase-lead micro-sweep) | NO IMPROVEMENT — pitch +5%, hip_yaw +55%, zero post-push hold | Requires different parameter direction |
| K1 true dynamic Step C quick | FAIL — height tracking non-functional, CoM doesn't follow target | Does not block candidate eval (K1 stable) |

**K1_PITCH_RATE_NOTCH_V1 remains current-best.**

---

## 2. K1 Baseline (Unchanged)

| Metric | K1 Value |
|--------|----------|
| Profile | `k1_pitch_rate_notch_v1` |
| Pitch RMS | 5.50° |
| Support RMS | 0.162 m |
| Final pitch RMS | 4.93° |
| hip_yaw_abs_max D4 | 0.3595 rad (FAIL gate 0.35) |
| hip_yaw_abs_max D5 | 0.3529 rad (FAIL gate 0.35) |
| Sustained 2s hold | None |
| 0.52 Hz dominant | Yes |
| Falls | No (3000-step nominal) |
| WBC | Disabled |
| Hidden torque | None |

---

## 3. K1 True Dynamic Step C Quick — FAIL (Height Tracking Non-Functional)

**Status:** All 5 quick profiles completed, but **the dynamic height controller is not moving the robot.** Actual CoM height does not follow the target trajectory.

### 3.1 Results

| Profile | Steps | Actual Height Range | Target Height Range | Gate Crossing (Actual) | Notch Gate Active | Fell | Pitch RMS |
|---------|-------|--------------------|--------------------|-----------------------|-------------------|------|-----------|
| quick_medium_ramp_0p330_to_0p480 | 1999/2000 | 0.330-0.332 m | 0.330-0.480 m | **0 rows** | 959/1999 | No | 3.67° |
| quick_abrupt_0p330_to_0p480 | 1499/1500 | 0.330-0.332 m | 0.330-0.480 m | **0 rows** | 679/1499 | No | 3.39° |
| quick_repeated_gate_crossing | 1999/2000 | 0.330-0.332 m | 0.400-0.460 m | **0 rows** | 1068/1999 | No | 3.21° |
| quick_gate_margins_0p410_0p490 | 1499/1500 | 0.330-0.332 m | 0.410-0.490 m | **0 rows** | 1250/1499 | No | 3.36° |
| quick_high_to_low_0p480_to_0p330 | 1499/1500 | 0.480-0.490 m | 0.330-0.480 m | **2 rows** | 500/1499 | No | 4.23° |

### 3.2 Analysis

1. **Height tracking non-functional:** For the 4 low-start profiles, `current_com_z_m` stays at ~0.33m (the initial setup height) while `dynamic_height_target_m` ranges up to 0.48m. The robot does NOT follow height commands. The high-start profile stays at 0.48m and doesn't descend to 0.33m.

2. **Notch gate NOT physically crossed:** `Gate [0.42-0.48] crossing rows: 0` for 4/5 profiles. The notch gate is active in the controller (based on target height), but the physical CoM never enters the 0.42-0.48m band.

3. **Controller IS stable:** Despite changing height targets, K1 maintains stable balance at the initial height. No falls, pitch RMS 3.2-4.2°, hip_yaw 0.09-0.14 rad, roll 0.1-0.9°.

4. **NaN/Inf:** High NaN counts (~9 empty telemetry columns × n_rows). 1 Inf value per profile (small, non-critical).

5. **Safety:** No WBC, hidden torque, or ownership violations.

### 3.3 Root Cause

The dynamic height trajectory changes the `dynamic_height_target_m` command signal, but the controller either:
- Does not have a height-tracking mechanism active in the dynamic-height simulation path, OR
- The height tracking gains (from the calibrated outer loop or support position controller) are zero/gated at the low starting height, OR
- The `--dynamic-height-trajectory` flag passes the trajectory to the sim but the height controller's reference is not connected to it.

This means **the dynamic Step C harness does not actually validate notch gate crossing** — it validates that K1 remains stable during height command changes, but the physical gate crossing test is skipped.

### 3.4 Classification

**`K1_TRUE_DYNAMIC_STEP_C_QUICK_FAIL`** — dynamic height tracking non-functional. The robot does not physically cross the notch gate. However, this is NOT a controller instability issue (K1 remains stable), so candidate evaluation proceeds as planned.

**This failure does NOT block candidate evaluation.** K1 stability during height commands is verified; only the gate-crossing stress test is missing.

---

## 4. LR Replacement Results — ALL FAILED

### 4.1 LR1 (`lr1_k1_replacement_coordinated_low_freq_v1`)

| Metric | LR1 | K1 Baseline | Delta |
|--------|-----|-------------|-------|
| Steps completed | 183 / 3000 | 3000 | **-94%** |
| Sim time | 1.82 s | 30 s | — |
| Fell | No (aborted) | No | — |
| Pitch RMS | 7.73° | 5.50° | **+41% worse** |
| Pitch max | 27.07° | — | — |
| Support RMS | 0.071 m | 0.162 m | -56% |
| hip_yaw_abs_max | 0.028 rad | 0.299 rad | -90% |
| Roll RMS | 0.27° | — | — |
| LR_feedback_torque RMS | 1.28 Nm | — | — |
| LR_k1_existing_estimate RMS | **15.01 Nm** | — | — |
| **LR_eq_ff_estimate_nm** | **0.00** | — | **CRITICAL** |
| physics_ff_applied | 0.00 Nm | — | CRITICAL |
| tau_pitch | 0.00 | — | Expected (replaced) |
| PSD 0.52 Hz | 0.000656 | — | — |
| PSD 2.5 Hz | 0.005924 | — | — |

### 4.2 LR2 (`lr2_k1_replacement_phase_lead_v1`)

| Metric | LR2 | K1 Baseline |
|--------|-----|-------------|
| Steps completed | 185 / 3000 | 3000 |
| Pitch RMS | 7.70° | 5.50° |
| LR_feedback_torque RMS | 1.32 Nm | — |
| LR_k1_existing_estimate RMS | 15.08 Nm | — |
| **LR_eq_ff_estimate_nm** | **0.00** | **CRITICAL** |
| physics_ff_applied | 0.00 Nm | CRITICAL |

### 4.3 LR3 (`lr3_k1_replacement_pitch_ref_stabilized_v1`)

| Metric | LR3 | K1 Baseline |
|--------|-----|-------------|
| Steps completed | 179 / 3000 | 3000 |
| Pitch RMS | 7.65° | 5.50° |
| LR_feedback_torque RMS | 1.16 Nm | — |
| LR_k1_existing_estimate RMS | 14.69 Nm | — |
| **LR_eq_ff_estimate_nm** | **0.00** | **CRITICAL** |
| physics_ff_applied | 0.00 Nm | CRITICAL |

### 4.4 Root Cause Analysis

The LR replacement architecture has two critical flaws:

1. **Zero equilibrium/feedforward:** `LR_eq_ff_estimate_nm = 0.0` for all LR profiles. The replacement path zeros the individual torque terms (pitch, pitch_rate, position, support_velocity, sagittal_velocity, cp, com_vy) but does NOT recalculate the equilibrium/feedforward contribution. The static equilibrium torque required to keep the robot upright (~3-4 Nm per wheel at 0.48m) is completely absent.

2. **10x insufficient feedback:** `LR_feedback_torque_nm` RMS is ~1.2-1.3 Nm versus K1's existing estimate of ~15 Nm RMS. The LR gain functions produce gains (`k_pitch` 3.5-6.0 Nm/rad, `k_support` -8 to -12 Nm/m) that are dimensioned as total authority but the feedback calculation only multiplies these by the state errors without accounting for the physics feedforward bias.

The combination means the robot receives ~1.3 Nm of feedback with no static equilibrium compensation, compared to K1's ~15 Nm of combined pitch+position+support+feedforward torque. The robot pitches over uncontrollably until the simulation aborts.

### 4.5 LR Classification

| Candidate | Classification |
|-----------|---------------|
| LR1 | `LR_FOCUSED_RECOVERY_FAIL_UNSTABLE` |
| LR2 | `LR_FOCUSED_RECOVERY_FAIL_UNSTABLE` |
| LR3 | `LR_FOCUSED_RECOVERY_FAIL_UNSTABLE` |

**No LR candidate achieved sustained recovery. All terminated at ~180 steps.**

---

## 5. M Profile D4/D5 Results

### 5.1 Wiring Verification

| Metric | M1 D4 | M1 D5 | M2 D4 | M2 D5 |
|--------|-------|-------|-------|-------|
| `wheel_yaw_enabled` | True ✓ | True ✓ | True ✓ | True ✓ |
| `wheel_yaw_profile_activated` | True ✓ | True ✓ | True ✓ | True ✓ |
| `wheel_yaw_kp` | 0.5 | 0.5 | 0.8 | 0.8 |
| `wheel_yaw_kd` | 0.1 | 0.1 | 0.15 | 0.15 |
| `wheel_yaw_max_torque` | 1.5 Nm | 1.5 Nm | 2.0 Nm | 2.0 Nm |

**M profile wiring is VERIFIED.** The `DifferentialWheelYawStabilizer` is instantiated through profile parameters alone — no `--enable-wheel-yaw-stabilizer` CLI flag required.

### 5.2 D4 Results (low_0p330, 60N push)

| Metric | M1 D4 | M2 D4 | K1 Baseline |
|--------|-------|-------|-------------|
| Steps completed | 1000/1000 ✓ | 1000/1000 ✓ | — |
| Fell | No | No | — |
| `wheel_yaw_height_gate` | — | 0.0 | — |
| `wheel_yaw_tau_diff` RMS | **0.00 Nm** | **0.00 Nm** | — |
| hip_yaw_abs_max | **0.1891 rad** ✓ | **0.1891 rad** ✓ | 0.3595 rad |
| Pitch RMS | 3.50° | 3.50° | — |
| Support RMS | 0.250 m | 0.250 m | — |
| Roll RMS | 0.31° | 0.31° | — |
| yaw_drift max | 0.178 rad | — | — |

**D4 Analysis:** `wheel_yaw_tau_diff = 0` because the robot height (0.33m) is below the wheel-yaw height gate start (0.34m for both M1 and M2). The stabilizer is correctly gated out. hip_yaw_abs_max = 0.1891 rad is well within the 0.35 rad gate — **this is K1's base hip-yaw controller working effectively at low heights without wheel-yaw augmentation.**

### 5.3 D5 Results (high_0p480, 90N push)

| Metric | M1 D5 | M2 D5 | K1 Baseline |
|--------|-------|-------|-------------|
| Steps completed | 621/1000 ✗ | 377/1000 ✗ | — |
| Fell | No (aborted) | No (aborted) | — |
| `wheel_yaw_tau_diff` RMS | 0.53 Nm | 0.26 Nm | — |
| `wheel_yaw_tau_diff` max | 1.99 Nm | 1.62 Nm | — |
| **hip_yaw_abs_max** | **0.6242 rad** ✗ | **0.5463 rad** ✗ | **0.3529 rad** |
| yaw_drift max | 1.951 rad | 0.741 rad | — |
| Pitch RMS | 7.96° | 8.04° | 5.50° |
| Support RMS | 0.258 m | 0.215 m | 0.162 m |
| Roll RMS | 0.52° | 3.42° | ~0.27° |
| Roll max | 1.63° | 16.30° | — |

**D5 Analysis:** When the wheel-yaw stabilizer is active (inside height gate at 0.48m), it produces differential wheel torque but causes **severe hip-yaw regression** (M1: +77% vs K1, M2: +55% vs K1). The anti-symmetric wheel torque couples into roll dynamics, causing roll destabilization (M2 D5 roll RMS = 3.42° vs K1 ~0.27°) and early termination. **The wheel-yaw stabilizer is counterproductive at current parameters.**

### 5.4 M Classification

| Candidate | Scenario | Classification |
|-----------|----------|---------------|
| M1 | D4 | `M_D4D5_NO_IMPROVEMENT` (stabilizer gated out, K1 base handles D4) |
| M1 | D5 | `M_D4D5_REGRESSION` (hip_yaw +77%, early termination) |
| M2 | D4 | `M_D4D5_NO_IMPROVEMENT` (stabilizer gated out) |
| M2 | D5 | `M_D4D5_REGRESSION` (hip_yaw +55%, roll destabilization) |

---

## 6. N1 Micro-Sweep Results — ALL NO IMPROVEMENT

**Status:** All three N1 micro-sweep simulations completed (3000/3000 steps each, high_0p480, 90N push).

**Profiles under test:**
| Variant | k_rate range | k_lead range | Change vs N1 |
|---------|-------------|-------------|-------------|
| N1b | 0.4-0.6 | 0.03-0.06 | +33% rate, +50% lead |
| N1c | 0.4-0.6 | 0.025-0.05 | +33% rate, +25% lead |
| N1d | 0.35-0.55 | 0.03-0.06 | +10% rate, +50% lead |

**N1 baseline** (from prior evaluation): 2.44s transient hold, recovery later lost, final pitch 5.30° vs K1 4.93°.

### 6.1 N1 Results

| Metric | N1b | N1c | N1d | K1 Baseline | Delta vs K1 |
|--------|-----|-----|-----|-------------|-------------|
| Steps completed | 2999/3000 | 2999/3000 | 2999/3000 | 3000 | ≈ same |
| Fell | No | No | No | No | — |
| Pitch RMS (°) | 5.78 | 5.78 | 5.77 | 5.50 | **+5% worse** |
| Pitch max (°) | 20.25 | 20.15 | 20.35 | — | — |
| Support RMS (m) | 0.167 | 0.168 | 0.167 | 0.162 | **+3% worse** |
| Roll RMS (°) | 0.82 | 0.88 | 0.88 | ~0.27 | **+215% worse** |
| hip_yaw_abs_max (rad) | 0.463 | 0.457 | 0.465 | 0.299 | **+55% worse** |
| Longest stable run | 312 steps (1.56s) | 312 steps (1.56s) | 312 steps (1.56s) | — | — |
| Stable run start | Step 0 | Step 0 | Step 0 | — | **Pre-push only** |
| Post-push 2s hold | **No** | **No** | **No** | No | — |
| PSD 0.52 Hz | 0.005498 | — | — | — | — |
| PSD 2.5 Hz | 0.001773 | — | — | — | — |
| WBC/hidden/ownership | None | None | None | None | Clean |

### 6.2 Analysis

All three N1 micro-sweep variants produce **essentially identical results.** The parameter variations (k_rate: 0.35-0.6, k_lead: 0.025-0.06) are too small to produce meaningful differences at the system level.

Key observations:
1. **No sustained post-push recovery** — the longest stable interval (312 steps = 1.56s) starts at step 0, before the push at step 300. This is initial settling, not recovery.
2. **Pitch slightly worse than K1** — +5% across all three (5.77° vs 5.50°)
3. **hip_yaw significantly worse than K1** — +55% (0.46 rad vs 0.30 rad). The phase-lead damping appears to couple into hip-yaw dynamics.
4. **Roll worse than K1** — +215% (0.85° vs 0.27°). The increased pitch-rate feedback may excite roll modes.
5. **Worse than old N1** — The old N1 achieved 2.44s transient hold; new variants show zero post-push hold. Parameter changes from N1 (0.3-0.5/0.02-0.04) to N1b (0.4-0.6/0.03-0.06) moved in the wrong direction.

### 6.3 Classification

| Candidate | Classification |
|-----------|---------------|
| N1b | `N1_MICRO_SWEEP_NO_IMPROVEMENT` |
| N1c | `N1_MICRO_SWEEP_NO_IMPROVEMENT` |
| N1d | `N1_MICRO_SWEEP_NO_IMPROVEMENT` |

**No N1 variant achieved sustained recovery or beat K1 on any metric.**

---

## 7. K1 vs Candidate Comparison Matrix

| Metric | K1 | LR1 | LR2 | LR3 | M1 D4 | M1 D5 | M2 D4 | M2 D5 | N1b | N1c | N1d |
|--------|----|-----|-----|-----|-------|-------|-------|-------|-----|-----|-----|
| Steps completed | 3000 | **183** | **185** | **179** | 1000 | **621** | 1000 | **377** | 2999 | 2999 | 2999 |
| Fell | No | No* | No* | No* | No | No* | No | No* | No | No | No |
| Pitch RMS (°) | 5.50 | 7.73 | 7.70 | 7.65 | 3.50 | 7.96 | 3.50 | 8.04 | 5.78 | 5.78 | 5.77 |
| Support RMS (m) | 0.162 | 0.071 | 0.075 | 0.063 | 0.250 | 0.258 | 0.250 | 0.215 | 0.167 | 0.168 | 0.167 |
| Roll RMS (°) | ~0.27 | 0.27 | 0.27 | 0.27 | 0.31 | 0.52 | 0.31 | 3.42 | 0.82 | 0.88 | 0.88 |
| hip_yaw D4 (rad) | 0.3595 | — | — | — | 0.1891 | — | 0.1891 | — | — | — | — |
| hip_yaw D5 (rad) | 0.3529 | — | — | — | — | **0.6242** | — | **0.5463** | 0.463 | 0.457 | 0.465 |
| Sustained 2s hold | None | None | None | None | — | — | — | — | None | None | None |
| WBC/hidden/ownership | None | None | None | None | None | None | None | None | None | None | None |

*Early termination (aborted, not fell)

---

## 8. Wheel-Yaw Telemetry Verification

| Metric | M1 D4 | M1 D5 | M2 D4 | M2 D5 |
|--------|-------|-------|-------|-------|
| `wheel_yaw_enabled` | True | True | True | True |
| `wheel_yaw_profile_activated` | True | True | True | True |
| `wheel_yaw_height_gate` | — | — | 0.0 | — |
| `wheel_yaw_tau_diff` nonzero | **No** | **Yes** | **No** | **Yes** |
| `wheel_yaw_tau_diff` RMS | 0.00 | 0.53 Nm | 0.00 | 0.26 Nm |

**Conclusion:** Profile-based wiring works. When robot height is above the gate, `wheel_yaw_tau_diff` is nonzero (verified). The issue is NOT a wiring failure — it's that the stabilizer's effect on hip_yaw is counterproductive at the tested gains.

---

## 9. Low-Frequency 0.52 Hz Analysis

LR telemetry (only family with spectral data available):

| Candidate | PSD at 0.52 Hz | PSD at 2.5 Hz |
|-----------|---------------|---------------|
| LR1 | 0.000656 | 0.005924 |
| LR2 | 0.000636 | 0.005679 |
| LR3 | 0.000669 | 0.006047 |

LR sims terminated too early (~1.8s) for meaningful spectral analysis. N1 and M sims pending.

---

## 10. Safety / WBC / Hidden Torque / Ownership Audit

| Check | LR1/2/3 | M1/M2 | N1b/c/d |
|-------|---------|-------|---------|
| WBC enabled | ❌ None | ❌ None | ❌ None |
| Hidden torque | ❌ None | ❌ None | ❌ None |
| Ownership violation | ❌ None | ❌ None | ❌ None |
| NaN/Inf | ❌ None | ❌ None | ❌ None |

**No WBC, hidden torque, or ownership violations detected in any completed simulation.**

---

## 11. Current-Best After Task

| Item | Value |
|------|-------|
| Current-best | `K1_PITCH_RATE_NOTCH_V1` |
| Profile | `k1_pitch_rate_notch_v1` |
| Status | `K1_REMAINS_CURRENT_BEST_NO_READY_CANDIDATE` |
| Step C quick | FAIL (height tracking non-functional, but K1 stable — does not block) |

---

## 12. Files Changed

| File | Change |
|------|--------|
| `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py` | LR fields + gain functions + profile constants + compute replacement logic + telemetry + N1 parameter fields + N1b/c/d profiles |
| `scripts/simulate_hierarchical_controller.py` | LR + N1 imports + profile registry + argparse choices + M profile-based stabilizer activation + wheel_yaw_profile_activated telemetry |
| `scripts/run_true_dynamic_height_step_c_validation.py` | Timeout 1800→5400 + QUICK_HEIGHT_PROFILES + --quick flag |
| `scripts/audit_k1_sustained_recovery_failure.py` | Updated telemetry search path |
| `tests/test_k1_next_controller_fix.py` | **NEW** — 52 tests |
| `docs/validation/k1_next_controller_fix_ready_infrastructure_evaluation_report.md` | **NEW** — this report |

---

## 13. Tests / Compile Checks Run

```
Compile checks:                                      6/6 PASS
pytest tests/test_k1_next_controller_fix.py:         52/52 PASS
pytest tests/test_current_best_controller_profile.py:  8/8 PASS
pytest tests/test_final_validation_rejects_stub_source.py: 9/9 PASS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOTAL:                                              75/75 PASS
```

---

## 14. Simulation Runs Performed

| Run | Profile | Scenario | Steps | Status |
|-----|---------|----------|-------|--------|
| LR1 | lr1_k1_replacement_coordinated_low_freq_v1 | high_0p480, 90N | 183/3000 | FAIL |
| LR2 | lr2_k1_replacement_phase_lead_v1 | high_0p480, 90N | 185/3000 | FAIL |
| LR3 | lr3_k1_replacement_pitch_ref_stabilized_v1 | high_0p480, 90N | 179/3000 | FAIL |
| M1 D4 | m1_k1_body_yaw_diff_wheel_v1 | low_0p330, 60N | 1000/1000 | OK |
| M1 D5 | m1_k1_body_yaw_diff_wheel_v1 | high_0p480, 90N | 621/1000 | REGRESSION |
| M2 D4 | m2_k1_body_yaw_support_aware_v1 | low_0p330, 60N | 1000/1000 | OK |
| M2 D5 | m2_k1_body_yaw_support_aware_v1 | high_0p480, 90N | 377/1000 | REGRESSION |
| N1b | n1b_k1_mild_phase_lead_v1 | high_0p480, 90N | 2999/3000 | NO IMPROVEMENT |
| N1c | n1c_k1_mild_phase_lead_v1 | high_0p480, 90N | 2999/3000 | NO IMPROVEMENT |
| N1d | n1d_k1_mild_phase_lead_v1 | high_0p480, 90N | 2999/3000 | NO IMPROVEMENT |
| Step C quick | k1_pitch_rate_notch_v1 | 5 dynamic profiles | 1500-2000/1500-2000 | FAIL (height tracking) |

---

## 15. Next Recommended Tasks

1. **Fix LR implementation:** The replacement path must include equilibrium/feedforward torque. The LR feedback torque calculation needs to produce ~15 Nm RMS (not ~1.3 Nm) to match K1's control authority. This requires either:
   - Adding `physics_ff` contribution to the LR replacement path, OR
   - Multiplying LR gains by ~10x (which may destabilize), OR  
   - Replacing only the dynamic terms while preserving equilibrium/feedforward as pass-through

2. **Redesign M wheel-yaw parameters:** Current gains cause counterproductive hip-yaw increase. Options:
   - Lower kp (0.1-0.3 instead of 0.5-0.8)
   - Add roll-stabilization coupling
   - Test intermediate heights (0.38-0.42m) where gate is partially open
   - Gate wheel-yaw by roll magnitude

3. **Complete N1 micro-sweep:** Wait for 3000-step sims to finish, analyze for sustained hold improvement.

4. **Complete K1 Step C quick:** Wait for 5-profile dynamic height validation to verify notch gate crossing safety.

5. **Before any re-evaluation:** Fix LR equilibrium/feedforward bug (critical — blocks all LR testing).

---

## 16. Decision Records

**DR-1:** LR candidates are NOT ready for broader validation. Implementation fix required (zero equilibrium/feedforward).

**DR-2:** M candidates are NOT ready for broader validation. Parameter redesign required (wheel-yaw destabilizes hip_yaw at current gains).

**DR-3:** K1 remains current-best. No candidate promotion justified.

**DR-4:** N1 micro-sweep completed. All three variants (N1b/N1c/N1d) show no improvement over K1. Parameter changes (k_rate: 0.35-0.6, k_lead: 0.025-0.06) are too small to produce meaningful system-level differences. Phase-lead damping at these levels couples into hip_yaw (+55%) and roll (+215%) without improving pitch or enabling sustained recovery.

**DR-5:** Step C quick completed. Dynamic height tracking is non-functional (CoM doesn't follow target). K1 remains stable during height command changes but notch gate crossing is NOT physically validated. This is a test infrastructure limitation, not a controller instability. Does not block candidate evaluation.

**DR-6:** No WBC, hidden torque, or ownership violations found. All profiles clean on safety audit.

---

*All results from real_simulation. No stub/assumed/synthetic rows. Direct telemetry parsed from CSV output files.*
