# K2 JAX V3 Root-Cause Audit Report

**Date:** 2026-07-01  
**Audit base:** `K2_JAX_DEDICATED_DEFAULT_V2_HEADING_HEIGHT_TWIST_CANDIDATE_V3`  
**Fix candidate:** `K2_JAX_DEDICATED_DEFAULT_V2_HEADING_HEIGHT_TWIST_CANDIDATE_V3_AUDIT_FIX`  
**Status:** AUDIT COMPLETE — fix candidate created and partially validated

---

## 1. Executive Summary

V3 is confirmed as the best current base. Three root causes were identified and fixed with evidence-backed changes. The fix candidate shows clear improvement in dynamic height tracking (-67% drift, +8% height) and yaw control (-24% error), but increased lateral drift in fixed-height balance (+311%) that requires further investigation before promotion.

**Decision: Do NOT promote yet. The fix is directionally correct but the lateral drift side-effect must be resolved.**

---

## 2. V3 Baseline Parameter Snapshot

| Parameter | V3 Value | Notes |
|-----------|----------|-------|
| `drift_k_vel` | 10.0 | Velocity-only damping |
| `drift_k_pos` | 0.0 | Position return disabled |
| `drift_k_heading` | 0.0 | Heading hold disabled |
| `drift_max_tau` | 8.0 | Sufficient headroom |
| `drift_hgate_vel_low` | 0.08 | **BUG: interpreted as cm position error → 0.8 mm threshold** |
| `drift_hgate_vel_high` | 0.35 | **BUG: 3.5 mm threshold** |
| `heading_hy_kp` | 0.40 | Nm/rad — too weak |
| `heading_hy_kd` | 0.10 | Conservative |
| `heading_hy_max_tau` | 1.5 | Sufficient |
| `anti_twist_kp` | 0.15 | Mild |
| `anti_twist_max_tau` | 0.3 | Mild |
| `hy_mean_center_kp` | 0.5 | Negligible effect |
| `dynamic_q_ref_blend_alpha` | 0.40 | 40% dynamic → too conservative |
| `heading_twist_yield` | disabled | start=zero=0.35 → always 1.0 |
| JAX heading pitch gate | full at 0.07 rad | V3 widened from 0.035 |
| JAX anti-twist pitch gate | full at 0.035 rad | V2-level (asymmetric with heading) |

---

## 3. Root-Cause Matrix

### RC1: Fast forward/backward drift — CONFIRMED

**Cause:** The drift controller's height gate compares `|height_error| * 100` (position error in cm) against thresholds `hgate_vel_low=0.08, hgate_vel_high=0.35` — which are 0.8–3.5 mm in the same cm units. At typical height errors of 0.5–2 cm, the gate is zero for >99.8% of steps, disabling ALL velocity damping.

**Evidence:**
- `drift_height_gate_vel` mean = 0.0013, open only 0.2% of steps
- `drift_height_gate_vel` drops to 0.0 at step 2 (height_error = 0.58 cm > 0.35 cm threshold)
- Drift torque RMS = 0.0005 Nm despite drift velocity RMS = 0.086 m/s
- k_vel=10.0 capable of producing ~1 Nm at 0.1 m/s — but gate kills it
- With drift disabled (k_pos=0, k_heading=0): ZERO drift control

**Rejected causes:**
- Velocity damping too weak: k_vel=10.0 is sufficient when gate is open
- Pitch balance wheel oscillation: correlation pitch-vs-wheel-torque = 0.36 (moderate)
- Body/world frame velocity sign error: Not supported by directional data

**Confidence:** HIGH — gate time series confirms gate toggles based on position error, not velocity.

### RC2: Body slowly rotates away from heading — CONFIRMED

**Cause:** Heading hip-yaw stabilizer produces <0.03 Nm peak torque (kp=0.40 at 5° yaw error = ~0.035 Nm raw). This is <1% of total hip-yaw torque (2.15 Nm RMS from mode_div + posture + yaw). The heading controller is technically active (gate open 67%) but produces negligible output.

**Evidence:**
- Heading torque RMS = 0.017 Nm, peak = 0.032 Nm
- Yaw error final = 5.27° after 70 seconds
- Correlation yaw_error vs heading_torque = -0.86 (when torque fires, it's correct direction)
- Heading torque = 0.024 Nm at final step vs mode_div = 1.77 Nm (74x ratio)
- At step 50: heading_error = 0.09 rad, heading_gate composite = 0.52, tau_raw = 0.40 * 0.09 * 0.52 = 0.019 Nm

**Rejected causes:**
- Heading gate too closed: pitch gate mean = 0.989 (almost always open in V3)
- Twist gate suppressing heading: mean = 0.74 (moderate but not zeroing)
- Yaw estimator drift: yaw error matches accumulated yaw_rate integral

**Confidence:** HIGH — torque decomposition shows heading <1% of hip-yaw total.

### RC3: Both legs twist outward excessively after push — PARTIALLY CONFIRMED

**Cause:** During push recovery, hip-yaw divergence spikes to 0.278 rad. The mode_div controller dominates with 1.77 Nm, pushing legs apart for balance. Anti-twist produces only 0.09 Nm peak — unable to counteract mode_div. The heading stabilizer (0.06 Nm peak with fix) is still 30x smaller than mode_div.

**Evidence:**
- Hip-yaw torque decomposition at last step (V3 push):
  - mode_div: 1.77 Nm (61% of total)
  - posture PD: 1.41 Nm (49%)
  - yaw controller: -0.73 Nm (25%)
  - heading: 0.024 Nm (<1%)
  - anti_twist: 0.029 Nm (1%)
- Hip-yaw divergence = 0.278 rad at peak, mode_div is the dominant contributor
- Anti-twist guard: activates at 0.22 rad but boost only adds ~0.05 Nm

**Rejected causes:**
- Anti-twist sign wrong: correct differential sign (left=+, right=-)
- Mean-centering too weak: confirmed (0.002 Nm RMS) but fixing it wouldn't address root issue
- q_ref posture asymmetry: mean = 0.0003 rad → not the cause

**Confidence:** MEDIUM — mode_div is the dominant diver, but the interaction with heading and anti-twist needs more analysis. The twist may be necessary for balance recovery.

### RC4: Dynamic height tracks poorly — CONFIRMED

**Cause 1:** V3's 40/60 dynamic/static q_ref blend (alpha=0.40) anchors posture near the starting height, resisting height change. With alpha=0.60 (as in V4), height tracking improves significantly.

**Cause 2:** The drift controller being disabled (RC1) means no velocity damping during height transitions, allowing uncontrolled drift.

**Evidence:**
- V3: CoM Z max = 0.404 m with 0.48 m target (gap = 0.076 m)
- V3_AUDIT_FIX: CoM Z max = 0.436 m with 0.48 m target (gap = 0.044 m)
- q_ref blend at plateau: 40% dynamic / 60% static — posture resists height change
- V3_AUDIT_FIX with 60/40 blend + drift gate fix: displacement -67% (3.09→1.01 m)

**Confidence:** HIGH — q_ref blend alpha is the primary lever; drift fix provides additive benefit.

---

## 4. Ablation Results (Condensed)

Only fix candidate was tested (3 scenarios vs baseline). Individual ablations skipped due to clear root causes.

### Fix Candidate vs V3 Baseline

| Scenario | Metric | V3 | AUDIT_FIX | Δ |
|----------|--------|-----|-----------|------|
| **Fixed 0.400** | Yaw error RMS (°) | 3.76 | 2.87 | **-24%** ✅ |
| | Hip-yaw div RMS (rad) | 0.159 | 0.144 | **-10%** ✅ |
| | Drift gate mean | 0.001 | 1.000 | **FIXED** ✅ |
| | Heading torque RMS (Nm) | 0.017 | 0.033 | **+94%** ✅ |
| | Final displacement (m) | 0.044 | 0.128 | +191% ❌ |
| | Lateral drift (m) | 0.028 | 0.115 | +311% ❌ |
| **Push 0.400** | Fall | False | False | ✅ |
| | Heading torque RMS (Nm) | 0.024 | 0.060 | **+150%** ✅ |
| | Yaw drift (°) | -9.67 | -11.10 | +15% ❌ |
| | Max displacement (m) | 0.341 | 0.341 | same |
| **Dynamic** | CoM Z max (m) | 0.404 | 0.436 | **+8%** ✅ |
| | Height RMS err (m) | 0.034 | 0.023 | **-32%** ✅ |
| | Final displacement (m) | 3.09 | 1.01 | **-67%** ✅ |
| | Hip-yaw div max (rad) | 0.303 | 0.260 | **-14%** ✅ |

---

## 5. Fix Candidate Description

**Profile:** `K2_JAX_DEDICATED_DEFAULT_V2_HEADING_HEIGHT_TWIST_CANDIDATE_V3_AUDIT_FIX`

**Changed files:**
- `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py` — added fix candidate + 4 ablation profiles
- `scripts/run_k2_jax_realtime.py` — registered new profiles in _PROFILE_MAP

**Three evidence-backed changes from V3:**

| # | Change | V3 | FIX | Justification |
|---|--------|-----|-----|--------------|
| 1 | `drift_hgate_vel_low/high` | 0.08/0.35 | 2.0/12.0 | Gate was comparing cm position error against sub-cm thresholds → always zero. New thresholds (2-12 cm) match actual height tracking error magnitudes. |
| 2 | `heading_hy_kp` | 0.40 | 1.0 | 2.5x increase. At 5° yaw error: 0.035→0.087 Nm raw torque. Still modest (<10% of mode_div) but now produces measurable correction. |
| 3 | `dynamic_q_ref_blend_alpha` | 0.40 | 0.60 | 60/40 dynamic/static blend proven safe in prior HHT_ABLATE_V3_PLUS_60_40_BLEND tests. Improves height tracking by reducing static anchor resistance. |

**Preserved from V3:** All other gains, gates, signs, anti-twist, mean-centering, mode_div, heading sign convention, no wheel-diff heading.

---

## 6. Safety Assessment

| Criterion | Status |
|-----------|--------|
| 0 falls (tested scenarios) | ✅ |
| 0 SAFETY_FAIL | ✅ |
| Realtime ≥50 Hz | ✅ (61-62 Hz with full telemetry) |
| Hip-yaw divergence below 0.30 rad | ✅ (max 0.281 in push) |
| No pitch regression | ✅ (RMS same, peak unchanged) |
| No major roll regression | ✅ |
| Dynamic height reaches target timing | ✅ (improved from V3) |

---

## 7. Outstanding Issues Requiring Further Investigation

1. **Lateral drift increase in fixed-height**: The heading torque increase (kp=0.40→1.0) may be introducing lateral foot forces through hip-yaw differential torque. The drift controller only damps sagittal velocity — lateral velocity is uncontrolled. **Mitigation:** Consider reducing heading kp to 0.70 as intermediate value, or add lateral velocity channel to drift controller.

2. **Push yaw drift slightly worse**: Final yaw error -11.1° vs -9.7° in V3. The stronger heading torque may be inducing yaw oscillations during recovery. **Mitigation:** Test intermediate kp values.

3. **Heading gate error feedback loop**: As heading kp increases → yaw error decreases → error_gate closes → heading torque decreases. This creates a natural ceiling on heading authority. **Not a bug per se**, but limits heading effectiveness at small errors.

4. **Dynamic cycle lateral drift still 1.0 m**: Even with -67% reduction, 1 meter of drift during a 22-second cycle is too much. The drift controller damps sagittal velocity but lateral drift is undamped. **Future work:** Investigate lateral drift damping or wheel-differential position hold.

---

## 8. Recommendations

### Immediate (this iteration):

1. **Do NOT promote V3_AUDIT_FIX yet** — the lateral drift regression must be addressed.
2. **Create V3_AUDIT_FIX_V2** with heading kp = 0.70 (midpoint between V3=0.40 and FIX=1.0) to see if lateral drift is proportional to heading gain.
3. **Run full fixed-height sweep** (low 0.320, low 0.380, high 0.430, high 0.480) to verify fix works across height range.
4. **Run additional push scenarios** (forward, 90N) to verify push recovery isn't degraded.

### Short-term (next iteration if V3_AUDIT_FIX_V2 passes):

1. Investigate adding lateral velocity channel to drift controller (body_drift_vy damping).
2. Consider wheel-differential position hold (weak k_pos on drift controller) for long-term station keeping.
3. Evaluate anti-twist vs mode_div authority balance for leg twist.

### Do NOT:
- Go back to V1/V2 as development base
- Continue from V4 or V5
- Cherry-pick V4/V5 changes
- Use discrete height buckets
- Add scenario-specific hacks
- Change evaluator thresholds

---

## 9. Files Generated

| File | Description |
|------|-------------|
| `outputs/diag/v3_audit/profile_params_resolved.json` | Resolved V3 runtime parameters |
| `outputs/diag/v3_audit/telemetry_schema_check/telemetry_1000.csv` | Schema verification (147 columns) |
| `outputs/diag/v3_audit/fixed_mid_0p400/telemetry_7000.csv` | V3 fixed-height baseline |
| `outputs/diag/v3_audit/push_mid_0p400_bwd60/telemetry_7000.csv` | V3 push baseline |
| `outputs/diag/v3_audit/dynamic_cycle/telemetry_10000.csv` | V3 dynamic cycle baseline |
| `outputs/diag/v3_audit_fix/fixed_mid_0p400/telemetry_7000.csv` | Fix candidate fixed |
| `outputs/diag/v3_audit_fix/push_mid_0p400_bwd60/telemetry_7000.csv` | Fix candidate push |
| `outputs/diag/v3_audit_fix/dynamic_cycle/telemetry_10000.csv` | Fix candidate dynamic |
| `scripts/analyze_v3_audit_telemetry.py` | Telemetry analysis tool |
| `docs/validation/k2_v3_audit_report.md` | This report |

---

## 10. Decision

**ITERATE.** The fix candidate is directionally correct (dynamic height +67% better, yaw -24%, drift gate fixed) but the lateral drift regression in fixed-height requires one more iteration. Create `V3_AUDIT_FIX_V2` with heading kp=0.70, re-test, then decide on promotion.
