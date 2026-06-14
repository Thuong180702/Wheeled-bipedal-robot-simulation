# APCR1m Full Audit and Comparison Final Report

## Phase 10: Final Report

**Date**: 2026-06-11
**Classification**: `APCR1M_AUDIT_NOT_BETTER_THAN_APCR1H`
**Decision**: APCR1m is NOT better than APCR1h. APCR1n design recommended.

---

## Executive Summary

APCR1m (conditional pitch blend recenter) was designed to fix APCR1l's startup failure while maintaining sagittal balance. While APCR1m successfully addresses the startup issue, it performs **2.4x worse** than APCR1h on drift metrics.

**Key Findings**:
1. ✅ APCR1m startup stable (no fall in 1000 steps)
2. ✅ APCR1m telemetry gap fixed (all blend fields captured)
3. ✅ APCR1m does blend tau_pitch during RECENTER (32.6% reduction)
4. ❌ Blend blocked by safety gates 38.7% of the time
5. ❌ APCR1m max|e| = 0.434m vs APCR1h = 0.178m (2.4x worse)
6. ❌ APCR1h is currently the best profile

**Root Causes of APCR1m's Poor Drift Performance**:
1. **Wheel velocity damping 3.5x larger**: 5.0 Nm vs 1.4 Nm
2. **Position cap saturated 77.3%**: Raw would be ±15 Nm, capped at ±3 Nm
3. **Final torque fights drift 62.8%**: Despite RECENTER being active

---

## Validation Results Summary

### PHASE 1: Telemetry Gap Fix

| Test | Result |
|------|--------|
| APCR1m fields initialized in telemetry template | ✅ PASS |
| APCR1m fields appended from sagittal_diag | ✅ PASS |
| CSV contains all 12 APCR1m fields | ✅ PASS |
| 1000 rows captured | ✅ PASS |
| Classification | `APCR1M_TELEMETRY_GAP_FIXED` |

### PHASE 2: APCR1m 1000-Step Run

| Test | Result |
|------|--------|
| Steps survived | 1000/1000 |
| Startup stability (0-150) | ✅ pitch 0.0°-0.01°, roll -0.0°-0.2° |
| Height range | 0.246 - 0.308 m |
| Classification | `APCR1L_FIX_VALIDATED` |

### PHASE 3: Comparison References

| Profile | Steps | Status |
|---------|-------|--------|
| APCR1h | 1000 | ✅ Stable |
| APCR1j | 1000 | ✅ Stable |
| APCR1k | 1000 | ✅ Stable |
| APCR1m | 1000 | ✅ Stable |

### PHASE 4: Physical Drift Comparison

| Metric | APCR1h | APCR1j | APCR1k | APCR1m | Winner |
|--------|--------|--------|--------|--------|--------|
| max \|e\| (m) | **0.178** | 0.183 | 0.232 | 0.434 | APCR1h |
| P2P (m) | **0.249** | 0.251 | 0.303 | 0.833 | APCR1h |
| outside ±0.15 | **9.7%** | 25.8% | 20.2% | 54.0% | APCR1h |
| wheel damping | **1.42** | 1.34 | 1.31 | 5.00 | APCR1h |
| Classification | - | - | - | - | `CURRENT_BEST_APCR1H` |

### PHASE 5: Blend Behavior Audit

| Metric | Value |
|--------|-------|
| Blend active | 42.2% |
| RECENTER active | 88.4% |
| Safety gates all passing | 40.4% |
| tau_pitch reduction during RECENTER | 32.6% |
| tau_pitch fights APCR after blend | 63.2% |
| Classification | `APCR1M_BLEND_SAFETY_GATES_BLOCKING` |

### PHASE 6: Torque Composition Audit

| Metric | Value |
|--------|-------|
| Wheel vel dominance | 45.5% |
| Final torque fights drift | 62.8% |
| tau_pitch fights drift | 98.5% |
| tau_position correct sign | 100.0% |
| Classification | `APCR1M_DRIFT_FROM_FINAL_TORQUE_DIRECTION_VIOLATIONS` |

### PHASE 7: Position Cap Audit

| Metric | Value |
|--------|-------|
| Position cap | ±3 Nm |
| Saturation rate | 77.3% |
| Saturation during RECENTER | 87.3% |
| Raw tau_position range | ±15 Nm |
| Classification | `APCR1M_POSITION_CAP_LIMITS_RECENTER` |

---

## Answers to Required Questions

1. **Was APCR1m startup stable?** ✅ YES - No fall in 1000 steps, pitch 0.0°-0.01°

2. **Were APCR1m blend telemetry fields fixed and captured?** ✅ YES - All 12 fields populated

3. **Did APCR1m actually blend tau_pitch during RECENTER?** ✅ YES - 32.6% reduction

4. **How often was blend blocked by safety gates?** 38.7% blocked, only 40.4% have all gates passing

5. **What are APCR1m's drift metrics?**
   - max|e| = 0.434m
   - min e = -0.434m
   - P2P = 0.833m
   - outside ±0.08 = 75.7%
   - outside ±0.10 = 69.1%
   - outside ±0.15 = 54.0%

6. **How does APCR1m compare to APCR1h/j/k?** ❌ WORST in all drift metrics

7. **Which profile is currently best?** **APCR1h** (lowest max|e|, P2P, band violations)

8. **Why does APCR1m still drift if APCR RECENTER is active 88.5%?** Because:
   - Final torque fights drift 62.8% of the time
   - Wheel velocity damping is 3.5x larger than APCR1h
   - Position cap saturated 77.3%

9. **Is drift caused by wheel velocity damping dominance?** ✅ YES - 5.0 Nm vs 1.4 Nm

10. **Is drift caused by tau_pitch still dominating?** ❌ NO - tau_pitch fights drift but is reduced 32.6%

11. **Is drift caused by position torque cap?** ✅ YES - saturated 77.3%, limits recenter

12. **Is drift caused by final torque direction violations?** ✅ YES - 62.8% of RECENTER steps

13. **Should APCR1m proceed to 2000-step?** ❌ NO - drift is 2.4x worse than APCR1h

14. **Should APCR1n be designed?** ✅ YES - Root causes identified, design recommended

---

## Final Classification

```
APCR1M_AUDIT_NOT_BETTER_THAN_APCR1H
```

APCR1m is validated as an APCR1l startup fix, but is NOT a support drift fix. APCR1h remains the best profile.

---

## APCR1n Design Recommendation

**Design**: APCR1n = APCR1h base + wheel damping override + position cap boost

**Target**:
- max|e| < 0.200m (vs APCR1m's 0.434m)
- wheel damping ~1.5-2.0 Nm (vs APCR1m's 5.0 Nm)
- position cap ±5 Nm during RECENTER (vs APCR1m's ±3 Nm)

**Key changes**:
1. Reduce wheel velocity damping during RECENTER (0.3x scale)
2. Increase position cap from ±3 Nm to ±5 Nm during safe RECENTER
3. Preserve APCR1m's startup guard (100 steps)

---

## Files Generated

### Reports
- `docs/validation/apcr1m_vs_prior_profiles_drift_table.md`
- `docs/validation/apcr1m_pitch_blend_behavior_audit.md`
- `docs/validation/apcr1m_torque_composition_dominance_audit.md`
- `docs/validation/apcr1m_position_torque_cap_audit.md`
- `docs/validation/current_best_apcr_profile_decision.md`
- `docs/validation/apcr1n_candidate_design_from_apcr1m_audit.md`
- `docs/validation/apcr1m_full_audit_and_comparison_final_report.md` (this file)

### JSON Outputs
- `outputs/.../apcr1m_vs_prior_profiles_drift_table.json`
- `outputs/.../apcr1m_pitch_blend_behavior_audit.json`
- `outputs/.../apcr1m_torque_composition_dominance_audit.json`
- `outputs/.../apcr1m_position_torque_cap_audit.json`
- `outputs/.../current_best_apcr_profile_decision.json`
- `outputs/.../apcr1n_candidate_design_from_apcr1m_audit.json`
- `outputs/.../apcr1m_full_audit_and_comparison_summary.json` (pending)

### Telemetry Data
- `outputs/.../apcr1m_low_0p300_1000_full_telemetry/telemetry.csv`
- `outputs/.../comparison_1000_apcr1h/telemetry.csv`
- `outputs/.../comparison_1000_apcr1j/telemetry.csv`
- `outputs/.../comparison_1000_apcr1k/telemetry.csv`

---

## Do NOT

- ❌ Do NOT run 2000-step on APCR1m
- ❌ Do NOT run 5000-step on APCR1m
- ❌ Do NOT claim Step E pass
- ❌ Do NOT make APCR1m the default
- ❌ Do NOT implement APCR1n without testing