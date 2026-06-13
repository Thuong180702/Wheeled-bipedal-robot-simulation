# T6B Runtime Configuration Identity Audit

**Date:** 2026-06-12  
**Status:** Phase 1 complete  
**Classification:** T6B_RUNTIME_CONFIG_CORRECT

---

## Executive Summary

**T6B runtime configuration VERIFIED CORRECT.** Telemetry confirms T6B_high_stronger_emergency was correctly loaded with expected parameter values:

- Emergency position cap: 8.0 Nm (vs T5 7.0 Nm) ✓
- Emergency damping scale: 0.05 (vs T5 0.10) ✓
- Variant name: T6B ✓

**Conclusion:** T6B parameters were correctly applied during 5000-step validation. The identical behavior to T5 is NOT due to configuration error or implementation bug. T6B changes reached the controller.

---

## Verification Results

### T6B Configuration

**Variant Name:**
- Expected: T6B
- Actual: T6B
- Status: PASS ✓

**Position Cap (tuned_position_cap_current):**
- Min: 4.0 Nm (desired band cap)
- Max: 8.0 Nm (emergency band cap)
- Mean: 5.5 Nm
- Expected emergency cap: 8.0 Nm
- Status: PASS ✓

**Wheel Damping Scale (tuned_wheel_damping_scale):**
- Min: 0.05 (emergency damping)
- Max: 1.00 (outside APCR, no damping applied)
- Mean: 0.72
- Expected emergency damping: 0.05
- Status: PASS ✓

### T5 Configuration (Reference)

**Variant Name:**
- Expected: T5
- Actual: T5
- Status: PASS ✓

**Position Cap (tuned_position_cap_current):**
- Min: 4.0 Nm
- Max: 7.0 Nm (emergency band cap)
- Mean: 5.3 Nm
- Expected emergency cap: 7.0 Nm
- Status: PASS ✓

**Wheel Damping Scale (tuned_wheel_damping_scale):**
- Min: 0.10 (emergency damping)
- Max: 1.00
- Mean: 0.73
- Expected emergency damping: 0.10
- Status: PASS ✓

---

## Key Differences Confirmed

| Parameter | T5 | T6B | Delta |
|-----------|-----|-----|-------|
| Emergency position cap | 7.0 Nm | 8.0 Nm | +14% |
| Hard position cap | 6.5 Nm | 7.0 Nm | +8% |
| Desired position cap | 5.5 Nm | 5.8 Nm | +5% |
| Emergency damping scale | 0.10 | 0.05 | -50% |
| Hard damping scale | 0.15 | 0.10 | -33% |
| Desired damping scale | 0.30 | 0.30 | 0% |

All differences are in the expected direction:
- T6B has **stronger authority** (higher caps)
- T6B has **more aggressive damping** (lower scales = less damping fights recovery)

---

## Interpretation

**T6B configuration was correctly applied.** The runtime telemetry shows:

1. Variant name correctly logged as "T6B"
2. Position cap reached 8.0 Nm (T6B emergency cap), never exceeded 7.0 Nm for T5
3. Damping scale reached 0.05 (T6B emergency damping), never went below 0.10 for T5

This rules out:
- Configuration file not loaded
- Wrong profile selected
- Parameters overridden at runtime
- Telemetry recording wrong controller

**Implication:** T6B's identical dynamics to T5 are NOT due to implementation error. The 14% cap increase and 50% damping reduction were correctly transmitted to the controller but produced zero observable effect on the final trajectory.

---

## Next Steps

Proceed to Phase 2: Compare T5 vs T6B torque path step-by-step to determine:
1. Did T6B produce different intermediate torque values than T5?
2. Did T6B produce different final wheel torques than T5?
3. If torques differed, why did dynamics remain identical?
4. If torques were identical, where was the T6B effect erased?

---

## Artifacts

**Analysis Script:**
- `audit_t6b_runtime_config.py`

**Data:**
- `outputs/.../t6b_runtime_config_identity_audit.json`

**Status:** Phase 1 complete  
**Classification:** T6B_RUNTIME_CONFIG_CORRECT  
**Date:** 2026-06-12
