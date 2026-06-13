# T5 High 0.480m 200-Step Smoke Validation Report

**Date:** 2026-06-12  
**Profile:** APCR1nD_T5_band_limited_balanced  
**Height Variant:** high_0p480  
**Steps:** 200  
**Classification:** T5_HIGH_0P480_SMOKE_PASS_PROCEED_TO_5000

---

## Smoke Pass Criteria

| Criterion | Target | Result | Status |
|-----------|--------|--------|--------|
| Survives 200 steps | 200 | 199 (200 simulated) | ✅ PASS |
| Contact maintained | Stable | 100% both wheels | ✅ PASS |
| CoM Z does not collapse | > 0.43 m | 0.481 - 0.491 m | ✅ PASS |
| Roll remains safe | < 5 deg RMS | 0.072 deg | ✅ PASS |
| Pitch remains safe | < 10 deg RMS | 4.095 deg | ✅ PASS |
| No WBC/hidden/ownership | 0 | 0 | ✅ PASS |
| Tuned telemetry valid | 19 fields | 19 fields | ✅ PASS |

**Result:** All 7 smoke criteria passed

---

## Key Observations

### Tuned Telemetry Verification
- **Fields present:** 19/19 ✅
- **tuned_variant_name:** T5
- **tuned_band_state:** normal, emergency
- **tuned_band_state_id max:** 4 (emergency band)
- **tuned_recenter_active:** True (activated during run)

T5 tuned features ARE activating correctly at high_0p480.

### Stability
- **Contact:** 100% left + 100% right (perfect)
- **CoM Z:** 0.481 - 0.491 m (stable)
- **Pitch RMS:** 4.095 deg (acceptable for high height)
- **Roll RMS:** 0.072 deg (excellent)
- **Wheel velocity max:** 5.46 rad/s (safe, below 7 rad/s threshold)

### Drift Behavior (200 steps)
- **Max |e|:** 0.1871 m
- **Mean |e|:** 0.1041 m

Drift is elevated compared to low_0p300, but this is expected at extreme height. The 5000-step run will reveal whether drift accumulates or stabilizes.

### Band State Activity
- T5 reached **emergency band** (state_id = 4)
- Recenter feature activated
- This indicates T5 is responding to higher drift at extreme height

---

## Comparison to Low_0p300 Smoke (Expected Differences)

High_0p480 is a more challenging height variant:
- Higher CoM → less stable equilibrium
- Larger gravitational torque arm → stronger pitch coupling
- Elevated drift expected

The smoke test confirms:
- ✅ Robot survives at high_0p480
- ✅ T5 tuned features activate appropriately
- ✅ No catastrophic instability
- ✅ Safety gates hold

---

## Next Step

**Proceed to 5000-step validation** at high_0p480.

**Command:**
```bash
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --vd-sagittal-authority-profile APCR1nD_T5_band_limited_balanced \
  --height-variant-setup outputs/physical_target_height_setups/high_0p480_setup.json \
  --steps 5000 \
  --telemetry-decimation 1 \
  --failure-window-steps 5000 \
  --write-run-summary-sidecar
```

---

## Classification

**T5_HIGH_0P480_SMOKE_PASS_PROCEED_TO_5000**

All smoke criteria passed. No blockers detected. Ready for 5000-step validation.

---

**Date:** 2026-06-12  
**Phase:** 2 (200-Step Smoke Validation) COMPLETE
