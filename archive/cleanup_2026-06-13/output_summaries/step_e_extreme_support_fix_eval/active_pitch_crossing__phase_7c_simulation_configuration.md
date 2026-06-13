# Phase 7C: Paired 1200-Step Diagnostic Configuration

**Date:** 2026-06-12  
**Phase:** 7C - Torque Transmission Validation  
**Status:** Simulations running

---

## Configuration

### T5 Reference Baseline

**Profile:** `APCR1nD_T5_band_limited_balanced`

**Command:**
```bash
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --vd-sagittal-authority-profile APCR1nD_T5_band_limited_balanced \
  --height-variant-setup outputs/physical_target_height_setups/high_0p480_setup.json \
  --steps 1200 \
  --telemetry-decimation 1 \
  --failure-window-steps 1200 \
  --write-run-summary-sidecar
```

**Expected behavior:**
- `arch_fix_enabled = False`
- `max_position_tau_nominal = 4.0 Nm`
- Upstream clip limits position torque to 4.0 Nm
- Tuned caps (7.0 Nm emergency) ineffective due to upstream clip
- Final transmitted torque ≤ 4.0 Nm

**Result:** ✓ Completed (1199 steps)  
**Telemetry:** `outputs/hierarchical_controller_sim/telemetry_1781249941.csv`

---

### T6F Architecture Fix Candidate

**Profile:** `T6F_budget_cap_raise`

**Command:**
```bash
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --vd-sagittal-authority-profile T6F_budget_cap_raise \
  --height-variant-setup outputs/physical_target_height_setups/high_0p480_setup.json \
  --steps 1200 \
  --telemetry-decimation 1 \
  --failure-window-steps 1200 \
  --write-run-summary-sidecar
```

**Expected behavior:**
- `arch_fix_enabled = True`
- `arch_fix_type = "budget_cap_raise"`
- Normal/soft/desired bands: 4.0 Nm (unchanged from T5)
- Hard band at high_0p480: raises upstream cap to 6.5 Nm when gates pass
- Emergency band at high_0p480: raises upstream cap to 7.0 Nm when gates pass
- Architecture fix activates only when ALL 4 gates pass:
  - Height gate: `schedule_height_ref >= 0.45 m`
  - Band gate: `abs_error >= 0.10 m` (hard or emergency)
  - Safety gate: contact valid, CoM Z >= 0.27 m, roll <= 0.15 rad, pitch <= 0.15 rad
  - Recenter gate: recenter_priority_direct_enabled = True

**Result:** Running (bug fixed)  
**Bug encountered:** `height_cmd` undefined → fixed to `schedule_height_ref` (2 locations)  
**Telemetry:** Pending

---

## Height Variant Setup

**File:** `outputs/physical_target_height_setups/high_0p480_setup.json`

**Configuration:**
- Target CoM Z: 0.480 m
- Hip pitch: 0.6261 rad
- Knee: 1.2234 rad
- Termination floor: 0.431 m (achieved_com_z - 0.05)

---

## Common Parameters

- **Steps:** 1200 (12.0 seconds)
- **Telemetry decimation:** 1 (every step recorded)
- **Failure window:** 1200 steps (full episode)
- **Controller mode:** balance-core
- **Sagittal controller:** velocity-damped

---

## Pass Criteria for Phase 7

T6F_TORQUE_TRANSMISSION_PASS requires ALL of:

1. **arch_fix_active > 0 steps** - Architecture fix activates
2. **effective_max_position_tau_after_arch_fix > 4.0** - Cap raised in hard/emergency
3. **tau_position > 4.0** - Torque > 4.0 Nm after upstream clip in some safe steps
4. **apcr1n_tau_position_after_cap differs from T5** - Intermediate torque path diverges
5. **final_wheel_tau_with_apc differs from T5** - Final wheel torque diverges
6. **No immediate fall** - T6F survives >= 1000 steps
7. **No WBC/hidden/ownership violation** - Safety preserved

---

## Next Steps

After both simulations complete:

1. Copy T6F telemetry to `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t6f_torque_transmission_T6F_1200/`
2. Run `python analyze_t6f_torque_transmission.py`
3. Generate Phase 7D analysis report
4. Classify Phase 7E result
5. If PASS: proceed to Phase 8 (2000-step screening)
6. If FAIL: investigate failure mode

---

**Status:** T5 complete, T6F running (bug fixed)
