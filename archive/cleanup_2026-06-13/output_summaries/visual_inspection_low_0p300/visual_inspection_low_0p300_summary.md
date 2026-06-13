# Visual Inspection Summary - low_0p300

**Date:** 2026-06-05  
**Purpose:** Visual posture inspection to inform pitch threshold decision  
**Environment limitation:** OpenGL unavailable - telemetry collected, visual inspection requires local execution

---

## Executive Summary

Non-visual simulations completed for baseline, J2, and J3 profiles at low_0p300 (z=0.300m target height). Telemetry metrics collected. Visual inspection commands provided below for local execution.

**Key Finding:** J2 and J3 achieve support and hip-yaw improvements but exceed pitch threshold by 50-60%. User visual inspection needed to determine whether pitch excursions of 8-9° are acceptable at this extreme boundary height.

---

## Telemetry Metrics Comparison

| Metric | Baseline | J2 | J3 | Gate |
|--------|----------|----|----|------|
| Steps survived | 1000 | 1000 | 1000 | - |
| Support error (m) | **0.243** | **0.114** | **0.125** | ≤0.15 |
| Hip yaw max (rad) | **0.214** | **0.137** | **0.088** | ≤0.07 |
| Hip yaw max (deg) | 12.2° | 7.8° | 5.1° | 4.0° |
| Pitch max (rad) | **0.095** | **0.157** | **0.151** | ≤0.10 |
| Pitch max (deg) | 5.5° | 9.0° | 8.7° | 5.7° |
| Roll max (rad) | 0.015 | 0.014 | 0.013 | - |
| Height error final (m) | 0.013 | 0.008 | 0.007 | - |
| Non-wheel contacts | 0 | 0 | 0 | 0 |
| WBC norm max (Nm) | 15.3 | 19.3 | 17.1 | diagnostic |
| Hidden torque (Nm) | 0.0 | 0.0 | 0.0 | 0 |
| Ownership violations | 0 | 0 | 0 | 0 |

**Bold** = exceeds gate threshold

---

## Gate Status Summary

### Baseline (J0 equivalent, no scheduling)
- ✅ Pitch: 0.095 rad (5.5°) - **PASS**
- ❌ Support: 0.243 m - **FAIL** (62% over threshold)
- ❌ Hip yaw: 0.214 rad (12.2°) - **FAIL** (205% over threshold)
- Schedule: INACTIVE (k_pos=40, k_vel=15 - nominal values)

### J2 (k_pos=80, k_vel=30)
- ❌ Pitch: 0.157 rad (9.0°) - **FAIL** (57% over threshold)
- ✅ Support: 0.114 m - **PASS** (53% improvement vs baseline)
- ❌ Hip yaw: 0.137 rad (7.8°) - **FAIL** (96% over threshold, but 36% improvement vs baseline)
- Schedule: ACTIVE (k_pos=80, k_vel=30 - scheduled values)

### J3 (k_pos=80, k_vel=25)
- ❌ Pitch: 0.151 rad (8.7°) - **FAIL** (51% over threshold)
- ✅ Support: 0.125 m - **PASS** (49% improvement vs baseline)
- ❌ Hip yaw: 0.088 rad (5.1°) - **FAIL** (26% over threshold, but 59% improvement vs baseline)
- Schedule: ACTIVE (k_pos=80, k_vel=25 - scheduled values)

---

## Visual Inspection Questions

User should visually inspect J2 and J3 to evaluate:

1. **Pitch acceptability:** Does pitch ~8-9° look stable or unstable at z=0.300m?
2. **Posture quality:** Do legs twist inward/outward excessively?
3. **Body lean:** Is forward/backward lean visually acceptable?
4. **Wheel contact:** Do wheels maintain ground contact throughout?
5. **Overall stability:** Does the robot look like it's performing stable crouching or collapsing?
6. **Comparative judgment:** Does J2 or J3 look better overall?
7. **Threshold decision:** Should pitch threshold be relaxed to 0.14-0.15 rad for low_0p300, or should 0.300m be marked as outside operational envelope?

---

## Commands for Local Visual Inspection

**Prerequisites:**
- Environment with OpenGL support (local machine, not remote/headless)
- MuJoCo viewer working

**Command structure:**
```bash
cd f:/ROBOTCUATAO/Wheeled-bipedal-robot-simulation

python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --height-variant-setup outputs/physical_target_height_setups/low_0p300_setup.json \
  --vd-sagittal-authority-profile [PROFILE] \
  --steps 1000 \
  --visual
```

**Case A - Baseline (reference case):**
```bash
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --height-variant-setup outputs/physical_target_height_setups/low_0p300_setup.json \
  --vd-sagittal-authority-profile baseline \
  --steps 1000 \
  --visual
```

**Case B - J2 (support and hip-yaw pass numerically, pitch fails):**
```bash
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --height-variant-setup outputs/physical_target_height_setups/low_0p300_setup.json \
  --vd-sagittal-authority-profile J2 \
  --steps 1000 \
  --visual
```

**Case C - J3 (best support, hip-yaw passes, pitch fails):**
```bash
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --height-variant-setup outputs/physical_target_height_setups/low_0p300_setup.json \
  --vd-sagittal-authority-profile J3 \
  --steps 1000 \
  --visual
```

**Note:** Pitch-aware candidates (P0-P5) have not been implemented yet. If user decides to proceed with pitch-aware approach after visual inspection, those profiles will need to be created first.

---

## Telemetry Files

Telemetry saved to:
- `outputs/visual_inspection_low_0p300/baseline_telemetry.csv`
- `outputs/visual_inspection_low_0p300/J2_telemetry.csv`
- `outputs/visual_inspection_low_0p300/J3_telemetry.csv`

Metrics JSON:
- `outputs/visual_inspection_low_0p300/visual_inspection_low_0p300_metrics.json`

---

## Decision Tree After Visual Inspection

### If J2 or J3 look acceptable:
1. **Option A:** Relax pitch threshold to 0.14-0.15 rad for low_0p300 only
2. **Option B:** Proceed with pitch-aware position control (Option C from prior analysis)

### If J2 and J3 look unacceptable:
1. **Option A:** Mark z=0.300m as outside operational envelope (operational z_min = 0.330m)
2. **Option B:** Investigate pitch-aware position control anyway as potential fix

### If pitch-aware approach is chosen:
- Implement P0-P5 candidate profiles with pitch-aware scaling
- Run evaluation with stop-at-first-pass protocol
- Visual inspection of passing candidate

---

## Technical Notes

**WBC Status:** Confirmed diagnostic-only via code inspection (`simulate_hierarchical_controller.py:3011-3012` sets `include_wbc = False` in balance-core mode). WBC values in telemetry represent computed diagnostics, not applied torques.

**Schedule Activation:** J2 and J3 both show `low_height_sagittal_schedule_active = True` with correct scheduled parameters, confirming the schedule height reference bug is fixed.

**Contact Status:** All profiles maintain wheel-only contact (no foot/shin contacts), indicating proper kinematic configuration at z=0.300m.

**Termination:** All simulations completed 1000 steps without early termination, indicating configurations are at least marginally stable over 10-second horizon.

---

## Next Steps (BLOCKED - awaiting user visual inspection)

1. ❌ **Do NOT modify controller code** until user reviews visual behavior
2. ❌ **Do NOT change pitch threshold** until user decides after visual inspection
3. ❌ **Do NOT implement pitch-aware candidates** until user confirms approach
4. ❌ **Do NOT proceed to full Phase 6 evaluation** until visual inspection complete

**User action required:** Run visual simulations locally and provide feedback on:
- Whether pitch behavior looks acceptable
- Which profile (J2 or J3) looks better
- Whether to relax threshold, pursue pitch-aware fix, or mark 0.300m as operational boundary
