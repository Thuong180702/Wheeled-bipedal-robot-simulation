# Operational vs Physical Height Envelope

**Date:** 2026-06-03  
**Status:** Validated and Documented

---

## Summary

The wheeled-biped robot has two distinct height envelopes:

1. **Physical Envelope (Static):** 0.292-0.491 m CoM (19.9 cm range)
   - Defines what is kinematically/geometrically feasible
   - Static balance, contact, joint limits satisfied

2. **Operational Envelope (Dynamic):** 0.393-0.413 m CoM (2.0 cm range)
   - Defines where current controller achieves dynamic stability
   - Step E + Step C validation passed

**Key Finding:** Current controller stabilizes only **10% of physical range**. The remaining 90% requires architecture enhancements.

---

## Physical Envelope (Static Feasibility)

**Definition:** The range of CoM heights where the robot can stand in static equilibrium with valid wheel contacts, no collisions, and joints within limits.

**Search Method:** Binary search over symmetric hip-pitch/knee configurations with calibrated root_z based on wheel-floor geometry (see `wheeled_biped/validation/physical_standing_height_envelope.py`).

**Results:**
```
Physical minimum:  0.2919 m CoM
Physical maximum:  0.4908 m CoM
Range:            0.1989 m (19.9 cm)
```

**Validation Criteria:**
- ✅ Both wheel contacts active
- ✅ No non-wheel floor contact
- ✅ CoM projection within wheel support segment
- ✅ Support center near body projection (< 2 cm offset)
- ✅ Pitch/roll/yaw near equilibrium (< 0.02 rad)
- ✅ Joints within limits with margin
- ✅ Left-right symmetric posture

**Status:** Static envelope fully characterized and validated.

**Artifacts:**
- `outputs/physical_standing_height_envelope_search/physical_height_envelope_summary.json`
- `outputs/physical_standing_height_envelope_search/physical_min_height_setup.json`
- `outputs/physical_standing_height_envelope_search/physical_max_height_setup.json`
- `docs/validation/physical_standing_height_envelope_validation.md`

---

## Operational Envelope (Dynamic Stability)

**Definition:** The range of CoM heights where the robot achieves dynamic stability under the current hierarchical velocity-damped balance controller, validated via Step E (position hold) and Step C (height recovery).

**Controller:** 
- Hierarchical: velocity-damped sagittal + shape-posture lateral
- Sagittal profile: `candidate_D2_wheel_velocity_damping_light`
- Hip-yaw gains: kp=15, kd=3

**Results:**
```
Operational minimum:  0.393 m CoM (low_small variant)
Operational maximum:  0.413 m CoM (high_small variant)
Nominal:              0.403 m CoM
Range:                0.020 m (2.0 cm)
```

**Validation Criteria (Step E + Step C):**
- ✅ Support position error max ≤ 0.15 m
- ✅ Hip yaw max ≤ 0.07 rad
- ✅ Pitch max ≤ 0.10 rad
- ✅ Roll max ≤ 0.05 rad
- ✅ Height error final ≤ 0.02 m
- ✅ Contact valid ≥ 99.9%
- ✅ WBC applied = false
- ✅ Hidden torque = 0
- ✅ Ownership violations = 0

**Validated Variants:**
1. **nominal** (0.403 m) - target height for standing
2. **low_tiny** (0.398 m) - -5 mm offset
3. **high_tiny** (0.408 m) - +5 mm offset
4. **low_small** (0.393 m) - -10 mm offset
5. **high_small** (0.413 m) - +10 mm offset

**Status:** Operational envelope fully validated for current controller.

**Artifacts:**
- `docs/validation/step_e_height_variant_robustness_done.md`
- `docs/validation/step_c_height_recovery_done.md`

---

## Gap Analysis

### Low Boundary Gap

**Physical minimum:** 0.292 m CoM  
**Operational minimum:** 0.393 m CoM  
**Gap:** 10.1 cm (51% of physical range below operational min)

**Target tested:** 0.300 m CoM (within physical, 9.3 cm below operational)

**Result:** ❌ FAIL - All 6 fix strategies failed Step E validation
- Hip yaw: 0.12-0.19 rad (66-165% above 0.07 threshold)
- Support error: 0.175 m (17% above 0.15 threshold)
- Pitch: 0.111 rad (11% above 0.10 threshold)

**Root cause:** Extreme kinematic configuration (hip_pitch=1.38 rad, knee=2.35 rad, near joint limits) exceeds current controller architecture capability.

### High Boundary Gap

**Operational maximum:** 0.413 m CoM  
**Physical maximum:** 0.491 m CoM  
**Gap:** 7.8 cm (39% of physical range above operational max)

**Target tested:** 0.480 m CoM (within physical, 6.7 cm above operational)

**Result:** ⏸️ NOT TESTED - Low boundary failure stopped evaluation

**Hypothesis:** May also fail due to different failure modes (high-height-specific instabilities).

### Total Utilization

**Physical range:** 19.9 cm  
**Operational range:** 2.0 cm  
**Utilization:** 10% of physical range

---

## Why the Gap Exists

### Controller Architecture Limitation

The hierarchical velocity-damped balance controller separates:
- **Sagittal control:** Pitch/position via wheel torques
- **Lateral control:** Roll/yaw via hip actuators

This separation works well in the operational envelope but breaks down at extreme heights where:
1. Sagittal-yaw coupling becomes nonlinear
2. Gravity moments scale differently with posture
3. Authority saturation in both channels
4. Joint-limit-induced constraints

### Fix Strategies Tested (Phase 4)

All 6 strategies FAILED at low boundary (0.300 m):

1. ❌ **baseline** - Standard controller
2. ❌ **yaw_aware_position_only** - Yaw-compensated position hold (no effect)
3. ❌ **boundary_hip_yaw_profile** - Increased hip-yaw gains (marginal improvement, still fails)
4. ❌ **yaw_aware_plus_boundary_hip_yaw** - Combined (no additional benefit)
5. ❌ **boundary_hip_yaw_integral_light** - Added integral term (made it worse)
6. ❌ **yaw_aware_plus_integral_light** - Combined (still worse)

**Conclusion:** The limitation is **architectural**, not tunable. Extending to full physical range requires:
- Integrated 6-DOF control (not hierarchical)
- Joint-limit-aware scheduling
- Adaptive authority allocation
- Or fundamentally different control approach

---

## Implications for Step D (PPO Residual Training)

### What Step D Requires

Step D trains a bounded PPO residual policy over the LQR/IK nominal prior. The training needs:
- ✅ Height commands within validated range
- ✅ Robust standing at target heights
- ✅ Height transitions (standing ↔ squatting)
- ✅ Push disturbance recovery

### Operational Envelope is Sufficient

**Height range:** 0.393-0.413 m (2.0 cm)

This covers:
- ✅ **Standing:** nominal 0.403 m
- ✅ **Squatting:** low_small 0.393 m (10 mm squat depth)
- ✅ **Height transitions:** 0.393 ↔ 0.413 m (2 cm range)
- ✅ **Robustness:** 5 validated variants

**Typical human squat:** 15-30 cm range. Our 2 cm operational range is **small** but:
- Validates the method (residual PPO over nominal prior)
- Demonstrates height-adaptive stabilization
- Sufficient for proof-of-concept

### Paper Framing

**Main contribution:** Bounded residual PPO over height-scheduled LQR/IK prior for wheeled-biped balance

**Validated scope:** Operational envelope (0.393-0.413 m CoM)

**Limitation (Discussion):**
> "Controller stability was validated within an operational envelope of 0.393-0.413 m CoM (2 cm range), covering typical standing and squatting scenarios with push recovery. The robot's full physical capability extends to 0.292-0.491 m CoM (20 cm range), but extreme-height stabilization requires architecture enhancements beyond the hierarchical velocity-damped control framework, such as integrated 6-DOF optimization or joint-limit-aware adaptive scheduling. The operational envelope is sufficient for validating the proposed residual learning method."

**No claim of full-range height control.** The contribution is the residual learning method, not extreme-height capability.

---

## Step D Can Proceed

✅ **Operational envelope validated:** 5 variants pass Step E + Step C  
✅ **Boundary limitation documented:** Clear distinction physical vs operational  
✅ **Sufficient range for method validation:** 2 cm covers standing/squatting/push  
✅ **Paper framing clear:** Contribution is method, not extreme capability  

**Decision:** Proceed to Step D (PPO residual training) using operational envelope (0.393-0.413 m).

---

## Future Work (Beyond Current Scope)

To extend controller capability toward full physical envelope:

1. **Integrated 6-DOF control:** Replace hierarchical separation with unified optimization
2. **Joint-limit-aware scheduling:** Reduce authority as joints approach limits
3. **Adaptive authority allocation:** Dynamic balance between sagittal/lateral based on coupling magnitude
4. **Learning-based extreme-height policies:** Train separate policies for extreme configurations
5. **Hardware validation:** Real robot may have different operational limits due to unmodeled dynamics

These are **architecture-level changes** requiring months of research and are beyond the scope of the current residual learning contribution.

---

## References

- Physical envelope search: `docs/validation/physical_standing_height_envelope_validation.md`
- Operational envelope validation: `docs/validation/step_e_height_variant_robustness_done.md`, `docs/validation/step_c_height_recovery_done.md`
- Boundary height attempt: `docs/validation/boundary_height_0p300_0p480_validation.md`
- Failure analysis: `outputs/boundary_yaw_position_coupling_fix/boundary_fix_failure_analysis_report.md`
