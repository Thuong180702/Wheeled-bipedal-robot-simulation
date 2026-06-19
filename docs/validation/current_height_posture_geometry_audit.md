# Current Height Posture Geometry Audit

**Date:** 2026-06-19  
**Analyst:** Claude  
**Status:** COMPLETE

---

## Key Finding

**Static sagittal CoM-x is already centered for all 10 heights.**

The previous hypothesis that `pitch_ref_offset` compensates a static sagittal CoM-x bias is **wrong**. The pitch offset actually compensates a dynamic control coupling between `tau_pitch` and `tau_position` under the active controller — not a static posture geometry error.

## Updated Classification

```
CURRENT_POSTURE_GEOMETRY_HEIGHT_DEPENDENT_BIAS
```

### Sub-classifications

| Aspect | Finding | Code |
|--------|---------|------|
| Sagittal CoM-x | Centered (10/10) | `STATIC_SAGITTAL_COM_CENTERED_DYNAMIC_CONTROL_COUPLING` |
| Lateral CoM-y | Biased (6/10) | `LATERAL_COM_BIAS_AND_NONSMOOTH_POSTURE_SCHEDULE` |
| Hip/knee schedule | Non-monotonic (2 transitions) | — |

---

## Per-Height Posture Summary

| Height | hip_pitch | knee | com_err_x | com_err_y | class |
|--------|-----------|------|-----------|-----------|-------|
| low_0p300 | 1.3761 | 2.3484 | 3.6e-07 | +0.01255 | LATERAL_BIASED |
| low_0p320 | 1.1511 | 2.1984 | 5.0e-07 | -0.01055 | LATERAL_BIASED |
| low_0p330 | 1.0761 | 2.1234 | 5.6e-07 | -0.01704 | LATERAL_BIASED, HEIGHT_MISMATCH |
| low_0p340 | 1.1511 | 2.1234 | 4.9e-07 | -0.00076 | CENTERED_OK |
| low_0p360 | 1.0011 | 1.9734 | 6.3e-07 | -0.01470 | LATERAL_BIASED |
| low_0p380 | 1.0761 | 1.8984 | 5.2e-07 | +0.01571 | LATERAL_BIASED |
| high_0p430 | 0.8511 | 1.5984 | 8.5e-07 | +0.00363 | CENTERED_OK |
| high_0p450 | 0.7761 | 1.4484 | 1.0e-06 | +0.00750 | LATERAL_BIASED |
| high_0p465 | 0.7011 | 1.3734 | 1.2e-06 | -0.00336 | CENTERED_OK |
| high_0p480 | 0.6261 | 1.2234 | 1.4e-06 | -0.00011 | CENTERED_OK |

### Centered heights (abs(com_err_y) <= 0.005m): low_0p340, high_0p430, high_0p465, high_0p480
### Biased heights: low_0p300, low_0p320, low_0p330, low_0p360, low_0p380, high_0p450

---

## Hip/Knee Schedule Non-Monotonicity

| Transition | dh | dhip_pitch | dknee | Issue |
|-----------|-----|-----------|-------|-------|
| 0.300→0.320 | +0.020 | -0.225 | -0.150 | — |
| 0.320→0.330 | +0.010 | -0.075 | -0.075 | — |
| **0.330→0.340** | **+0.010** | **+0.075** | **0.000** | **hip_pitch UP as height increases** |
| 0.340→0.360 | +0.020 | -0.150 | -0.150 | — |
| **0.360→0.380** | **+0.020** | **+0.075** | **-0.075** | **hip_pitch UP as height increases** |
| 0.380→0.430 | +0.050 | -0.225 | -0.300 | — |
| 0.430→0.450 | +0.020 | -0.075 | -0.150 | — |
| 0.450→0.465 | +0.015 | -0.075 | -0.075 | — |
| 0.465→0.480 | +0.015 | -0.075 | -0.150 | — |

Two non-monotonic transitions, both caused by the coarse 0.075-rad grid step in the search.

---

## Static Feasibility

| Check | Result |
|-------|--------|
| All static_feasible | PASS (10/10) |
| Both wheel contacts | PASS (10/10) |
| No non-wheel contact | PASS (10/10) |
| Joint margin safety | PASS (all >= 0.35 rad) |
| Pitch/Roll/Yaw near zero | PASS (all 0.0 rad) |

---

## Revised Root-Cause Model

```
Static CoM-x centered (10/10)
         ↓
tau_pitch still biased during active control
         ↓
Reason: dynamic control coupling, not static geometry
         ↓
Previous pitch_ref_offset compensates dynamic tau_pitch bias,
not static CoM-x error
         ↓
Separate issues for posture optimizer:
  (1) Lateral CoM-y bias (6/10 heights)
  (2) Non-smooth hip/knee height schedule
```

---

## Artifacts

- `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/current_height_posture_geometry_audit.csv` — per-height metrics
- `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/current_height_posture_geometry_audit.json` — structured audit

---

## Next Steps

Proceed to Phase 2: Design centered posture optimizer focusing on:
1. Lateral CoM centering via small hip_roll adjustments
2. Smooth monotonic hip_pitch_ref(height) and knee_ref(height) functions
3. Preserve sagittal centering
4. Keep static feasibility
