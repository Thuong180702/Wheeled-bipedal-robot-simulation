# APCR1d Phase 8 Failure Audit

## Classification: `APCR1D_2000_HARNESS_MISMATCH`

## Summary

The APCR1d 2000-step failure at step 17-18 is **NOT a true controller failure**. The failed run used a different harness configuration than the successful 500-step run.

---

## Evidence

### Critical Difference: sagittal_schedule_profile

| Run | sagittal_schedule_profile | Result |
|-----|---------------------------|--------|
| Success (500-step) | `APCR1d_symmetric_soft_band_control` | Survived 500 steps |
| Failed (2000-step) | `baseline` | Failed at step 17 |

**The failed run did NOT use APCR1d profile.** It used the `baseline` profile which has no active pitch crossing authority.

### Critical Difference: contact_force_valid at Step 0

| Run | contact_force_valid (step 0) | contact_supervisor_state |
|-----|-------------------------------|--------------------------|
| Success (500-step) | `False` | `double_contact` |
| Failed (2000-step) | `True` | `None` |

The failed run started with valid contact forces while the success run had invalid contact forces (which is expected for the first step with force validation).

### Pitch Divergence Timeline

| Step | Success pitch_x | Failed pitch_x | Divergence |
|------|-----------------|----------------|------------|
| 0 | -0.000000 | -0.000000 | Identical |
| 1 | +0.000046 | -0.003428 | Started |
| 2 | +0.000248 | -0.014865 | Growing |
| 3 | -0.000642 | -0.035011 | Growing |
| 4 | -0.000592 | -0.059698 | Growing |
| 5 | -0.000277 | -0.086593 | Growing |
| 6 | -0.000420 | -0.116171 | Growing |
| 7 | -0.000335 | -0.149489 | Growing |
| 8 | -0.000027 | -0.187611 | Growing |
| 9 | +0.000160 | -0.231653 | Growing |
| 10 | +0.000404 | -0.282623 | Growing |
| 11 | +0.000781 | -0.338970 | Growing |
| 12 | +0.001162 | -0.398277 | Growing |
| 13 | +0.001572 | -0.458382 | Growing |
| 14 | +0.002062 | -0.518757 | Growing |
| 15 | +0.002592 | -0.579335 | Growing |
| 16 | +0.003157 | -0.655974 | Growing |
| 17 | +0.003779 | -0.748812 | **Failed at step 17** |

### CoM Height Divergence

| Step | Success com_z | Failed com_z | Divergence |
|------|----------------|--------------|------------|
| 0 | 0.295450 | 0.295476 | +0.000026 |
| 5 | 0.294124 | 0.293652 | -0.000472 |
| 10 | 0.293568 | 0.283981 | -0.009587 |
| 15 | 0.293347 | 0.254718 | -0.038629 |
| 17 | 0.293302 | 0.240288 | **-0.053014** |

The failed run loses height rapidly because it has no APCR authority to correct pitch.

---

## Root Cause

The failed 2000-step run was executed with `--vd-sagittal-authority-profile baseline` (or the default was `baseline` instead of `APCR1d_symmetric_soft_band_control`).

Without APCR authority, the robot:
1. Accumulates negative pitch (falling backward)
2. CoM drops below height floor
3. Terminates at step 17

The success run correctly used `APCR1d_symmetric_soft_band_control` profile which provides proportional soft-band correction authority.

---

## Files Analyzed

- `outputs/hierarchical_controller_sim/telemetry_1780973047.csv` - Success run (500 steps, APCR1d profile)
- `outputs/hierarchical_controller_sim/telemetry_1780974140.csv` - Failed run (18 steps, baseline profile)

---

## Conclusion

**APCR1d 2000-step does NOT fail due to the controller.** The failure was caused by a harness configuration mismatch where the wrong profile was used.

**The APCR1d profile is NOT broken.** The 500-step validation confirms it works correctly.

**The failed run was a baseline run, not an APCR1d run.**

---

## Recommendations

1. **Re-run APCR1d 2000-step** with correct `--vd-sagittal-authority-profile APCR1d_symmetric_soft_band_control` argument
2. **Verify harness** always passes the correct profile name
3. **Do NOT tune APCR1d** based on this failed run - it was a harness error
4. **Proceed to Phase 2** for controlled reproducibility check with identical arguments
