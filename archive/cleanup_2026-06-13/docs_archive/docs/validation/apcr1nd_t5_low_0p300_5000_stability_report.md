# APCR1nD T5 Low 0.300m 5000-Step Stability Report

**Date:** 2026-06-12  
**Profile:** APCR1nD_T5_band_limited_balanced  
**Classification:** PASS

---

## Survival

- **Steps:** 5000/5000 (100%)
- **Terminated:** No
- **Status:** Completed successfully

---

## Contact Stability

| Metric | Value |
|--------|-------|
| Contact % | 100.0% |
| Double contact % | 99.4% |
| Single contact | 0.6% |
| No contact | 0.0% |

**Status:** ✅ Stable

---

## Height Stability

| Metric | Value |
|--------|-------|
| CoM Z min | 0.274 m |
| CoM Z mean | 0.282 m |
| CoM Z max | 0.295 m |
| Target | 0.300 m |
| Height error max | 0.022 m |
| Height error mean | 0.013 m |
| Height error final | -0.021 m |

**Status:** ✅ Stable

---

## Attitude Stability

| Metric | Value |
|--------|-------|
| Pitch min | -0.008 deg |
| Pitch max | 0.136 deg |
| **Pitch RMS** | **0.049 deg** |
| Roll min | 0.000 deg |
| Roll max | 0.015 deg |
| **Roll RMS** | **0.0047 deg** |

**Status:** ✅ Excellent

---

## Wheel Velocity

| Metric | Value |
|--------|-------|
| Max | 4.39 rad/s |
| **RMS** | **1.24 rad/s** |
| Steps > 5 rad/s | 0 |
| Steps > 6 rad/s | 0 |
| Steps > 7 rad/s | 0 |
| % > 5 rad/s | 0.0% |

**Status:** ✅ Safe (no spikes)

---

## Structural Integrity

| Metric | Value |
|--------|-------|
| Hidden torque max | 0.0 Nm |
| Ownership violation max | 0 |
| WBC violations | 0 |

**Status:** ✅ No violations

---

## Comparison to APCR1n Baseline

| Metric | APCR1n | T5 | Improvement |
|--------|--------|----|----|
| Pitch RMS | 3.52 deg | 0.049 deg | **-98.6%** |
| Roll RMS | 0.268 deg | 0.0047 deg | **-98.2%** |
| Wheel RMS | 2.41 rad/s | 1.24 rad/s | **-48.5%** |
| Wheel max | 4.77 rad/s | 4.39 rad/s | -8.0% |

**T5 significantly smoother and more stable.**

---

## Conclusion

✅ All stability gates passed  
✅ Contact maintained  
✅ Height stable  
✅ Attitude excellent  
✅ Wheel velocity safe  
✅ No structural violations  
✅ Ready for high_0p480
