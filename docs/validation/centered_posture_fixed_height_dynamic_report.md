# Centered Posture Fixed-Height Dynamic Validation Report

**Date:** 2026-06-19  
**Status:** CENTERED_POSTURE_DYNAMIC_PASS_WITH_MONITORING

---

## Tested Configurations

| Setup | Controller | Steps | Outcome |
|-------|-----------|-------|---------|
| Centered high_0p480 | balance-core | 2000 | ✅ PASS — no fall |
| Centered low_0p380 | balance-core | 2000 | ✅ PASS — no fall |
| Centered low_0p300 | balance-core | 2000 | ✅ PASS — no fall |
| Old high_0p480 | balance-core | 2000 | ✅ PASS — no fall (baseline) |

## Dynamic Metrics Comparison

### Pitch (degrees)

| Height | min | max | mean |
|--------|-----|-----|------|
| Centered high_0p480 | +0.0002 | +3.70 | +3.28 |
| Old high_0p480 | +0.0002 | +3.52 | +3.12 |
| Centered low_0p380 | -0.0000 | +3.29 | +2.78 |
| Centered low_0p300 | -0.0007 | +3.99 | +3.31 |

→ Pitch range similar across all heights. Centered schedule does not introduce pitch instability.

### Roll (degrees)

| Height | min | max | mean |
|--------|-----|-----|------|
| Centered high_0p480 | -0.05 | +0.14 | -0.02 |
| Old high_0p480 | -0.06 | +0.14 | -0.02 |
| Centered low_0p380 | -0.04 | +0.28 | +0.10 |
| Centered low_0p300 | -0.11 | +0.32 | +0.14 |

→ Roll is slightly higher at low heights (intrinsic lateral bias), but well within safe limits (< 0.5 deg). No roll divergence observed.

### Yaw Drift (rad)

| Height | min | max |
|--------|-----|-----|
| Centered high_0p480 | -0.046 | +0.023 |
| Old high_0p480 | -0.046 | +0.023 |
| Centered low_0p380 | -0.052 | +0.022 |
| Centered low_0p300 | -0.037 | +0.013 |

→ Yaw drift comparable across all configurations. Lateral bias does not cause yaw divergence.

### Height Error (m)

| Height | min | max | mean |
|--------|-----|-----|------|
| Centered high_0p480 | 0.000 | +0.010 | +0.008 |
| Old high_0p480 | 0.000 | +0.010 | +0.008 |
| Centered low_0p380 | -0.001 | +0.003 | +0.003 |
| Centered low_0p300 | -0.006 | 0.000 | -0.002 |

→ Height tracking comparable. All within ±1 cm.

## Pass/Fail Summary

| Criteria | Result |
|----------|--------|
| No fall | ✅ PASS |
| No structural violation | ✅ PASS |
| Pitch bounded | ✅ PASS |
| Roll bounded | ✅ PASS |
| Contact validity | ✅ PASS |
| Hip-yaw stable | ✅ PASS |

## Conclusion

The centered posture schedule **does not regress** dynamic behavior compared to the old setups. All centered heights run successfully through 2000-step balance-core simulations. The lateral CoM bias (intrinsic to squat geometry) manifests as slightly higher roll at low heights but remains well within safe limits (< 0.5 deg at low_0p300).

Classification: **CENTERED_POSTURE_DYNAMIC_PASS_WITH_MONITORING**
