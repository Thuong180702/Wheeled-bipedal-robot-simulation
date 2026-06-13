# Event Order Audit Report

**Date:** 2026-06-05
**Phase:** Phase 4

## Summary Table

| Variant | Hip-Yaw 0.03 | Hip-Yaw 0.07 | Hip-Yaw 0.10 | Support 0.05 | Support 0.10 | Roll 0.05 | Classification |
|---------|--------------|--------------|--------------|--------------|-------------|-----------|----------------|
| low_0p300 | 273 | 552 | 699 | - | - | - | hip_yaw_0.10 |
| nominal | 464 | - | - | - | - | - | no_significant_event |
| high_0p480 | 716 | 1629 | 2258 | - | - | - | hip_yaw_0.10 |

## Analysis

### low_0p300

**Classification:** hip_yaw_0.10

**First events:**
- hip_yaw_0.10: step 699 (6.99s)
- pitch_0.15: step 783 (7.83s)

**Detailed events:**
- Hip-Yaw 0.03 rad: step 273 (2.73s)
- Hip-Yaw 0.07 rad: step 552 (5.52s)
- Hip-Yaw 0.10 rad: step 699 (6.99s)
- Hip-Yaw 0.15 rad: step 1443 (14.43s)
- Hip-Yaw 0.20 rad: step 2320 (23.20s)
- Hip-Yaw 0.25 rad: step 3634 (36.34s)
- Support 0.05 m: not reached
- Support 0.10 m: not reached
- Support 0.15 m: not reached
- Roll 0.05 rad: not reached
- Pitch 0.10 rad: step 50 (0.50s)
- Pitch 0.15 rad: step 783 (7.83s)

### nominal

**Classification:** no_significant_event

**First events:**

**Detailed events:**
- Hip-Yaw 0.03 rad: step 464 (4.64s)
- Hip-Yaw 0.07 rad: not reached
- Hip-Yaw 0.10 rad: not reached
- Hip-Yaw 0.15 rad: not reached
- Hip-Yaw 0.20 rad: not reached
- Hip-Yaw 0.25 rad: not reached
- Support 0.05 m: not reached
- Support 0.10 m: not reached
- Support 0.15 m: not reached
- Roll 0.05 rad: not reached
- Pitch 0.10 rad: not reached
- Pitch 0.15 rad: not reached

### high_0p480

**Classification:** hip_yaw_0.10

**First events:**
- hip_yaw_0.10: step 2258 (22.58s)

**Detailed events:**
- Hip-Yaw 0.03 rad: step 716 (7.16s)
- Hip-Yaw 0.07 rad: step 1629 (16.29s)
- Hip-Yaw 0.10 rad: step 2258 (22.58s)
- Hip-Yaw 0.15 rad: step 3146 (31.46s)
- Hip-Yaw 0.20 rad: step 3986 (39.86s)
- Hip-Yaw 0.25 rad: step 4860 (48.60s)
- Support 0.05 m: not reached
- Support 0.10 m: not reached
- Support 0.15 m: not reached
- Roll 0.05 rad: not reached
- Pitch 0.10 rad: not reached
- Pitch 0.15 rad: not reached

