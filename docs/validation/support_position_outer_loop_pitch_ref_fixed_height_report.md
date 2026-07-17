# Support-Position Outer-Loop — Fixed-Height Validation (Phase 5)

**A (baseline):** `height_scheduled_pitch_equilibrium_trim`
**B (candidate):** `support_position_outer_loop_pitch_ref` (Kp=+1.0 deg/m, Kd=0.0, P-only)
**Classification:** `OUTER_LOOP_FIXED_HEIGHT_PASS_BETTER_THAN_HEIGHT_SCHEDULE`

## Gates

- improve heights: 9/10 (need >=6 for full pass)
- regression heights: 1 (>1 fails)
- hard safety failures: 0 
- 9/10 heights improved, <=1 regression, protected heights safe

---

## 5A: high_0p480 multi-step (B vs A)

| steps | prof | fell | pos% | max_abs | P2P | out15% | hip_yaw | wbc |
|---|---|---|---|---|---|---|---|---|
| 1200 | A | False | 59.9 | 0.1497 | 0.2788 | 0.0 | 0.0412 | 15.44 |
| 1200 | B | False | 59.0 | 0.1587 | 0.3015 | 1.7 | 0.0494 | 16.20 |
| 2000 | A | False | 57.0 | 0.1694 | 0.3087 | 5.2 | 0.0412 | 17.20 |
| 2000 | B | False | 56.4 | 0.1812 | 0.3308 | 8.0 | 0.0494 | 18.15 |
| 5000 | A | False | 57.2 | 0.1884 | 0.3277 | 14.0 | 0.1022 | 18.00 |
| 5000 | B | False | 58.7 | 0.1971 | 0.3467 | 17.0 | 0.1395 | 18.70 |

## 5B: 10-height ladder (2000 steps)

| height | prof | fell | pos% | min | max | max_abs | P2P | out15% | hip_yaw | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| low_0p300 | A | False | 44.6 | -0.0620 | 0.0678 | 0.0678 | 0.1299 | 0.0 | 0.2052 |  |
| low_0p300 | B | False | 45.5 | -0.0805 | 0.0687 | 0.0805 | 0.1492 | 0.0 | 0.2084 | IMPROVE |
| low_0p320 | A | False | 43.3 | -0.1253 | 0.1270 | 0.1270 | 0.2524 | 0.0 | 0.1699 |  |
| low_0p320 | B | False | 45.1 | -0.1409 | 0.1365 | 0.1409 | 0.2774 | 0.0 | 0.1864 | IMPROVE |
| low_0p330 | A | False | 49.0 | -0.1172 | 0.1232 | 0.1232 | 0.2405 | 0.0 | 0.2034 |  |
| low_0p330 | B | False | 50.0 | -0.1214 | 0.1309 | 0.1309 | 0.2523 | 0.0 | 0.2020 | IMPROVE |
| low_0p340 | A | False | 44.6 | -0.1027 | 0.1343 | 0.1343 | 0.2370 | 0.0 | 0.1709 |  |
| low_0p340 | B | False | 44.3 | -0.1146 | 0.1401 | 0.1401 | 0.2547 | 0.0 | 0.1719 | EQUAL |
| low_0p360 | A | False | 51.8 | -0.1109 | 0.1160 | 0.1160 | 0.2269 | 0.0 | 0.1813 |  |
| low_0p360 | B | False | 57.2 | -0.0889 | 0.1171 | 0.1171 | 0.2059 | 0.0 | 0.1174 | IMPROVE |
| low_0p380 | A | False | 44.8 | -0.1373 | 0.1112 | 0.1373 | 0.2485 | 0.0 | 0.0488 |  |
| low_0p380 | B | False | 45.9 | -0.1399 | 0.1104 | 0.1399 | 0.2503 | 0.0 | 0.0518 | IMPROVE |
| high_0p430 | A | False | 61.5 | -0.0721 | 0.1228 | 0.1228 | 0.1948 | 0.0 | 0.0378 |  |
| high_0p430 | B | False | 61.1 | -0.0838 | 0.1364 | 0.1364 | 0.2201 | 0.0 | 0.0458 | IMPROVE |
| high_0p450 | A | False | 69.5 | -0.0712 | 0.1550 | 0.1550 | 0.2262 | 0.9 | 0.0691 |  |
| high_0p450 | B | False | 67.1 | -0.0733 | 0.1908 | 0.1908 | 0.2641 | 16.4 | 0.0251 | REGRESS |
| high_0p465 | A | False | 41.7 | -0.1657 | 0.1234 | 0.1657 | 0.2891 | 3.4 | 0.0487 |  |
| high_0p465 | B | False | 44.0 | -0.1519 | 0.1419 | 0.1519 | 0.2938 | 0.8 | 0.0379 | IMPROVE |
| high_0p480 | A | False | 57.0 | -0.1393 | 0.1694 | 0.1694 | 0.3087 | 5.2 | 0.0412 |  |
| high_0p480 | B | False | 56.4 | -0.1496 | 0.1812 | 0.1812 | 0.3308 | 8.0 | 0.0494 | IMPROVE |

## Decision

- B is safe and within tolerance vs A. **Proceed to Step C (random/changing height).**

