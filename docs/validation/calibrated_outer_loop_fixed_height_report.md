# Calibrated Outer-Loop — Fixed-Height Validation (Phase 6)

**A:** `height_scheduled_pitch_equilibrium_trim`
**B:** `support_position_outer_loop_pitch_ref` (Kp=+1.0, Kd=0.0)
**B2:** `calibrated_support_position_outer_loop_pitch_ref` (height-varying Kp/Kd)
**Classification:** `CALIBRATED_OUTER_LOOP_V2_NOT_BETTER`

## Gates

- improve heights (B2 vs B by score): 6/10 (need >=6 for full pass)
- regression heights (B2 vs B by score): 2/10 (>2 fails)
- hard safety failures: 0 
- protected height regression: ['high_0p480']

## Per-Height Comparison (2000 steps)

| height | prof | fell | pos% | min | max | maxabs | P2P | out15% | hip_yaw | score | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| low_0p300 | A | False | 44.6 | -0.0620 | 0.0678 | 0.0678 | 0.1299 | 0.0 | 0.2052 | 22.0 |  |
| low_0p300 | B | False | 45.5 | -0.0805 | 0.0687 | 0.0805 | 0.1492 | 0.0 | 0.2084 | 20.2 |  |
| low_0p300 | B2 | False | 45.9 | -0.0841 | 0.0731 | 0.0841 | 0.1572 | 0.0 | 0.2112 | 19.6 | IMPROVE |

| low_0p320 | A | False | 43.3 | -0.1253 | 0.1270 | 0.1270 | 0.2524 | 0.0 | 0.1699 | 349.9 |  |
| low_0p320 | B | False | 45.1 | -0.1409 | 0.1365 | 0.1409 | 0.2774 | 0.0 | 0.1864 | 450.4 |  |
| low_0p320 | B2 | False | 45.6 | -0.1345 | 0.1390 | 0.1390 | 0.2735 | 0.0 | 0.1617 | 483.8 | REGRESS |

| low_0p330 | A | False | 49.0 | -0.1172 | 0.1232 | 0.1232 | 0.2405 | 0.0 | 0.2034 | 275.8 |  |
| low_0p330 | B | False | 50.0 | -0.1214 | 0.1309 | 0.1309 | 0.2523 | 0.0 | 0.2020 | 243.0 |  |
| low_0p330 | B2 | False | 50.6 | -0.1231 | 0.1313 | 0.1313 | 0.2544 | 0.0 | 0.2034 | 241.2 | IMPROVE |

| low_0p340 | A | False | 44.6 | -0.1027 | 0.1343 | 0.1343 | 0.2370 | 0.0 | 0.1709 | 332.6 |  |
| low_0p340 | B | False | 44.3 | -0.1146 | 0.1401 | 0.1401 | 0.2547 | 0.0 | 0.1719 | 498.3 |  |
| low_0p340 | B2 | False | 44.0 | -0.1080 | 0.1378 | 0.1378 | 0.2457 | 0.0 | 0.1729 | 441.9 | IMPROVE |

| low_0p360 | A | False | 51.8 | -0.1109 | 0.1160 | 0.1160 | 0.2269 | 0.0 | 0.1813 | 118.4 |  |
| low_0p360 | B | False | 57.2 | -0.0889 | 0.1171 | 0.1171 | 0.2059 | 0.0 | 0.1174 | 204.8 |  |
| low_0p360 | B2 | False | 56.6 | -0.0592 | 0.1159 | 0.1159 | 0.1751 | 0.0 | 0.0930 | 76.9 | IMPROVE |

| low_0p380 | A | False | 44.8 | -0.1373 | 0.1112 | 0.1373 | 0.2485 | 0.0 | 0.0488 | 360.2 |  |
| low_0p380 | B | False | 45.9 | -0.1399 | 0.1104 | 0.1399 | 0.2503 | 0.0 | 0.0518 | 619.0 |  |
| low_0p380 | B2 | False | 44.7 | -0.1452 | 0.1111 | 0.1452 | 0.2563 | 0.0 | 0.0705 | 487.1 | IMPROVE |

| high_0p430 | A | False | 61.5 | -0.0721 | 0.1228 | 0.1228 | 0.1948 | 0.0 | 0.0378 | 363.5 |  |
| high_0p430 | B | False | 61.1 | -0.0838 | 0.1364 | 0.1364 | 0.2201 | 0.0 | 0.0458 | 413.9 |  |
| high_0p430 | B2 | False | 61.1 | -0.0838 | 0.1364 | 0.1364 | 0.2201 | 0.0 | 0.0458 | 413.9 | EQUAL |

| high_0p450 | A | False | 69.5 | -0.0712 | 0.1550 | 0.1550 | 0.2262 | 0.9 | 0.0691 | 704.6 |  |
| high_0p450 | B | False | 67.1 | -0.0733 | 0.1908 | 0.1908 | 0.2641 | 16.4 | 0.0251 | 2108.8 |  |
| high_0p450 | B2 | False | 69.4 | -0.0646 | 0.1524 | 0.1524 | 0.2170 | 0.7 | 0.0543 | 713.9 | IMPROVE |

| high_0p465 | A | False | 41.7 | -0.1657 | 0.1234 | 0.1657 | 0.2891 | 3.4 | 0.0487 | 946.2 |  |
| high_0p465 | B | False | 44.0 | -0.1519 | 0.1419 | 0.1519 | 0.2938 | 0.8 | 0.0379 | 906.4 |  |
| high_0p465 | B2 | False | 44.4 | -0.1657 | 0.1484 | 0.1657 | 0.3141 | 3.1 | 0.0357 | 1119.4 | EQUAL |

| high_0p480 | A | False | 57.0 | -0.1393 | 0.1694 | 0.1694 | 0.3087 | 5.2 | 0.0412 | 1224.1 |  |
| high_0p480 | B | False | 56.4 | -0.1496 | 0.1812 | 0.1812 | 0.3308 | 8.0 | 0.0494 | 1496.2 |  |
| high_0p480 | B2 | False | 56.4 | -0.1501 | 0.1850 | 0.1850 | 0.3351 | 10.0 | 0.0322 | 1702.5 | REGRESS |

## Decision

- **CALIBRATED_OUTER_LOOP_V2_NOT_BETTER**
- **Do not proceed. Keep `support_position_outer_loop_pitch_ref` as current best.**
