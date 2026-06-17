# Outer-Loop 500-Step Sign and Gain Sweep Report

**Profile:** `support_position_outer_loop_pitch_ref`
**Baseline:** `height_scheduled_pitch_equilibrium_trim`
**Height:** `high_0p480`  **Steps:** `500`
**Classification:** `OUTER_LOOP_500_CANDIDATE_SELECTED`

---

## Baseline (Phase A)

| Metric | Value |
|--------|-------|
| pos% | 61.1 |
| max_abs (m) | 0.0421 |
| P2P (m) | 0.0706 |
| out15% | 0.0 |
| hip_yaw_max (rad) | 0.0180 |
| zero_crossings | 7 |
| fell | False |

## Selected Candidate

**Sign:** `positive`  **Kp:** `+1.00 deg/m`  **Kd:** `0.00 deg/(m/s)`

| Metric | Baseline | Candidate | Delta |
|--------|----------|-----------|-------|
| pos% | 61.1000 | 60.3000 | -0.8000 |
| max_abs (m) | 0.0421 | 0.0437 | +0.0016 |
| P2P (m) | 0.0706 | 0.0760 | +0.0054 |
| out15% | 0.0000 | 0.0000 | +0.0000 |
| hip_yaw_max | 0.0180 | 0.0177 | -0.0003 |

**fell:** `False`

## Classification rationale

`OUTER_LOOP_500_CANDIDATE_SELECTED`

- No fall, posture safe, hip-yaw safe.
- maxabs/P2P within tolerance vs baseline.
- At least one centering metric improved.
- Candidate: Kp=+1.00 deg/m, Kd=0.00 deg/(m/s), sign=positive.

**Next step:** Phase 5 fixed-height full ladder validation.
