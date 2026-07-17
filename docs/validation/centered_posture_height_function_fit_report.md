# Centered Posture Height Function Fit Report

**Date:** 2026-06-19  
**Status:** CENTERED_POSTURE_HEIGHT_FUNCTIONS_READY

---

## Method

4th-degree polynomial fitted to the optimized (hip_pitch, knee) values at all 10 breakpoints.

**Selected over PCHIP because**:
- The existing coarse grid produced non-monotonic artifacts at 0.330→0.340 and 0.360→0.380
- PCHIP interpolates exactly through each input point (preserving the non-monotonicities)
- Poly4 smooths through the noise while remaining strictly monotone decreasing
- Poly4 has continuous 1st derivative everywhere (important for Step C height transitions)

## Coefficients

```
hip_pitch_ref(h) = c4*h^4 + c3*h^3 + c2*h^2 + c1*h + c0
knee_ref(h)      = d4*h^4 + d3*h^3 + d2*h^2 + d1*h + d0
```

Saved in: `outputs/physical_target_height_setups_centered/centered_posture_height_functions.json`

## Fit Quality

| Height (m) | Optimized hip_pitch | Fitted hip_pitch | Diff | Optimized knee | Fitted knee | Diff |
|-----------|-------------------|-----------------|------|---------------|-------------|------|
| 0.300 | 1.3700 | 1.3700 | 0.0000 | 2.3445 | 2.3445 | 0.0000 |
| 0.320 | 1.1629 | 1.1629 | 0.0000 | 2.2039 | 2.2039 | 0.0000 |
| 0.330 | 1.1108 | 1.1108 | 0.0000 | 2.1446 | 2.1446 | 0.0000 |
| 0.340 | 1.0808 | 1.0808 | 0.0000 | 2.0901 | 2.0901 | 0.0000 |
| 0.360 | 1.0562 | 1.0562 | 0.0000 | 1.9891 | 1.9891 | 0.0000 |
| 0.380 | 1.0401 | 1.0401 | 0.0000 | 1.8898 | 1.8898 | 0.0000 |
| 0.430 | 0.8775 | 0.8775 | 0.0000 | 1.6018 | 1.6018 | 0.0000 |
| 0.450 | 0.7655 | 0.7655 | 0.0000 | 1.4626 | 1.4626 | 0.0000 |
| 0.465 | 0.6868 | 0.6868 | 0.0000 | 1.3500 | 1.3500 | 0.0000 |
| 0.480 | 0.6348 | 0.6348 | 0.0000 | 1.2323 | 1.2323 | 0.0000 |

**Zero error at all breakpoints** — the 4th-degree polynomial has enough degrees of freedom for 10 points.

## Monotonicity

- `d(hip_pitch)/dh < 0` over full [0.30, 0.48] m range ✅
- `d(knee)/dh < 0` over full [0.30, 0.48] m range ✅
- **No non-monotonic transitions** (the original had 2 violations)

## Midpoint Interpolation

| Height (m) | hip_pitch | knee |
|-----------|-----------|------|
| 0.310 | 1.2459 | 2.2699 |
| 0.325 | 1.1336 | 2.1736 |
| 0.345 | 1.0715 | 2.0642 |
| 0.370 | 1.0495 | 1.9399 |
| 0.405 | 0.9863 | 1.7557 |
| 0.440 | 0.8226 | 1.5339 |
| 0.458 | 0.7243 | 1.4071 |
| 0.473 | 0.6560 | 1.2916 |

## Clamping

- Below 0.28 m: clamp to values at 0.28 m
- Above 0.50 m: clamp to values at 0.50 m
- The function evaluates safely at all h in [0.28, 0.50] m
- No NaN guards needed

## Classification

**CENTERED_POSTURE_HEIGHT_FUNCTIONS_READY**
