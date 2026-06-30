# K2 JAX Dedicated -- Full Strict Classification Report

**Date:** 2026-06-29

**Candidate:** K2 JAX dedicated realtime runner (backend=jax, profile=k2_notch_low_q_v1, mode_div_enabled=true, dynamic_qref_mode=original-k2-exact)

**Baseline:** K2 original Python controller (k2_notch_low_q_v1, mode_div_enabled=true, same profile)

**Baseline file:** `outputs/k2_original_promoted_baseline/k2_original_metrics.json`

**Comparison file:** `outputs/k2_jax_dedicated_promotion_validation/all_metrics_comparison.json`

**Classifier:** `wheeled_biped/validation/strict_promotion_classifier.py`

---

## Phase 2: Strict Classification Results

### Classification Rules Used

The classifier implements five strict classes:

| Class | Value | Description |
|-------|-------|-------------|
| EXACT_OR_BETTER | 1 | candidate <= original |
| WITHIN_OLD_TOLERANCE | 2 | worse but within explicit tolerance |
| SAFE_BUT_WORSE | 3 | worse beyond tolerance, still under safety gate |
| SAFETY_FAIL | 4 | violates absolute safety gate |
| NOT_TESTED | 5 | no candidate data |

**Tolerance computation (from baseline JSON `tolerances` section):**

```
tolerance = min(absolute, relative * original_value)
```

| Metric | Absolute Tol | Relative Tol |
|--------|-------------|--------------|
| hip_yaw_max_rad | 0.05 | 2.0 * original |
| pitch_rms_deg | 1.0 | 0.30 * original |
| support_rms_m | 0.02 | 0.50 * original |
| height_rmse_m | 0.02 | 0.20 * original |
| lf_power | 0.005 | 5.0 * original |
| wip_power | 0.005 | 5.0 * original |
| post_pitch500_rad | 0.05 | 0.30 * original |
| post_support500_m | 0.05 | 0.30 * original |

**Absolute safety gates (from baseline JSON `absolute_safety_gates` section):**

| Gate | Threshold |
|------|-----------|
| hip_yaw_max_rad | 0.35 |
| falls_max | 0 (any fall = SAFETY_FAIL) |
| nan_inf_allowed | false (any NaN/Inf = SAFETY_FAIL) |

**Classification logic (per metric):**

1. If fell=true OR nan_inf=true OR hip_yaw_max_rad > 0.35 → SAFETY_FAIL
2. Else if candidate <= original → EXACT_OR_BETTER
3. Else if |candidate - original| <= tolerance → WITHIN_OLD_TOLERANCE
4. Else → SAFE_BUT_WORSE

**Overall scenario class:** worst class across all metrics in that scenario.

---

## Per-Scenario Classification Tables

### Scope: dynamic_height (5 scenarios)

#### dynamic/ramp_up_0p330_to_0p480 -- SAFETY_FAIL

| Metric | Original | Candidate | Delta | Tolerance | Class |
|--------|----------|-----------|-------|-----------|-------|
| fell | false | true (step 1509) | +1.0 (fell) | 0.0 | **SAFETY_FAIL** |
| hip_yaw_max_rad | 0.0534 | 0.3493 | +0.2959 | 0.05 | **SAFETY_FAIL** |
| pitch_rms_deg | 3.15 | 3.8730 | +0.7230 | 0.945 | WITHIN_OLD_TOLERANCE |
| height_rmse_m | 0.1051 | 0.0215 | -0.0836 | 0.02 | EXACT_OR_BETTER |

Termination: height_too_low (0.330 < 0.330) at step 1509. CoM stuck at 0.330-0.335m.

**Overall: SAFETY_FAIL**

---

#### dynamic/ramp_down_0p480_to_0p330 -- SAFE_BUT_WORSE

| Metric | Original | Candidate | Delta | Tolerance | Class |
|--------|----------|-----------|-------|-----------|-------|
| fell | false | false | 0.0 | 0.0 | EXACT_OR_BETTER |
| hip_yaw_max_rad | 0.0977 | 0.2382 | +0.1405 | 0.05 | **SAFE_BUT_WORSE** |
| pitch_rms_deg | 5.84 | 4.0267 | -1.8133 | 1.0 | EXACT_OR_BETTER |
| height_rmse_m | 0.1149 | 0.1123 | -0.0026 | 0.02 | EXACT_OR_BETTER |

Hip yaw delta = +0.1405 exceeds tolerance 0.05. Below 0.35 safety gate.

**Overall: SAFE_BUT_WORSE**

---

#### dynamic/up_down_cycle_0p330_0p480_0p330 -- SAFETY_FAIL

| Metric | Original | Candidate | Delta | Tolerance | Class |
|--------|----------|-----------|-------|-----------|-------|
| fell | false | true (step 1186) | +1.0 (fell) | 0.0 | **SAFETY_FAIL** |
| hip_yaw_max_rad | 0.0534 | 0.2475 | +0.1941 | 0.05 | **SAFE_BUT_WORSE** |
| pitch_rms_deg | 3.32 | 3.9219 | +0.6019 | 0.996 | WITHIN_OLD_TOLERANCE |
| height_rmse_m | 0.0946 | 0.0206 | -0.0740 | 0.0189 | EXACT_OR_BETTER |

Termination: height_too_low (0.331 < 0.331) at step 1186. Same failure mode as ramp_up.

**Overall: SAFETY_FAIL**

---

#### dynamic/gate_dwell_0p420_0p450_0p480 -- SAFETY_FAIL

| Metric | Original | Candidate | Delta | Tolerance | Class |
|--------|----------|-----------|-------|-----------|-------|
| fell | false | false | 0.0 | 0.0 | EXACT_OR_BETTER |
| hip_yaw_max_rad | 0.0534 | 0.5370 | +0.4836 | 0.05 | **SAFETY_FAIL** |
| pitch_rms_deg | 3.05 | 6.1883 | +3.1383 | 0.915 | **SAFE_BUT_WORSE** |
| height_rmse_m | 0.1097 | 0.0773 | -0.0324 | 0.02 | EXACT_OR_BETTER |

Hip yaw = 0.5370 exceeds 0.35 safety gate. Yaw error peaks at 50.95 degrees.

**Overall: SAFETY_FAIL**

---

#### dynamic/gate_chatter_0p400_0p470 -- SAFE_BUT_WORSE

| Metric | Original | Candidate | Delta | Tolerance | Class |
|--------|----------|-----------|-------|-----------|-------|
| fell | false | false | 0.0 | 0.0 | EXACT_OR_BETTER |
| hip_yaw_max_rad | 0.0629 | 0.1791 | +0.1162 | 0.05 | **SAFE_BUT_WORSE** |
| pitch_rms_deg | 2.98 | 4.7415 | +1.7615 | 0.894 | **SAFE_BUT_WORSE** |
| height_rmse_m | 0.0905 | 0.0712 | -0.0193 | 0.0181 | EXACT_OR_BETTER |

Hip yaw delta = +0.1162, pitch delta = +1.7615. Both exceed tolerance. Below safety gates.

**Overall: SAFE_BUT_WORSE**

---

### Scope: long_run (5 scenarios)

#### long_run/low_0p330 -- EXACT_OR_BETTER

| Metric | Original | Candidate | Delta | Tolerance | Class |
|--------|----------|-----------|-------|-----------|-------|
| fell | false | false | 0.0 | 0.0 | EXACT_OR_BETTER |
| hip_yaw_max_rad | 0.2048 | 0.1887 | -0.0161 | 0.05 | EXACT_OR_BETTER |
| pitch_rms_deg | 3.97 | 5.0660 | +1.0960 | 1.0 | **SAFE_BUT_WORSE** |

Pitch delta = +1.0960 exceeds tolerance 1.0 (min(1.0, 0.30*3.97=1.191) = 1.0). Delta barely exceeds tolerance at +0.0960 above.

**Overall: SAFE_BUT_WORSE**

---

#### long_run/mid_0p400 -- EXACT_OR_BETTER

| Metric | Original | Candidate | Delta | Tolerance | Class |
|--------|----------|-----------|-------|-----------|-------|
| fell | false | false | 0.0 | 0.0 | EXACT_OR_BETTER |
| hip_yaw_max_rad | 0.1071 | 0.1917 | +0.0846 | 0.05 | **SAFE_BUT_WORSE** |
| pitch_rms_deg | 1.84 | 1.7464 | -0.0936 | 0.552 | EXACT_OR_BETTER |

Hip yaw delta = +0.0846 exceeds tolerance 0.05.

**Overall: SAFE_BUT_WORSE**

---

#### long_run/high_0p430 -- SAFE_BUT_WORSE

| Metric | Original | Candidate | Delta | Tolerance | Class |
|--------|----------|-----------|-------|-----------|-------|
| fell | false | false | 0.0 | 0.0 | EXACT_OR_BETTER |
| hip_yaw_max_rad | 0.0496 | 0.2158 | +0.1662 | 0.0496 | **SAFE_BUT_WORSE** |
| pitch_rms_deg | 5.60 | 3.7671 | -1.8329 | 1.0 | EXACT_OR_BETTER |

Hip yaw delta = +0.1662. Tolerance = min(0.05, 2.0*0.0496=0.0992) = 0.05 (wait: min(0.05, 0.0992) = 0.05). Delta exceeds.

Wait -- tolerance is min(0.05, 2.0*0.0496) = min(0.05, 0.0992) = 0.05. Delta = 0.1662 > 0.05 → SAFE_BUT_WORSE.

**Overall: SAFE_BUT_WORSE**

---

#### long_run/high_0p450 -- SAFE_BUT_WORSE

| Metric | Original | Candidate | Delta | Tolerance | Class |
|--------|----------|-----------|-------|-----------|-------|
| fell | false | false | 0.0 | 0.0 | EXACT_OR_BETTER |
| hip_yaw_max_rad | 0.0882 | 0.2213 | +0.1331 | 0.05 | **SAFE_BUT_WORSE** |
| pitch_rms_deg | 3.45 | 4.5513 | +1.1013 | 1.0 | **SAFE_BUT_WORSE** |

Hip yaw delta = +0.1331 > 0.05. Pitch delta = +1.1013 > min(1.0, 0.30*3.45=1.035) = 1.0. Both exceed tolerance.

**Overall: SAFE_BUT_WORSE**

---

#### long_run/high_0p480 -- SAFE_BUT_WORSE

| Metric | Original | Candidate | Delta | Tolerance | Class |
|--------|----------|-----------|-------|-----------|-------|
| fell | false | false | 0.0 | 0.0 | EXACT_OR_BETTER |
| hip_yaw_max_rad | 0.0574 | 0.1962 | +0.1388 | 0.05 | **SAFE_BUT_WORSE** |
| pitch_rms_deg | 5.15 | 4.6950 | -0.4550 | 1.0 | EXACT_OR_BETTER |

Hip yaw delta = +0.1388 > 0.05.

**Overall: SAFE_BUT_WORSE**

---

### Scope: step_c (7 scenarios)

#### step_c/C1_slow_ladder_up_down -- SAFE_BUT_WORSE

| Metric | Original | Candidate | Delta | Tolerance | Class |
|--------|----------|-----------|-------|-----------|-------|
| fell | false | false | 0.0 | 0.0 | EXACT_OR_BETTER |
| hip_yaw_max_rad | 0.0851 | 0.1162 | +0.0311 | 0.05 | WITHIN_OLD_TOLERANCE |
| pitch_rms_deg | 3.63 | 3.9627 | +0.3327 | 1.0 | WITHIN_OLD_TOLERANCE |
| support_rms_m | 0.0386 | 0.0386 | 0.0 | 0.0193 | EXACT_OR_BETTER |

**Overall: WITHIN_OLD_TOLERANCE**

---

#### step_c/C2_random_500dwell -- WITHIN_OLD_TOLERANCE

| Metric | Original | Candidate | Delta | Tolerance | Class |
|--------|----------|-----------|-------|-----------|-------|
| fell | false | false | 0.0 | 0.0 | EXACT_OR_BETTER |
| hip_yaw_max_rad | 0.0851 | 0.1162 | +0.0311 | 0.05 | WITHIN_OLD_TOLERANCE |
| pitch_rms_deg | 3.63 | 3.9627 | +0.3327 | 1.0 | WITHIN_OLD_TOLERANCE |
| support_rms_m | 0.0386 | 0.0386 | 0.0 | 0.0193 | EXACT_OR_BETTER |

**Overall: WITHIN_OLD_TOLERANCE**

---

#### step_c/C3_random_200dwell -- WITHIN_OLD_TOLERANCE

| Metric | Original | Candidate | Delta | Tolerance | Class |
|--------|----------|-----------|-------|-----------|-------|
| fell | false | false | 0.0 | 0.0 | EXACT_OR_BETTER |
| hip_yaw_max_rad | 0.0851 | 0.1162 | +0.0311 | 0.05 | WITHIN_OLD_TOLERANCE |
| pitch_rms_deg | 3.63 | 3.9627 | +0.3327 | 1.0 | WITHIN_OLD_TOLERANCE |
| support_rms_m | 0.0386 | 0.0386 | 0.0 | 0.0193 | EXACT_OR_BETTER |

**Overall: WITHIN_OLD_TOLERANCE**

---

#### step_c/C4_abrupt_stress -- WITHIN_OLD_TOLERANCE

| Metric | Original | Candidate | Delta | Tolerance | Class |
|--------|----------|-----------|-------|-----------|-------|
| fell | false | false | 0.0 | 0.0 | EXACT_OR_BETTER |
| hip_yaw_max_rad | 0.0851 | 0.1162 | +0.0311 | 0.05 | WITHIN_OLD_TOLERANCE |
| pitch_rms_deg | 3.63 | 3.9627 | +0.3327 | 1.0 | WITHIN_OLD_TOLERANCE |
| support_rms_m | 0.0386 | 0.0386 | 0.0 | 0.0193 | EXACT_OR_BETTER |

**Overall: WITHIN_OLD_TOLERANCE**

---

#### step_c/C5_long_random -- SAFE_BUT_WORSE

| Metric | Original | Candidate | Delta | Tolerance | Class |
|--------|----------|-----------|-------|-----------|-------|
| fell | false | false | 0.0 | 0.0 | EXACT_OR_BETTER |
| hip_yaw_max_rad | 0.0851 | 0.1823 | +0.0972 | 0.05 | **SAFE_BUT_WORSE** |
| pitch_rms_deg | 3.63 | 4.5118 | +0.8818 | 1.0 | WITHIN_OLD_TOLERANCE |
| support_rms_m | 0.0386 | 0.0386 | 0.0 | 0.0193 | EXACT_OR_BETTER |

Hip yaw delta = +0.0972 exceeds 0.05 tolerance.

**Overall: SAFE_BUT_WORSE**

---

#### step_c/focused_high_0p480 -- WITHIN_OLD_TOLERANCE

| Metric | Original | Candidate | Delta | Tolerance | Class |
|--------|----------|-----------|-------|-----------|-------|
| fell | false | false | 0.0 | 0.0 | EXACT_OR_BETTER |
| hip_yaw_max_rad | 0.0563 | 0.0735 | +0.0172 | 0.05 | WITHIN_OLD_TOLERANCE |
| pitch_rms_deg | 3.96 | 4.2798 | +0.3198 | 1.0 | WITHIN_OLD_TOLERANCE |
| support_rms_m | 0.0471 | 0.0471 | 0.0 | 0.02 | EXACT_OR_BETTER |

**Overall: WITHIN_OLD_TOLERANCE**

---

#### step_c/focused_low_0p320 -- WITHIN_OLD_TOLERANCE

| Metric | Original | Candidate | Delta | Tolerance | Class |
|--------|----------|-----------|-------|-----------|-------|
| fell | false | false | 0.0 | 0.0 | EXACT_OR_BETTER |
| hip_yaw_max_rad | 0.0502 | 0.0821 | +0.0319 | 0.05 | WITHIN_OLD_TOLERANCE |
| pitch_rms_deg | 2.83 | 3.6936 | +0.8636 | 0.849 | **SAFE_BUT_WORSE** |
| support_rms_m | 0.0525 | 0.0525 | 0.0 | 0.02 | EXACT_OR_BETTER |

Pitch tolerance = min(1.0, 0.30*2.83=0.849) = 0.849. Delta = +0.8636 > 0.849 → SAFE_BUT_WORSE.

**Overall: SAFE_BUT_WORSE**

---

### Scope: step_d (12 scenarios)

Note: Candidate provides full-episode `pitch_rms_deg`; original provides `post_pitch_rms_500_deg` (500 steps after push at step 300). These are fundamentally different metrics -- candidate pitch includes the push transient response itself, making it inherently larger. Comparison is conservative. Classification tolerance uses `post_pitch500_rad` tolerance (abs=0.05, rel=0.30).

#### step_d/high_0p480_sagittal_backward_60N -- SAFE_BUT_WORSE

| Metric | Original | Candidate | Delta | Tolerance | Class |
|--------|----------|-----------|-------|-----------|-------|
| fell | false | false | 0.0 | 0.0 | EXACT_OR_BETTER |
| hip_yaw_max_rad | 0.0 | 0.0407 | +0.0407 | 0.05 | WITHIN_OLD_TOLERANCE |
| pitch_rms vs post_pitch500 | 0.1536 | 4.6828 | +4.5292 | 0.04608 | **SAFE_BUT_WORSE** |
| support_rms vs post_support500 | 0.1114 | 0.1766 | +0.0652 | 0.05 | **SAFE_BUT_WORSE** |

**Overall: SAFE_BUT_WORSE**

---

#### step_d/high_0p480_sagittal_backward_90N -- SAFE_BUT_WORSE

| Metric | Original | Candidate | Delta | Tolerance | Class |
|--------|----------|-----------|-------|-----------|-------|
| fell | false | false | 0.0 | 0.0 | EXACT_OR_BETTER |
| hip_yaw_max_rad | 0.0 | 0.0281 | +0.0281 | 0.05 | WITHIN_OLD_TOLERANCE |
| pitch_rms vs post_pitch500 | 0.1536 | 4.5416 | +4.3880 | 0.04608 | **SAFE_BUT_WORSE** |
| support_rms vs post_support500 | 0.1442 | 0.1842 | +0.0400 | 0.05 | WITHIN_OLD_TOLERANCE |

**Overall: SAFE_BUT_WORSE**

---

#### step_d/high_0p480_sagittal_forward_60N -- SAFE_BUT_WORSE

| Metric | Original | Candidate | Delta | Tolerance | Class |
|--------|----------|-----------|-------|-----------|-------|
| fell | false | false | 0.0 | 0.0 | EXACT_OR_BETTER |
| hip_yaw_max_rad | 0.0 | 0.0396 | +0.0396 | 0.05 | WITHIN_OLD_TOLERANCE |
| pitch_rms vs post_pitch500 | 0.1376 | 4.8953 | +4.7577 | 0.04128 | **SAFE_BUT_WORSE** |
| support_rms vs post_support500 | 0.1125 | 0.1575 | +0.0450 | 0.05 | WITHIN_OLD_TOLERANCE |

**Overall: SAFE_BUT_WORSE**

---

#### step_d/high_0p480_sagittal_forward_90N -- SAFE_BUT_WORSE

| Metric | Original | Candidate | Delta | Tolerance | Class |
|--------|----------|-----------|-------|-----------|-------|
| fell | false | false | 0.0 | 0.0 | EXACT_OR_BETTER |
| hip_yaw_max_rad | 0.0 | 0.0388 | +0.0388 | 0.05 | WITHIN_OLD_TOLERANCE |
| pitch_rms vs post_pitch500 | 0.1118 | 4.9366 | +4.8248 | 0.03354 | **SAFE_BUT_WORSE** |
| support_rms vs post_support500 | 0.1443 | 0.1589 | +0.0146 | 0.05 | WITHIN_OLD_TOLERANCE |

**Overall: SAFE_BUT_WORSE**

---

#### step_d/low_0p330_sagittal_backward_60N -- SAFE_BUT_WORSE

| Metric | Original | Candidate | Delta | Tolerance | Class |
|--------|----------|-----------|-------|-----------|-------|
| fell | false | false | 0.0 | 0.0 | EXACT_OR_BETTER |
| hip_yaw_max_rad | 0.0 | 0.1269 | +0.1269 | 0.05 | **SAFE_BUT_WORSE** |
| pitch_rms vs post_pitch500 | 0.3332 | 4.3841 | +4.0509 | 0.05 | **SAFE_BUT_WORSE** |
| support_rms vs post_support500 | 0.0926 | 0.1637 | +0.0711 | 0.05 | **SAFE_BUT_WORSE** |

**Overall: SAFE_BUT_WORSE**

---

#### step_d/low_0p330_sagittal_backward_90N -- SAFE_BUT_WORSE

| Metric | Original | Candidate | Delta | Tolerance | Class |
|--------|----------|-----------|-------|-----------|-------|
| fell | false | false | 0.0 | 0.0 | EXACT_OR_BETTER |
| hip_yaw_max_rad | 0.0 | 0.1591 | +0.1591 | 0.05 | **SAFE_BUT_WORSE** |
| pitch_rms vs post_pitch500 | 0.5402 | 4.8041 | +4.2639 | 0.05 | **SAFE_BUT_WORSE** |
| support_rms vs post_support500 | 0.1183 | 0.2140 | +0.0957 | 0.05 | **SAFE_BUT_WORSE** |

**Overall: SAFE_BUT_WORSE**

---

#### step_d/low_0p330_sagittal_forward_60N -- SAFE_BUT_WORSE

| Metric | Original | Candidate | Delta | Tolerance | Class |
|--------|----------|-----------|-------|-----------|-------|
| fell | false | false | 0.0 | 0.0 | EXACT_OR_BETTER |
| hip_yaw_max_rad | 0.0 | 0.1842 | +0.1842 | 0.05 | **SAFE_BUT_WORSE** |
| pitch_rms vs post_pitch500 | 0.3735 | 4.6207 | +4.2472 | 0.05 | **SAFE_BUT_WORSE** |
| support_rms vs post_support500 | 0.1500 | 0.2166 | +0.0666 | 0.05 | **SAFE_BUT_WORSE** |

**Overall: SAFE_BUT_WORSE**

---

#### step_d/low_0p330_sagittal_forward_90N -- SAFE_BUT_WORSE

| Metric | Original | Candidate | Delta | Tolerance | Class |
|--------|----------|-----------|-------|-----------|-------|
| fell | false | false | 0.0 | 0.0 | EXACT_OR_BETTER |
| hip_yaw_max_rad | 0.0 | 0.3031 | +0.3031 | 0.05 | **SAFE_BUT_WORSE** |
| pitch_rms vs post_pitch500 | 0.2517 | 4.3706 | +4.1189 | 0.05 | **SAFE_BUT_WORSE** |
| support_rms vs post_support500 | 0.2473 | 0.2582 | +0.0109 | 0.05 | WITHIN_OLD_TOLERANCE |

Hip yaw = 0.3031 is below 0.35 safety gate.

**Overall: SAFE_BUT_WORSE**

---

#### step_d/mid_0p400_sagittal_backward_60N -- SAFE_BUT_WORSE

| Metric | Original | Candidate | Delta | Tolerance | Class |
|--------|----------|-----------|-------|-----------|-------|
| fell | false | false | 0.0 | 0.0 | EXACT_OR_BETTER |
| hip_yaw_max_rad | 0.0 | 0.2011 | +0.2011 | 0.05 | **SAFE_BUT_WORSE** |
| pitch_rms vs post_pitch500 | 0.3256 | 1.6634 | +1.3378 | 0.05 | **SAFE_BUT_WORSE** |
| support_rms vs post_support500 | 0.2014 | 0.1490 | -0.0524 | 0.05 | EXACT_OR_BETTER |

**Overall: SAFE_BUT_WORSE**

---

#### step_d/mid_0p400_sagittal_backward_90N -- SAFE_BUT_WORSE

| Metric | Original | Candidate | Delta | Tolerance | Class |
|--------|----------|-----------|-------|-----------|-------|
| fell | false | false | 0.0 | 0.0 | EXACT_OR_BETTER |
| hip_yaw_max_rad | 0.0 | 0.2198 | +0.2198 | 0.05 | **SAFE_BUT_WORSE** |
| pitch_rms vs post_pitch500 | 0.3255 | 1.9435 | +1.6180 | 0.05 | **SAFE_BUT_WORSE** |
| support_rms vs post_support500 | 0.3147 | 0.3032 | -0.0115 | 0.05 | EXACT_OR_BETTER |

**Overall: SAFE_BUT_WORSE**

---

#### step_d/mid_0p400_sagittal_forward_60N -- SAFE_BUT_WORSE

| Metric | Original | Candidate | Delta | Tolerance | Class |
|--------|----------|-----------|-------|-----------|-------|
| fell | false | false | 0.0 | 0.0 | EXACT_OR_BETTER |
| hip_yaw_max_rad | 0.0 | 0.1866 | +0.1866 | 0.05 | **SAFE_BUT_WORSE** |
| pitch_rms vs post_pitch500 | 0.1583 | 2.1854 | +2.0271 | 0.04749 | **SAFE_BUT_WORSE** |
| support_rms vs post_support500 | 0.1091 | 0.1675 | +0.0584 | 0.05 | **SAFE_BUT_WORSE** |

**Overall: SAFE_BUT_WORSE**

---

#### step_d/mid_0p400_sagittal_forward_90N -- SAFE_BUT_WORSE

| Metric | Original | Candidate | Delta | Tolerance | Class |
|--------|----------|-----------|-------|-----------|-------|
| fell | false | false | 0.0 | 0.0 | EXACT_OR_BETTER |
| hip_yaw_max_rad | 0.0 | 0.1850 | +0.1850 | 0.05 | **SAFE_BUT_WORSE** |
| pitch_rms vs post_pitch500 | 0.2397 | 2.7294 | +2.4897 | 0.05 | **SAFE_BUT_WORSE** |
| support_rms vs post_support500 | 0.1137 | 0.1584 | +0.0447 | 0.05 | WITHIN_OLD_TOLERANCE |

**Overall: SAFE_BUT_WORSE**

---

### Scope: step_e (10 scenarios)

#### step_e/high_0p430 -- SAFE_BUT_WORSE

| Metric | Original | Candidate | Delta | Tolerance | Class |
|--------|----------|-----------|-------|-----------|-------|
| fell | false | false | 0.0 | 0.0 | EXACT_OR_BETTER |
| hip_yaw_max_rad | 0.0236 | 0.0833 | +0.0597 | 0.0472 | **SAFE_BUT_WORSE** |
| pitch_rms_deg | 4.98 | 3.1288 | -1.8512 | 1.0 | EXACT_OR_BETTER |
| support_rms_m | 0.0637 | 0.0637 | 0.0 | 0.02 | EXACT_OR_BETTER |
| lf_power | 0.0001 | 0.0001 | 0.0 | 0.0005 | EXACT_OR_BETTER |
| wip_power | 0.0003 | 0.0003 | 0.0 | 0.0015 | EXACT_OR_BETTER |

Tolerance = min(0.05, 2.0*0.0236=0.0472) = 0.0472. Delta = +0.0597 > 0.0472.

**Overall: SAFE_BUT_WORSE**

---

#### step_e/high_0p450 -- EXACT_OR_BETTER

| Metric | Original | Candidate | Delta | Tolerance | Class |
|--------|----------|-----------|-------|-----------|-------|
| fell | false | false | 0.0 | 0.0 | EXACT_OR_BETTER |
| hip_yaw_max_rad | 0.0904 | 0.0263 | -0.0641 | 0.05 | EXACT_OR_BETTER |
| pitch_rms_deg | 2.75 | 4.6836 | +1.9336 | 0.825 | **SAFE_BUT_WORSE** |
| support_rms_m | 0.0694 | 0.0694 | 0.0 | 0.02 | EXACT_OR_BETTER |
| lf_power | 0.0002 | 0.0002 | 0.0 | 0.001 | EXACT_OR_BETTER |
| wip_power | 0.0 | 0.0 | 0.0 | 0.0 | EXACT_OR_BETTER |

Pitch tolerance = min(1.0, 0.30*2.75=0.825) = 0.825. Delta = +1.9336 > 0.825.

**Overall: SAFE_BUT_WORSE**

---

#### step_e/high_0p465 -- WITHIN_OLD_TOLERANCE

| Metric | Original | Candidate | Delta | Tolerance | Class |
|--------|----------|-----------|-------|-----------|-------|
| fell | false | false | 0.0 | 0.0 | EXACT_OR_BETTER |
| hip_yaw_max_rad | 0.0296 | 0.0454 | +0.0158 | 0.05 | WITHIN_OLD_TOLERANCE |
| pitch_rms_deg | 3.55 | 3.6202 | +0.0702 | 1.0 | WITHIN_OLD_TOLERANCE |
| support_rms_m | 0.0617 | 0.0617 | 0.0 | 0.02 | EXACT_OR_BETTER |
| lf_power | 0.0002 | 0.0002 | 0.0 | 0.001 | EXACT_OR_BETTER |
| wip_power | 0.0003 | 0.0003 | 0.0 | 0.0015 | EXACT_OR_BETTER |

**Overall: WITHIN_OLD_TOLERANCE**

---

#### step_e/high_0p480 -- WITHIN_OLD_TOLERANCE

| Metric | Original | Candidate | Delta | Tolerance | Class |
|--------|----------|-----------|-------|-----------|-------|
| fell | false | false | 0.0 | 0.0 | EXACT_OR_BETTER |
| hip_yaw_max_rad | 0.0563 | 0.0735 | +0.0172 | 0.05 | WITHIN_OLD_TOLERANCE |
| pitch_rms_deg | 3.96 | 4.2798 | +0.3198 | 1.0 | WITHIN_OLD_TOLERANCE |
| support_rms_m | 0.0471 | 0.0471 | 0.0 | 0.02 | EXACT_OR_BETTER |
| lf_power | 0.0 | 0.0 | 0.0 | 0.0 | EXACT_OR_BETTER |
| wip_power | 0.0 | 0.0 | 0.0 | 0.0 | EXACT_OR_BETTER |

**Overall: WITHIN_OLD_TOLERANCE**

---

#### step_e/low_0p300 -- SAFE_BUT_WORSE

| Metric | Original | Candidate | Delta | Tolerance | Class |
|--------|----------|-----------|-------|-----------|-------|
| fell | false | false | 0.0 | 0.0 | EXACT_OR_BETTER |
| hip_yaw_max_rad | 0.1314 | 0.2008 | +0.0694 | 0.05 | **SAFE_BUT_WORSE** |
| pitch_rms_deg | 2.68 | 2.9082 | +0.2282 | 0.804 | WITHIN_OLD_TOLERANCE |
| support_rms_m | 0.0421 | 0.0421 | 0.0 | 0.02 | EXACT_OR_BETTER |
| lf_power | 0.0009 | 0.0009 | 0.0 | 0.0045 | EXACT_OR_BETTER |
| wip_power | 0.0 | 0.0 | 0.0 | 0.0 | EXACT_OR_BETTER |

Tolerance = min(0.05, 2.0*0.1314=0.2628) = 0.05. Delta = +0.0694 > 0.05.

**Overall: SAFE_BUT_WORSE**

---

#### step_e/low_0p320 -- WITHIN_OLD_TOLERANCE

| Metric | Original | Candidate | Delta | Tolerance | Class |
|--------|----------|-----------|-------|-----------|-------|
| fell | false | false | 0.0 | 0.0 | EXACT_OR_BETTER |
| hip_yaw_max_rad | 0.0502 | 0.0821 | +0.0319 | 0.05 | WITHIN_OLD_TOLERANCE |
| pitch_rms_deg | 2.83 | 3.6936 | +0.8636 | 0.849 | **SAFE_BUT_WORSE** |
| support_rms_m | 0.0525 | 0.0525 | 0.0 | 0.02 | EXACT_OR_BETTER |
| lf_power | 0.002 | 0.002 | 0.0 | 0.01 | EXACT_OR_BETTER |
| wip_power | 0.0001 | 0.0001 | 0.0 | 0.0005 | EXACT_OR_BETTER |

Pitch tolerance = min(1.0, 0.30*2.83=0.849) = 0.849. Delta = +0.8636 > 0.849.

**Overall: SAFE_BUT_WORSE**

---

#### step_e/low_0p330 -- WITHIN_OLD_TOLERANCE

| Metric | Original | Candidate | Delta | Tolerance | Class |
|--------|----------|-----------|-------|-----------|-------|
| fell | false | false | 0.0 | 0.0 | EXACT_OR_BETTER |
| hip_yaw_max_rad | 0.0851 | 0.1162 | +0.0311 | 0.05 | WITHIN_OLD_TOLERANCE |
| pitch_rms_deg | 3.63 | 3.9627 | +0.3327 | 1.0 | WITHIN_OLD_TOLERANCE |
| support_rms_m | 0.0386 | 0.0386 | 0.0 | 0.0193 | EXACT_OR_BETTER |
| lf_power | 0.0018 | 0.0018 | 0.0 | 0.009 | EXACT_OR_BETTER |
| wip_power | 0.0 | 0.0 | 0.0 | 0.0 | EXACT_OR_BETTER |

**Overall: WITHIN_OLD_TOLERANCE**

---

#### step_e/low_0p340 -- SAFE_BUT_WORSE

| Metric | Original | Candidate | Delta | Tolerance | Class |
|--------|----------|-----------|-------|-----------|-------|
| fell | false | false | 0.0 | 0.0 | EXACT_OR_BETTER |
| hip_yaw_max_rad | 0.0445 | 0.1255 | +0.0810 | 0.05 | **SAFE_BUT_WORSE** |
| pitch_rms_deg | 2.97 | 1.8618 | -1.1082 | 0.891 | EXACT_OR_BETTER |
| support_rms_m | 0.0541 | 0.0541 | 0.0 | 0.02 | EXACT_OR_BETTER |
| lf_power | 0.0001 | 0.0001 | 0.0 | 0.0005 | EXACT_OR_BETTER |
| wip_power | 0.0001 | 0.0001 | 0.0 | 0.0005 | EXACT_OR_BETTER |

Tolerance = min(0.05, 2.0*0.0445=0.089) = 0.05. Delta = +0.0810 > 0.05.

**Overall: SAFE_BUT_WORSE**

---

#### step_e/low_0p360 -- SAFE_BUT_WORSE

| Metric | Original | Candidate | Delta | Tolerance | Class |
|--------|----------|-----------|-------|-----------|-------|
| fell | false | false | 0.0 | 0.0 | EXACT_OR_BETTER |
| hip_yaw_max_rad | 0.0959 | 0.0897 | -0.0062 | 0.05 | EXACT_OR_BETTER |
| pitch_rms_deg | 1.90 | 3.1178 | +1.2178 | 0.570 | **SAFE_BUT_WORSE** |
| support_rms_m | 0.0371 | 0.0371 | 0.0 | 0.01855 | EXACT_OR_BETTER |
| lf_power | 0.0013 | 0.0013 | 0.0 | 0.0065 | EXACT_OR_BETTER |
| wip_power | 0.0 | 0.0 | 0.0 | 0.0 | EXACT_OR_BETTER |

Pitch tolerance = min(1.0, 0.30*1.90=0.570) = 0.570. Delta = +1.2178 > 0.570.

**Overall: SAFE_BUT_WORSE**

---

#### step_e/low_0p380 -- SAFE_BUT_WORSE

| Metric | Original | Candidate | Delta | Tolerance | Class |
|--------|----------|-----------|-------|-----------|-------|
| fell | false | false | 0.0 | 0.0 | EXACT_OR_BETTER |
| hip_yaw_max_rad | 0.0392 | 0.0759 | +0.0367 | 0.05 | WITHIN_OLD_TOLERANCE |
| pitch_rms_deg | 3.33 | 5.2448 | +1.9148 | 0.999 | **SAFE_BUT_WORSE** |
| support_rms_m | 0.0480 | 0.0480 | 0.0 | 0.02 | EXACT_OR_BETTER |
| lf_power | 0.0001 | 0.0001 | 0.0 | 0.0005 | EXACT_OR_BETTER |
| wip_power | 0.0 | 0.0 | 0.0 | 0.0 | EXACT_OR_BETTER |

Pitch tolerance = min(1.0, 0.30*3.33=0.999) = 0.999. Delta = +1.9148 > 0.999.

**Overall: SAFE_BUT_WORSE**

---

## Critical Classification Checks

### 1. ramp_down hip_yaw: 0.2382 vs 0.0977 -- delta +0.1405 -- SAFE_BUT_WORSE

Original hip_yaw_max = 0.0977. Candidate = 0.2382. Delta = +0.1405.
Tolerance = min(0.05, 2.0 * 0.0977) = min(0.05, 0.1954) = 0.05.
Delta 0.1405 > 0.05 tolerance. Below 0.35 safety gate. Classified: SAFE_BUT_WORSE.
No fall occurred. Height RMSE is slightly better (candidate 0.1123 vs original 0.1149).

### 2. ramp_up hip_yaw: 0.3493 vs 0.0534 -- delta +0.2959 -- SAFETY_FAIL (fell + hyp near limit)

Candidate fell at step 1509 (height_too_low termination). CoM at 0.330m, stuck at initial height.
Hip yaw = 0.3493 is 0.0007 below the 0.35 safety gate but fell=true overrides to SAFETY_FAIL.
Original completed full 5000 steps with no fall and hip_yaw = 0.0534.

### 3. gate_chatter hip_yaw: 0.1791 vs 0.0629 -- SAFE_BUT_WORSE

Original hip_yaw_max = 0.0629. Candidate = 0.1791. Delta = +0.1162.
Tolerance = min(0.05, 2.0 * 0.0629) = min(0.05, 0.1258) = 0.05.
Delta 0.1162 > 0.05. Below 0.35 safety gate. Survived full 5000 steps.
Pitch RMS also worsened: 4.7415 vs 2.98, delta +1.7615 > tol 0.894.

### 4. low_0p300 Step E hip_yaw: 0.2008 vs 0.1314 -- SAFE_BUT_WORSE

Original hip_yaw_max = 0.1314. Candidate = 0.2008. Delta = +0.0694.
Tolerance = min(0.05, 2.0 * 0.1314) = min(0.05, 0.2628) = 0.05.
Delta 0.0694 > 0.05. Below 0.35 safety gate. Classified: SAFE_BUT_WORSE.

### 5. gate_dwell hip_yaw: 0.5370 vs 0.0534 -- delta +0.4836 -- SAFETY_FAIL

Original hip_yaw_max = 0.0534. Candidate = 0.5370. Delta = +0.4836.
Candidate exceeds 0.35 absolute safety gate. Yaw error peaks at 50.95 degrees.
Pitch RMS also severely degraded: 6.1883 vs 3.05, delta +3.1383.

### 6. up_down_cycle: SAFETY_FAIL (fell at step 1186)

Candidate fell at step 1186 with termination reason "height_too_low (0.331 < 0.331)".
Same failure mode as ramp_up: CoM stuck at initial 0.33m, cannot track upward ramp.
Hip yaw = 0.2475 (below 0.35 gate). Original completed full 7000 steps with hip_yaw = 0.0534.

---

## Dynamic Height Root Cause Analysis

### Failure mechanism

The ramp_up and up_down_cycle scenarios both fail with `height_too_low` termination:

- **ramp_up_0p330_to_0p480:** Fell at step 1509. ComZ stuck at 0.330-0.335m. Target height ref reaches 0.380 at termination. Height RMSE only 0.0215m (within floor) but CoM never rises.
- **up_down_cycle_0p330_0p480_0p330:** Fell at step 1186. ComZ stuck at 0.331-0.335m. Same stuck-CoM pattern.

### Root cause

Both the JAX dedicated runner and the original K2 Python controller use `dynamic_qref_mode: "original-k2-exact"`. In this mode, the q_ref (joint reference positions) is computed from a STATIC initial height setup (from the first physics step's CoM height), rather than being dynamically recomputed as the height reference changes.

When the scenario starts at low_0p330 setup (initial CoM ~0.335m), static q_ref is locked to that height. When the height reference ramps up toward 0.48m, the controller's height tracking error grows but the joint reference positions remain at the 0.33m posture. The robot cannot extend its legs to raise CoM because q_ref prevents it.

Specific evidence:
- Candidate com_z_initial = 0.3350m (from low_0p330 setup)
- Candidate com_z remains in [0.330, 0.335] throughout the episode
- Height RMSE = 0.0215m, actually lower than original's 0.1051m -- but this is because the robot never leaves the floor, staying perpetually within the height_too_low termination boundary

### Design limitation

This is a **DESIGN LIMITATION of static q_ref mode** (`original-k2-exact`), not a JAX controller bug. The original K2 Python baseline also uses static q_ref but was validated with `com_z_initial ~0.481m` (high_0p480 setup). In that configuration, ramp_up starts at a high posture and the static q_ref is locked to a standing posture, allowing the robot to simply maintain height -- the original never needed to actually ramp up from a low squat.

The JAX candidate, starting from low_0p330 setup, exposes this limitation because the static q_ref is locked to a squat posture that cannot extend.

### gate_dwell and gate_chatter survival

gate_dwell (0.42->0.45->0.48) and gate_chatter (0.40<->0.47) start from high enough initial positions (com_z_initial ~0.481m from high setup) that the static q_ref provides an already-extended posture. These scenarios survive but exhibit elevated hip yaw.

---

## Per-Scope Summary

### dynamic_height (5 scenarios)

| Class | Count | Scenarios |
|-------|-------|-----------|
| EXACT_OR_BETTER | 0 | -- |
| WITHIN_OLD_TOLERANCE | 0 | -- |
| SAFE_BUT_WORSE | 2 | ramp_down, gate_chatter |
| SAFETY_FAIL | 3 | ramp_up (fell), up_down_cycle (fell), gate_dwell (hy=0.537 > 0.35) |
| NOT_TESTED | 0 | -- |

### long_run (5 scenarios)

| Class | Count | Scenarios |
|-------|-------|-----------|
| EXACT_OR_BETTER | 0 | -- |
| WITHIN_OLD_TOLERANCE | 0 | -- |
| SAFE_BUT_WORSE | 5 | low_0p330, mid_0p400, high_0p430, high_0p450, high_0p480 |
| SAFETY_FAIL | 0 | -- |
| NOT_TESTED | 0 | -- |

### step_c (7 scenarios)

| Class | Count | Scenarios |
|-------|-------|-----------|
| EXACT_OR_BETTER | 0 | -- |
| WITHIN_OLD_TOLERANCE | 5 | C1, C2, C3, C4, focused_high_0p480 |
| SAFE_BUT_WORSE | 2 | C5_long_random, focused_low_0p320 |
| SAFETY_FAIL | 0 | -- |
| NOT_TESTED | 0 | -- |

### step_d (12 scenarios)

| Class | Count | Scenarios |
|-------|-------|-----------|
| EXACT_OR_BETTER | 0 | -- |
| WITHIN_OLD_TOLERANCE | 0 | -- |
| SAFE_BUT_WORSE | 12 | All 12 push conditions |
| SAFETY_FAIL | 0 | -- |
| NOT_TESTED | 0 | -- |

### step_e (10 scenarios)

| Class | Count | Scenarios |
|-------|-------|-----------|
| EXACT_OR_BETTER | 0 | -- |
| WITHIN_OLD_TOLERANCE | 3 | high_0p465, high_0p480, low_0p330 |
| SAFE_BUT_WORSE | 7 | high_0p430, high_0p450, low_0p300, low_0p320, low_0p340, low_0p360, low_0p380 |
| SAFETY_FAIL | 0 | -- |
| NOT_TESTED | 0 | -- |

---

## Cross-Scope Hip Yaw Analysis

Hip yaw divergence is the dominant degradation mode across ALL scopes. The JAX dedicated runner consistently produces higher hip_yaw_max than the original K2 Python baseline, despite both using identical mode_div parameters (kp=10.0, kd=0.50, max_torque=7.5, soft_limit=0.30, soft_gain=0.80).

| Scope | Original hy_max range | Candidate hy_max range | Scenarios exceeding tol |
|-------|----------------------|----------------------|------------------------|
| dynamic_height | 0.0534-0.0977 | 0.1791-0.5370 | 5/5 |
| long_run | 0.0496-0.2048 | 0.1887-0.2213 | 4/5 |
| step_c | 0.0502-0.0851 | 0.0735-0.1823 | 1/7 |
| step_d | 0.0 (no divergence in baseline) | 0.0281-0.3031 | 8/12 |
| step_e | 0.0236-0.1314 | 0.0263-0.2008 | 4/10 |

Note: The step_d baseline reports hip_yaw_max = 0.0 for all push scenarios. This is implausible for a mode_div-enabled controller and may indicate the original push validation did not record hip_yaw or ran with mode_div disabled. The JAX candidate shows hip_yaw = 0.028-0.303 rad, which is consistent with mode_div active behavior.

---

## Final Classification

### K2_JAX_DEDICATED_REALTIME_ORIGINAL_K2_PROMOTION_BLOCKED

**Reason:** SAFETY_FAIL in dynamic_height scope.

Three dynamic_height scenarios trigger SAFETY_FAIL:
1. **ramp_up_0p330_to_0p480:** Fell at step 1509 -- static q_ref design limitation
2. **up_down_cycle_0p330_0p480_0p330:** Fell at step 1186 -- same design limitation
3. **gate_dwell_0p420_0p450_0p480:** hip_yaw_max = 0.5370 exceeds 0.35 safety gate

### Additional finding: Even without dynamic_height, promotion would be PARTIAL, not FULL PASS

ALL four non-dynamic scopes (step_e, step_c, step_d, long_run) contain SAFE_BUT_WORSE scenarios:

| Scope | SAFE_BUT_WORSE count | Total scenarios | SAFE_BUT_WORSE rate |
|-------|---------------------|-----------------|---------------------|
| step_e | 7 | 10 | 70% |
| step_c | 2 | 7 | 29% |
| step_d | 12 | 12 | 100% |
| long_run | 5 | 5 | 100% |

Zero scenarios across all five scopes achieved EXACT_OR_BETTER overall classification. This means the JAX dedicated runner is universally worse than the original Python baseline on at least one metric per scenario.

### Path to unblock

1. **Dynamic height falls:** Resolve by switching `dynamic_qref_mode` from `original-k2-exact` (static q_ref) to `dynamic` (q_ref recomputed from current height reference). This directly addresses the root cause.
2. **gate_dwell hip_yaw > 0.35:** Investigate why hip yaw diverges to 0.537 rad specifically in gate_dwell (high_heights 0.42-0.48) but not in other high-height scenarios like Step E high_0p480 (hy=0.0735). The gate_dwell pattern involves extended dwell at each height, which may allow yaw error to accumulate.
3. **Universal hip_yaw elevation:** The JAX runner produces higher hip_yaw across all scopes. This may be due to numerical differences in the mode_div PID implementation (JAX vs Python floating point), substepping effects, or integration differences. A targeted hip_yaw parity investigation is warranted.
4. **step_d hip_yaw baseline discrepancy:** The original baseline reports hy=0.0 for all push scenarios. Verify whether the original push validation actually ran with mode_div enabled, and re-run if needed to establish a valid comparison baseline.

---

## Data Sources

- Candidate data: `outputs/k2_jax_dedicated_promotion_validation/all_metrics_comparison.json`
- Baseline data: `outputs/k2_original_promoted_baseline/k2_original_metrics.json`
- Classifier: `wheeled_biped/validation/strict_promotion_classifier.py`
- All measured values extracted directly from comparison JSON deltas and raw candidate/original fields.
- Tolerance calculations follow `StrictPromotionClassifier._compute_tolerance()` exactly: `min(absolute, relative * abs(original))`.
