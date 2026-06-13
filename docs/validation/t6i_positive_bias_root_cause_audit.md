# T6I Positive Bias Root-Cause Audit

Date: 2026-06-13  
Profile: `T6I_phase_aware_release`  
Scenario: `high_0p480` 5000-step staged validation  
Telemetry: [telemetry_5000.csv](outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t6i_high_0p480_5000/telemetry_5000.csv)

## 1. Drift column selection
Per requested priority, the audit used:
1. `active_pitch_crossing_signed_error_m` ✅ present and used
2. `sagittal_position_error_m`
3. `support_position_error_m`
4. `hip_yaw_comp_support_error_m`

Chosen physical drift column: `active_pitch_crossing_signed_error_m`.

## 2. Core statistics
- min error: **-0.0287 m**
- max error: **+0.2122 m**
- mean error: **+0.0953 m**
- median error: **+0.0906 m**
- final error: **+0.1309 m**
- positive %: **95.6%**
- negative %: **4.38%**
- zero crossings: **11**

Outside-band occupancy:
- outside ±0.03: **79.2%**
- outside ±0.05: **66.1%**
- outside ±0.08: **53.6%**
- outside ±0.10: **46.7%**
- outside ±0.15: **29.2%**

These values match the previously validated 5000-step summary for T6I and confirm that the remaining issue is not survival but centering precision.

## 3. 500-step window behavior
Windowed mean error / positive occupancy / zero crossings:

- 0-500: mean **+0.0795 m**, positive **84.2%**, zero crossings **3**
- 500-1000: mean **+0.1059 m**, positive **100.0%**, zero crossings **0**
- 1000-1500: mean **+0.0882 m**, positive **100.0%**, zero crossings **0**
- 1500-2000: mean **+0.0949 m**, positive **82.8%**, zero crossings **4**
- 2000-2500: mean **+0.0820 m**, positive **89.0%**, zero crossings **4**
- 2500-3000: mean **+0.1065 m**, positive **100.0%**, zero crossings **0**
- 3000-3500: mean **+0.0882 m**, positive **100.0%**, zero crossings **0**
- 3500-4000: mean **+0.1104 m**, positive **100.0%**, zero crossings **0**
- 4000-4500: mean **+0.0911 m**, positive **100.0%**, zero crossings **0**
- 4500-4999: mean **+0.1065 m**, positive **100.0%**, zero crossings **0**

Late-run summary (steps 2500-4999):
- late mean error: **+0.1005 m**
- late positive %: **100.0%**
- late zero crossings: **0**

Interpretation: the bias is not confined to startup. After mid-run, the response becomes fully one-sided positive and no longer crosses zero.

## 4. Correlation audit
Computed correlations against the selected drift column:

- error vs `t6i_current_cap`: **+0.913**
- error vs `t6i_error_converging`: **+0.059**
- error vs `arch_fix_active`: **+0.894**
- error vs pitch: **+0.996**
- error vs wheel velocity: **+0.016**
- error vs final wheel torque: **+0.826**

Additional summary:
- `t6i_converging_pct`: **6.46%**
- `arch_fix_active_pct`: **46.69%**
- final wheel torque mean: **+0.0121 Nm**
- final wheel torque median: **-0.0506 Nm**

Interpretation:
- The drift is **very tightly coupled to pitch state**.
- It is also strongly coupled to **high-cap / arch-fix-active** periods.
- Wheel velocity itself has almost no explanatory power for the steady bias.
- The final wheel torque does not show a clean persistent positive DC bias; mean torque is near zero. So the issue is not a simple constant output bias.
- T6I convergence detection exists, but its direct correlation with drift is weak and its activation rate is low.

## 5. Answers to required diagnostic questions

### 1. Is the bias persistent across the whole run or mainly early?
**Persistent across the whole run.**
The strongest evidence is that steps 2500-4999 are **100% positive** with **0 zero crossings**.

### 2. Does the error settle around a positive equilibrium?
**Yes.**
Late-run mean error remains around **+0.10 m**, indicating a positive operating offset rather than centered oscillation.

### 3. Is T6I cap decay too rarely active to correct bias?
**Yes.**
`T6I` convergence / release is active only **6.46%** of steps, which is too sparse to function as a centering mechanism.

### 4. Does high authority only prevent runaway but not recenter?
**Yes.**
T6I clearly bounds the drift and prevents falls, but final error remains **+0.1309 m** and late windows remain strongly positive.

### 5. Is there enough negative-side excursion?
**No, not in a practically useful sense.**
There is a small negative excursion down to **-0.0287 m**, but it is shallow and brief relative to the positive side up to **+0.2122 m**. So negative-side motion exists, but it is insufficient to center the behavior.

### 6. Does drift cross zero but quickly return positive?
**Yes.**
There are **11** total zero crossings, but all occur before the late-run regime. After mid-run the trajectory stays positive.

### 7. Does final torque contain a small positive bias?
**Not strongly.**
The final wheel torque mean is only **+0.0121 Nm**, while the median is **-0.0506 Nm**. This does not support the idea of a simple constant torque bias as the primary cause.

### 8. Is there a missing integral/bias correction term?
**Yes, effectively.**
T6I’s existing mechanism is a **release / cap-decay** mechanism, not a slow centering bias corrector. The observed steady positive equilibrium suggests T6I lacks a dedicated mechanism that accumulates long-horizon signed support bias and nudges it back toward zero.

## 6. Root-cause interpretation
The evidence supports this picture:

1. **T6I is stable but not centered.** It keeps the robot bounded and alive.
2. The drift is **pitch-coupled**, not wheel-velocity-driven.
3. The controller enters a **persistent positive support-error equilibrium** at high_0p480.
4. T6I’s phase-aware release can reduce cap occasionally, but it is **not designed to integrate out long-term signed bias**.
5. Therefore, T6I needs a **small, slow, bounded centering trim** layered on top of the existing stabilizing logic, rather than more emergency authority, sign inversion, pitch suppression, or damping suppression.

## 7. Final classification
**T6I_BIAS_FROM_MISSING_CENTERING_INTEGRAL**

This is the best fit because:
- the bias persists late,
- centering does not happen naturally,
- cap decay is too sparse to solve it,
- and the data argues for a missing long-horizon signed-bias correction path rather than a raw sign error or large-output instability.
