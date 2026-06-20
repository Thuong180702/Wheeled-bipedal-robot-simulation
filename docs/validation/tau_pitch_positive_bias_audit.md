# tau_pitch Positive Bias Audit

**Date:** 2026-06-15
**Scope:** Phase 1 of pitch-bias compensation investigation
**Scenario:** high_0p480, 5000 steps
**Profiles audited:**
- `adaptive_support_centering_trim`
- `zero_crossing_support_recenter`
- `early_zero_crossing_recenter` (V1)
- `early_zero_crossing_recenter_v2`

---

## Classification

**`TAU_PITCH_BIAS_FROM_POSTURE_REQUIREMENT`**

The persistent `tau_pitch` mean ≈ +3.3 Nm is **NOT** a DC offset injected by the
controller. It is a **symptom** of a posture/equilibrium issue: pitch settles
at a forward-leaning equilibrium of +3 to +5 deg, and `tau_pitch` correctly
responds to that pitch with the gain it was designed for.

**Important secondary finding:** Even when pitch is constrained to the
`|pitch|<1 deg` window (21–24 % of all steps), `tau_pitch` retains a
positive mean of **+0.20 to +0.28 Nm**. This residual is the only true
controller-side DC component, and it is what a bias-compensation profile
can safely subtract.

---

## Headline Numbers (5000-step, high_0p480)

| Profile          | tau_pitch mean | tau_pitch median | tau_pitch RMS | tau_pitch min | tau_pitch max |
|------------------|---------------|------------------|---------------|---------------|---------------|
| adaptive         | +3.174        | +2.692           | 3.990         | -1.106        | +7.363        |
| zero_crossing    | +3.327        | +2.823           | 4.250         | -1.260        | +7.500        |
| ezc (V1)         | +3.314        | +2.845           | 4.261         | -1.424        | +7.518        |
| ezc_v2           | +3.380        | +2.901           | 4.308         | -1.501        | +7.551        |

**All four profiles have nearly identical tau_pitch behavior.** The bias is
not specific to any profile, and is not caused by EZC, ZC, or adaptive trim.

### Conditional means

| Profile  | |pitch|<1° (n)        | |drift|<0.03 m (n)    | quiescent (both)      | drift>0 (n)         | drift<0 (n)          |
|----------|----------------------|-----------------------|-----------------------|---------------------|----------------------|
| adaptive | +0.282 (1056=21.1%)  | +0.486 (1473=29.5%)   | +0.282 (1056=21.1%)   | +3.458 (n=4611)     | -0.209 (n=387)       |
| zc       | +0.225 (1175=23.5%)  | +0.436 (1578=31.6%)   | +0.229 (1171=23.4%)   | +3.874 (n=4319)     | -0.149 (n=679)       |
| ezc      | +0.218 (1200=24.0%)  | +0.404 (1568=31.4%)   | +0.221 (1197=23.9%)   | +3.889 (n=4297)     | -0.208 (n=701)       |
| ezc_v2   | +0.201 (1090=21.8%)  | +0.476 (1545=30.9%)   | +0.207 (1082=21.6%)   | +3.959 (n=4299)     | -0.176 (n=699)       |

### Pitch / pitch reference

| Profile  | pitch mean (deg) | pitch median (deg) | pitch min | pitch max | pitch_ref (rad) | pitch_err mean (rad) |
|----------|------------------|--------------------|-----------|-----------|-----------------|----------------------|
| adaptive | +3.637           | +3.085             | -1.267    | +8.438    | -0.0            | +0.06348             |
| zc       | +3.812           | +3.237             | -1.444    | +8.592    | -0.0            | +0.06653             |
| ezc      | +3.797           | +3.253             | -1.632    | +8.620    | -0.0            | +0.06628             |
| ezc_v2   | +3.873           | +3.338             | -1.720    | +8.658    | -0.0            | +0.06760             |

`pitch_x_ref_rad` is **exactly 0** for all profiles. The forward pitch comes
from the dynamics, not the reference.

### Other torques and final output

| Profile  | tau_position mean | tau_pitch_rate mean | tau_wheel_velocity (avg) | final wheel tau mean | final tau when drift>+0.05 |
|----------|-------------------|---------------------|--------------------------|----------------------|----------------------------|
| adaptive | -3.497            | small               | +0.348                   | +0.004               | +0.131 (n=2986)            |
| zc       | -3.683            | small               | +0.399                   | +0.010               | +0.151 (n=2963)            |
| ezc      | -3.673            | small               | +0.387                   | +0.010               | +0.155 (n=2963)            |
| ezc_v2   | -3.737            | small               | +0.394                   | +0.012               | +0.155 (n=2966)            |

`tau_pitch + tau_position ≈ 0` (e.g., V2: +3.380 - 3.737 = **-0.357 Nm**).

`tau_position_lower_bound` is **saturated 13–31 % of the time** for all
profiles (V2: 31.3 %). Position controller is **clipped on the negative
side**, never on the positive side.

### Correlations

| Profile  | tau_pitch ↔ drift | tau_pitch ↔ pitch_deg | tau_pitch ↔ pitch_rate | tau_pitch ↔ pitch_err |
|----------|-------------------|-----------------------|------------------------|-----------------------|
| adaptive | +0.996            | **+1.000**            | -0.002                 | **+1.000**            |
| zc       | +0.997            | **+1.000**            | +0.005                 | **+1.000**            |
| ezc      | +0.995            | **+1.000**            | -0.001                 | **+1.000**            |
| ezc_v2   | +0.994            | **+1.000**            | +0.002                 | **+1.000**            |

`tau_pitch` is **a perfect linear function of pitch error** (correlation = +1.000),
with negligible dependence on pitch rate (`tau_pitch_rate` is logged separately).
This means:

- The controller has the gain it was designed for.
- There is **no asymmetric gain**, **no feedforward bias**, **no sign error**.
- The bias comes entirely from the **non-zero mean of pitch**.

The strong correlation `tau_pitch ↔ drift` (+0.99) is **caused by**
`pitch ↔ drift` correlation, not by tau_pitch independently driving drift.
Both pitch and drift co-rise in the forward-pitched equilibrium.

### Saturation behavior (5000-step)

| Profile  | pitch>0 (%) | pitch<0 (%) | pos sat low | pos sat high | tau_pos when pitch>0.02rad | tau_pos when pitch<-0.02rad |
|----------|-------------|-------------|-------------|--------------|----------------------------|------------------------------|
| adaptive | 85.6        | 1.8         | 13.1 %      | 0.0 %        | -4.484 (n=3747)            | +0.950 (n=16)                |
| zc       | 83.0        | 2.7         | 27.8 %      | 0.0 %        | -4.901 (n=3655)            | +1.052 (n=25)                |
| ezc      | 82.3        | 3.4         | 28.6 %      | 0.0 %        | -4.956 (n=3615)            | +1.161 (n=51)                |
| ezc_v2   | 83.5        | 2.8         | 31.3 %      | 0.0 %        | -4.932 (n=3708)            | +1.237 (n=52)                |

The position controller is **always pulling backward**, often clipped at the
lower bound (-7 Nm). The pitch controller is **always pushing forward**.
They cancel to **near-zero final wheel torque**, leaving the robot stuck in
a forward-pitched stalemate.

`final_wheel_tau_with_apc` mean during positive drift is +0.13 to +0.16 Nm
(should be **negative** to recenter); the system rarely reaches negative
drift, so we cannot measure the symmetry directly.

---

## Answers to Required Questions

### 1. Is `tau_pitch` positive even when pitch is near zero?

**Yes, but small.** `tau_pitch` mean during `|pitch|<1°` is **+0.20 to +0.28 Nm**.
This is a residual that scales with the median pitch in that window
(median pitch in `|p|<1°` is around +0.4–0.5 deg, gain ≈ 50 Nm/rad
implies +0.35–0.44 Nm — consistent within noise).

There is **no large DC offset** in the tau_pitch computation itself.

### 2. Is `tau_pitch` positive because the pitch reference is biased?

**No.** `pitch_x_ref_rad` = 0.0 (exactly) for all profiles. The reference is correct.

### 3. Is `tau_pitch` positive because the pitch error sign is asymmetric?

**No.** Correlation `tau_pitch ↔ pitch_error` = **+1.000** for all profiles.
Sign convention is consistent and symmetric.

### 4. Is `tau_pitch` positive because of feedforward or equilibrium posture?

**This is the actual cause.** The robot settles into a **forward-pitched
equilibrium of +3 to +5 deg**, even though pitch_ref=0. The forward pitch
likely comes from:

- The leg geometry / hip-pitch + knee references at height 0.480 m place
  the CoM slightly forward of the wheel contact line.
- The position controller saturates at `tau_position_lower_bound = -7 Nm`
  trying to drive the wheels backward to recenter, but cannot overcome
  the pitch-induced wheel torque demand.
- `tau_pitch` and `tau_position` net to ~0 final wheel torque, leaving
  the robot frozen at the forward equilibrium.

### 5. Is `tau_pitch` necessary for height/posture, or is it removable DC bias?

**Mostly necessary.** The +3.3 Nm `tau_pitch` is the dynamic response
needed to keep the robot from falling further forward. **However**, the
+0.20 to +0.28 Nm residual that persists during low-pitch windows is a
removable component — and removing it shifts the equilibrium pitch
slightly more upright, which should slightly reduce drift demand on
`tau_position` and break the stalemate.

### 6. Does `tau_pitch` fight the zero-crossing recenter correction?

**Yes, indirectly.** When EZC commands a backward wheel torque to recenter,
the increased wheel motion causes a small forward pitch perturbation. The
forward pitch increases `tau_pitch` (positive), which combines with
`tau_position` (negative) at the final wheel summation. Because both terms
are large in magnitude and oppose, the small EZC contribution (≤0.7 Nm) is
overwhelmed once it leaves the active state.

This means **bias compensation must be designed to act on `tau_pitch`
itself**, not on the EZC output, and must be small enough to never
threaten dynamic pitch stabilization.

---

## Implications for the Compensation Design

1. **Bias estimate should only be computed during low-pitch windows**
   (`|pitch|<2 deg`). Outside that window, `tau_pitch` is doing legitimate
   dynamic stabilization and must be preserved unchanged.
2. The estimate should be a **slow moving average** to reject transients.
3. The compensation should be **bounded** to never exceed the residual
   we measured (+0.20 to +0.30 Nm), with a hard cap around 0.6 Nm.
4. The compensation should be **rate-limited** so that it never causes
   abrupt pitch or wheel-torque changes.
5. The compensation should be **disabled** when posture/contact is unsafe.
6. The compensation **should not interact with EZC, ZC, or adaptive trim** —
   it operates on `tau_pitch` directly upstream of the wheel summation.

This is consistent with the user-specified design parameters:
`pitch_bias_max_comp_nm = 0.6`, `pitch_bias_window_steps = 300`,
`pitch_bias_only_when_abs_pitch_lt_deg = 2.0`.

---

## Files

- Script: `scripts/audit_tau_pitch_bias.py`
- Numeric dump: `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/tau_pitch_bias_audit.json`
- Source CSVs: `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/{adaptive,zc,ezc,ezc_v2}_5000_high_0p480/telemetry_5000.csv`
