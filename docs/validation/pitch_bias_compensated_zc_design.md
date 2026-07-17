# pitch_bias_compensated_zero_crossing_recenter Design

**Date:** 2026-06-15
**Scope:** Phase 3 — design pitch-bias compensation as an opt-in profile
**Source:** Phases 1–2 audit findings

---

## Classification

**`PITCH_BIAS_COMP_DESIGN_READY`**

All required evidence is in place. Design is ready for implementation.

---

## 1. Mechanism Summary

The robot's forward-pitched equilibrium (±3 to +5 deg) produces a
persistent `tau_pitch` mean of +3.3 Nm across all profiles. Even when
pitch is near zero (`|pitch|<1 deg`, 21-24% of steps), a residual
`tau_pitch = +0.20 to +0.28 Nm` remains — this is the removable DC
component. All four profiles audited show the same behavior: tau_pitch
is a perfect linear function of pitch (correlation +1.000), there is no
controller-injected bias, and the sign convention is symmetric.

The compensation works by:

1. **Estimating** a slow moving average of `tau_pitch` during stable
   posture windows (`|pitch|<2 deg`, contact valid, height safe).
2. **Rate-limiting** the estimate so it converges gradually.
3. **Applying** a bounded subtraction from `tau_pitch` upstream of all
   other logic (after schedule/clip, before blend/suppress).
4. **Decaying** the compensation when pitch deviates from the stable
   window, so dynamic correction is never weakened.

The mechanism does **not** zero tau_pitch, does **not** flip signs, and
does **not** suppress pitch correction — it removes only the slow DC
residual that persists when the robot is near-upright.

---

## 2. Profile Base

`pitch_bias_compensated_zero_crossing_recenter` inherits from
`EARLY_ZERO_CROSSING_RECENTER_V2` with the following additions:

### 2.1 New dataclass fields on SagittalAuthoritySchedule

Add after `ezc_antirebound_*` fields (~line 594):

```python
# Pitch bias DC compensation (Phase 7 new mechanism)
# Removes slow residual tau_pitch DC component during stable upright posture.
# Does NOT zero tau_pitch; does NOT suppress dynamic pitch correction.
pitch_bias_comp_enabled: bool = False       # Master enable
pitch_bias_window_steps: int = 300         # Moving-average window
pitch_bias_max_comp_nm: float = 0.60       # Hard cap on compensation (Nm)
pitch_bias_comp_rate_nm_per_step: float = 0.005  # Rate limit: converge toward estimate
pitch_bias_decay_rate_nm_per_step: float = 0.012  # Decay when pitch leaves stable window
pitch_bias_only_when_abs_pitch_lt_deg: float = 2.0  # Only estimate/apply when upright
pitch_bias_only_when_abs_error_lt_m: float = 0.12  # Only when drift is not severe
pitch_bias_disable_if_pitch_gt_deg: float = 12.0  # Safety disable
pitch_bias_disable_if_roll_gt_deg: float = 5.0     # Safety disable
pitch_bias_disable_if_contact_unstable: bool = True  # Safety disable
pitch_bias_disable_if_height_lt_m: float = 0.25   # Safety disable
pitch_bias_gate_abs_error_soft_m: float = 0.12   # Soft gate: reduce comp when error grows
pitch_bias_gate_abs_error_hard_m: float = 0.20   # Hard gate: disable comp above this
```

### 2.2 State variables on SagittalVelocityDampedBalanceController.__init__

```python
self._pitch_bias_estimate_nm = 0.0   # Slow moving average of tau_pitch in stable windows
self._pitch_bias_samples = 0          # Number of valid samples in current window
self._pitch_bias_comp_tau_nm = 0.0   # Current compensation to subtract from tau_pitch
```

### 2.3 Computation logic

Insert between line 2850 (post-clip `tau_pitch`) and line 2862
(`tau_pitch_before_blend = tau_pitch`):

```python
# =====================================================================
# Pitch Bias DC Compensation (Phase 7 new mechanism)
# Estimate and remove slow tau_pitch DC component during stable upright posture.
# tau_pitch bias is NOT injected by the controller — it is a symptom of the
# forward-pitched equilibrium at this height. Removing only the slow residual
# (not the dynamic response) shifts the equilibrium slightly more upright,
# reducing the net positive drift demand on tau_position.
# =====================================================================

pitch_bias_gate_pass = False
pitch_bias_block_reason = "none"

if self.authority_schedule.pitch_bias_comp_enabled:
    abs_pitch_deg = abs(float(pitch_x_rad)) * 180.0 / 3.141592653589793
    abs_roll_deg = abs(float(roll_y_rad)) * 180.0 / 3.141592653589793 if roll_y_rad is not None else 0.0
    abs_error = abs(float(sagittal_position_error_m))

    # Safety gates
    safety_pass = (
        not self.authority_schedule.pitch_bias_disable_if_contact_unstable or contact_valid
    ) and float(com_z_m) >= self.authority_schedule.pitch_bias_disable_if_height_lt_m
    safety_pass = safety_pass and abs_pitch_deg <= self.authority_schedule.pitch_bias_disable_if_pitch_gt_deg
    safety_pass = safety_pass and abs_roll_deg <= self.authority_schedule.pitch_bias_disable_if_roll_gt_deg

    # Estimation window: only when upright and not too far from zero drift
    estimation_active = (
        abs_pitch_deg < self.authority_schedule.pitch_bias_only_when_abs_pitch_lt_deg
        and abs_error < self.authority_schedule.pitch_bias_only_when_abs_error_lt_m
        and safety_pass
    )

    # Apply gate (softer than estimation — allows partial compensation near the edge)
    gate_soft = abs_error < self.authority_schedule.pitch_bias_gate_abs_error_soft_m
    gate_hard = abs_error >= self.authority_schedule.pitch_bias_gate_abs_error_hard_m
    if gate_hard:
        pitch_bias_gate_pass = False
        pitch_bias_block_reason = "error_hard_gate"
    elif estimation_active:
        pitch_bias_gate_pass = True
        pitch_bias_block_reason = "none"
    elif gate_soft:
        pitch_bias_gate_pass = True
        pitch_bias_block_reason = "near_estimation_window"
    else:
        pitch_bias_gate_pass = False
        pitch_bias_block_reason = "outside_window"

    # Telemetry: record tau_pitch before any compensation
    tau_pitch_before_bias_comp = tau_pitch  # NEW

    if estimation_active:
        # Grow moving average using exponential moving average (EMA)
        # Start from first sample, ramp up with EMA factor
        window_steps = self.authority_schedule.pitch_bias_window_steps
        alpha = min(1.0, 1.0 / window_steps)
        self._pitch_bias_estimate_nm = (
            (1.0 - alpha) * self._pitch_bias_estimate_nm
            + alpha * float(tau_pitch)
        )
        self._pitch_bias_samples += 1

    # Rate-limit the compensation toward the estimate
    # Compensate only a fraction of the estimated bias (bounded, conservative)
    max_comp = self.authority_schedule.pitch_bias_max_comp_nm
    rate = self.authority_schedule.pitch_bias_comp_rate_nm_per_step
    decay = self.authority_schedule.pitch_bias_decay_rate_nm_per_step

    # Estimate of the slow DC bias (the value we want to subtract)
    estimate = max(0.0, self._pitch_bias_estimate_nm)  # Only compensate positive bias

    # Rate-limit toward the target compensation
    target_comp = min(estimate, max_comp)
    if self._pitch_bias_comp_tau_nm < target_comp:
        self._pitch_bias_comp_tau_nm = min(target_comp, self._pitch_bias_comp_tau_nm + rate)
    else:
        # Decay toward zero when not in estimation window
        self._pitch_bias_comp_tau_nm = max(0.0, self._pitch_bias_comp_tau_nm - decay)

    pitch_bias_comp_tau = self._pitch_bias_comp_tau_nm if pitch_bias_gate_pass else 0.0

    # Apply compensation to tau_pitch
    tau_pitch = tau_pitch - pitch_bias_comp_tau
    tau_pitch_clipped = tau_pitch  # telemetry follows same pattern as line 2918

else:
    # No compensation
    pitch_bias_gate_pass = False
    pitch_bias_block_reason = "disabled"
    tau_pitch_before_bias_comp = tau_pitch
    pitch_bias_comp_tau = 0.0
    pitch_bias_estimate_nm = 0.0
    pitch_bias_samples = 0

tau_pitch_after_bias_comp = tau_pitch  # NEW telemetry
```

### 2.4 Telemetry additions

Add to the `diagnostics` dict after `tau_pitch_clipped`:

```python
"pitch_bias_comp_active": bool(pitch_bias_gate_pass),
"pitch_bias_estimate_nm": float(self._pitch_bias_estimate_nm),
"pitch_bias_comp_tau_nm": float(pitch_bias_comp_tau),
"pitch_bias_samples": int(self._pitch_bias_samples),
"pitch_bias_block_reason": str(pitch_bias_block_reason),
"tau_pitch_before_bias_comp": float(tau_pitch_before_bias_comp),
"tau_pitch_after_bias_comp": float(tau_pitch_after_bias_comp),
```

### 2.5 New profile definition

Add to `JOINT_FIX_PROFILES` dict:

```python
"PITCH_BIAS_COMP_ZC" = copy of EARLY_ZERO_CROSSING_RECENTER_V2 plus:
    profile_name="pitch_bias_compensated_zero_crossing_recenter"
    pitch_bias_comp_enabled=True
    pitch_bias_window_steps=300
    pitch_bias_max_comp_nm=0.60
    pitch_bias_comp_rate_nm_per_step=0.005
    pitch_bias_decay_rate_nm_per_step=0.012
    pitch_bias_only_when_abs_pitch_lt_deg=2.0
    pitch_bias_only_when_abs_error_lt_m=0.12
    pitch_bias_disable_if_pitch_gt_deg=12.0
    pitch_bias_disable_if_roll_gt_deg=5.0
    pitch_bias_disable_if_contact_unstable=True
    pitch_bias_disable_if_height_lt_m=0.25
    pitch_bias_gate_abs_error_soft_m=0.12
    pitch_bias_gate_abs_error_hard_m=0.20
```

### 2.6 CLI update

Add `"pitch_bias_compensated_zero_crossing_recenter"` to argparse choices
in `scripts/simulate_hierarchical_controller.py` SAGITTAL_AUTHORITY_PROFILES
and CLI.

---

## 3. Design Rationale

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| window_steps=300 | 5 sec | Long enough to average out pitch oscillations (period ~1-2 sec) but short enough to track slow equilibrium drift |
| max_comp=0.60 Nm | conservative | tau_pitch residual in stable windows is +0.20–0.28 Nm; 0.60 Nm is a 2-3x safety margin |
| comp_rate=0.005 Nm/step | 0.15 Nm/sec | Converges to 0.28 Nm in ~56 steps (fast enough to work in a single 500-step run) |
| decay=0.012 Nm/step | 0.36 Nm/sec | Decays 3x faster than it converges; ensures compensation doesn't persist inappropriately when pitch leaves window |
| abs_pitch_lt=2.0 deg | | Captures the quiescent window where tau_pitch is purely residual, not dynamic |
| abs_error_lt=0.12 m | | When drift is larger, the robot is actively in recenter and bias compensation should be conservative |
| gate_soft=0.12 m | | Below this, full compensation applies |
| gate_hard=0.20 m | | Above this, compensation is disabled entirely |

### 3.1 Why not compensate more aggressively?

- Removing too much tau_pitch risks destabilizing the robot when pitch
  is actually large. The mechanism is gated by `abs_pitch < 2 deg`, so
  it cannot weaken pitch correction when it matters most.
- The estimate grows only during stable windows, so transient pitch
  excursions are never compensated.
- The hard cap of 0.60 Nm means compensation can never exceed the
  measured residual during quiescent periods.

### 3.2 Why not compensate tau_position instead?

- tau_position mean is -3.7 Nm, mostly counterbalancing the large
  tau_pitch (not the residual). Saturating tau_position further by
  compensating it would worsen the stalemate.
- tau_position responds dynamically to drift; bias compensation on it
  would fight the EZC/ZC mechanisms directly.
- The residual tau_pitch (~0.20–0.28 Nm) is the cleanest target.

### 3.3 Interaction with EZC anti-rebound

- EZC anti-rebound provides transient correction (~30 steps) after
  zero crossing.
- Pitch bias compensation operates on tau_pitch continuously during
  stable windows, regardless of EZC state.
- They are orthogonal and additive.

---

## 4. Expected Effect

- **tau_pitch** during stable windows: reduced from +0.20–0.28 Nm toward
  0, reducing the positive drift pressure on the equilibrium.
- **tau_position** residual: should shift slightly less negative (needs
  less counterbalancing), increasing net negative final wheel torque.
- **Drift balance**: positive % should decrease; negative % should
  increase; P2P should remain bounded.
- **Pitch/roll/height**: unchanged under normal conditions (bias
  compensation is disabled when pitch > 2 deg).
- **Safety**: no degradation when pitch > 12 deg or roll > 5 deg
  (hard disable gates).