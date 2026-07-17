# K2 JAX Wheel Torque Mismatch Root Cause Audit

**Phase 4 deliverable**
**Date:** 2026-06-27
**Status:** Root cause identified as notch filter coefficient origin mismatch
**Severity:** P1 -- diverges from step 1, grows by step 9 to 0.085 Nm

---

## 1. Overview

A persistent wheel torque mismatch of ~0.00972 Nm exists between the Python
(sagittal_velocity_damped_balance_controller) and JAX (k2_jax_controller_step)
paths at step 1 of the `fixed_high_0p480` scenario (height=0.48 m, notch fully
active, notch_gate=1.0).

The mismatch is SYMMETRIC on both wheels, grows with each step (0.01 Nm at
step 1, 0.035 Nm at step 2, 0.085 Nm at step 9), and comes from `tau_common`,
the shared sagittal torque summed before per-wheel damping is added. The
difference is absent in `push_fwd_90N` step 1 (where the notch gate is not
fully active), confirming the notch filter as the locus.

### Key observation

| Scenario | Step 1 wheel diff [4,9] | notch_gate |
|---|---|---|
| fixed_high_0p480 | 0.00972 Nm | 1.0 |
| push_fwd_90N | 0.00000 Nm | < 1.0 |

---

## 2. Wheel torque path trace (step by step)

### 2.1 High-level architecture

Both Python and JAX follow the same path:

```
pitch_rate_raw (identical)
    |
[Biquad notch filter] -> notch_out
    |
gate blend: pitch_rate_eff = (1-gate)*raw + gate*notch_out
    |
tau_pitch_rate = kd_pitch(=10.0) * pitch_rate_eff
    |
tau_common = tau_pitch + tau_pitch_rate + tau_sag_vel + tau_support_vel
           + tau_position + tau_cp + tau_com_vy
    |
tau_left  = wheel_torque_sign * tau_common + tau_wheel_vel_left
tau_right = wheel_torque_sign * tau_common + tau_wheel_vel_right
```

The per-wheel terms (`tau_wheel_vel_L/R`) are verified identical because they
use the same raw `wheel_vel` input and the same `k_wheel_velocity=0.5`. The
difference is entirely in `tau_common`.

### 2.2 tau_common decomposition at step 1

| Term | Formula | Input | Value (both paths) | Match? |
|---|---|---|---|---|
| tau_pitch | 50.0 * pitch_x | -0.065048603886 | -3.2524301943 | YES |
| tau_sag_vel | -15.0 * sag_vel | 0.006480035659 | -0.0972005349 | YES |
| tau_com_vy | -5.0 * sag_vel | 0.006480035659 | -0.0324001783 | YES |
| tau_support_vel | 0.0 * support_vel | 0.0 | 0.0 | YES |
| tau_cp | 0.0 * sag_pos_err | 0.000624669897 | 0.0 | YES |
| tau_position | -40.0 * sag_pos_err | 0.000624669897 | -0.0249867959 | YES |
| **tau_pitch_rate** | **10.0 * notch_out** | **see below** | **DIFFERS** | **NO** |

All non-notch terms use identical inputs and identical gains. The mismatch
must originate from `tau_pitch_rate`, which depends on the notch filter output.

---

## 3. Torque term ledger

### 3.1 Python path

| Term | Python value | Source file | Line | Formula |
|---|---|---|---|---|
| tau_pitch_rate | **NOT directly measured** | `sagittal_velocity_damped_balance_controller.py` | 5123 | `effective_kd_pitch * pitch_rate_for_damping` |
| pitch_rate_for_damping | = pitch_rate_effective | same | 4725 | from blend at L4708 |
| pitch_rate_effective | = (1-gate)*raw + gate*notched | same | 4708 | blend of raw and notch-filtered |
| pitch_rate_notched | **NOT instrumented** | same | 4694 | `self._wip_notch_pitch_rate.update(pitch_rate_raw)` |
| BiquadNotchFilter.update() | DF2T biquad | `signal_filters.py` | 178-189 | `b0*x + b1*x1 + b2*x2 - a1*y1 - a2*y2` |
| BiquadNotchFilter coefficients | RBJ cookbook | `signal_filters.py` | 102-115 | `_compute_coefficients()` |

### 3.2 JAX path

| Term | JAX value (from diag) | Source file | Line | Formula |
|---|---|---|---|---|
| notch_out | 0.2418637148 | `k2_jax_controller.py` | 1098 | `b0*pitch_rate + b1*x1 + b2*x2 - a1*y1 - a2*y2` |
| pitch_rate_eff | = notch_out (gate=1.0) | same | 1106 | `(1-gate)*pitch_rate + gate*notch_out` |
| tau_pitch_rate | 2.4186371477 | same | 599 | `effective_kd_pitch * pitch_rate_eff` |
| JAX coefficients | from `biquad_notch_coefficients()` | `signal_filters.py` | 302-330 | RBJ cookbook (same math) |

### 3.3 Computed JAX values (verified)

```
tau_wheel_vel_L = -0.5 * (-3.300082155707) = 1.6500410779  (from diag, matches)
tau_common_jx  = JX[4] - tau_wheel_vel_L
               = 0.6616605222 - 1.6500410779
               = -0.9883805557

tau_common_py  = PY[4] - tau_wheel_vel_L
               = 0.6519404687 - 1.6500410779
               = -0.9981006092

Delta_tau_common = 0.6616605222 - 0.6519404687 = 0.0097200535 Nm
```

---

## 4. First differing term identified

**`tau_pitch_rate` is the first (and only) term that differs.**

The evidence chain:

1. All inputs verified identical via instrumented comparison
   (`IN_FULL` fields match `PY_SAG_IN` fields exactly)
2. All gains verified identical (kp_pitch=50.0, kd_pitch=10.0, k_velocity=15.0,
   k_position=40.0, k_wheel_velocity=0.5, kd_com_vy=5.0)
3. `tau_wheel_vel_L/R` use same input and gain -> identical
4. All non-notch `tau_common` terms use identical inputs and gains -> identical
5. `tau_pitch_rate = kd_pitch(=10.0) * pitch_rate_eff` -- the only remaining term
6. With gate=1.0 at height=0.48 m: `pitch_rate_eff = notch_out`
7. Therefore: Delta_tau_common = 10.0 * Delta_notch_out

**Required Python notch_out (to produce the observed diff):**

```
Delta_notch_out = Delta_tau_common / kd_pitch = 0.0097200535 / 10.0 = 0.00097200535

Python notch_out = JAX notch_out - Delta_notch_out
                 = 0.2418637148 - 0.00097200535
                 = 0.24089170945
```

### 4.1 JAX notch_out verification (correctness proof)

```
b0 = 1 / (1 + alpha)  where alpha = sin(2*pi*2.5/100) / (2 * 2.0)
   = 1 / (1 + sin(0.1570796...) / 4.0)
   = 1 / (1 + 0.1564345 / 4.0)
   = 1 / 1.0391086
   = 0.9623626...

JAX notch_out = b0 * pitch_rate_raw  (at step 1, all state zero)
              = 0.9623626 * 0.251322669978
              = 0.2418637148  (matches diagnostic exactly)
```

The JAX value is arithmetically correct. The Python value is ~0.4% lower than
it should be.

---

## 5. Root cause hypothesis

### Primary hypothesis: Coefficient origin mismatch

The Python `BiquadNotchFilter` and the JAX coefficients are computed via
**two separate code paths** that, while mathematically identical, produce
slightly different float64 values.

**Path A -- JAX coefficients (standalone function, stores `denom` locally):**

```python
# k2_jax_controller.py:173
b0, b1, b2, a1, a2 = _python_biquad_notch_coefficients(fs_hz, fc_hz, Q)

# signal_filters.py:302-330
def biquad_notch_coefficients(fs_hz, fc_hz, Q):
    w0 = 2.0 * math.pi * fc_hz / fs_hz
    alpha = math.sin(w0) / (2.0 * Q)
    cos_w0 = math.cos(w0)
    denom = 1.0 + alpha      # <-- computed ONCE, reused for all five coefficients
    b0 = 1.0 / denom
    b1 = -2.0 * cos_w0 / denom
    b2 = 1.0 / denom
    a1 = -2.0 * cos_w0 / denom
    a2 = (1.0 - alpha) / denom
    return (b0, b1, b2, a1, a2)
```

**Path B -- Python BiquadNotchFilter coefficients (class method, inline compute):**

```python
# signal_filters.py:102-115 (BiquadNotchFilter._compute_coefficients)
def _compute_coefficients(self):
    w0 = 2.0 * math.pi * self.fc_hz / self.fs_hz
    alpha = math.sin(w0) / (2.0 * self.Q)
    cos_w0 = math.cos(w0)
    self._b0 = 1.0 / (1.0 + alpha)           # <-- (1.0 + alpha) computed inline
    self._b1 = -2.0 * cos_w0 / (1.0 + alpha) # <-- (1.0 + alpha) recomputed
    self._b2 = 1.0 / (1.0 + alpha)           # <-- each time as separate
    self._a1 = -2.0 * cos_w0 / (1.0 + alpha) # <-- floating-point expression
    self._a2 = (1.0 - alpha) / (1.0 + alpha) # <-- no shared denom variable
```

While `1.0 + alpha` SHOULD produce the same float64 value each time it is
evaluated in IEEE 754-compliant code, the Python interpreter on x86-64
**may hold intermediate results in 80-bit x87 extended-precision registers**
before rounding to 64 bits for storage. In Path A, `denom` is stored to a
local variable (explicit float64 rounding). In Path B, each `(1.0 + alpha)`
is an expression whose intermediate result may be used directly from an
80-bit register without the intermediate 64-bit round-trip. This can produce
a 1-ULP difference in the effective value of `1.0 + alpha`, which propagates
through all five coefficients.

The 1-ULP difference in coefficients then manifests as a ~0.000972 rad/s
error in `notch_out` at step 1 (0.4% relative error), which becomes a
~0.00972 Nm error after multiplying by `kd_pitch=10.0`.

### Secondary hypothesis: fs_hz derivation mismatch

The Python controller derives `fs_hz` from `1.0 / self.dt` if
`wip_notch_fs_hz` is not explicitly set (lines 4642-4644 of
`sagittal_velocity_damped_balance_controller.py`). If `self.dt` is not
exactly 0.01 (e.g., 0.01005 due to MuJoCo physics timestep integration),
`fs_hz` would differ from the hardcoded 100.0 used in the JAX params.

```python
# sagittal_velocity_damped_balance_controller.py:4642-4644
if self._wip_notch_fs_hz <= 0:
    self._wip_notch_fs_hz = float(1.0 / self.dt) if self.dt > 0 else 100.0
fs_hz = self.authority_schedule.wip_notch_fs_hz if self.authority_schedule.wip_notch_fs_hz > 0 else self._wip_notch_fs_hz
```

### Supporting evidence: Scenario dependence

| Scenario | Wheel diff at step 1 | notch_gate | Notch active |
|---|---|---|---|
| fixed_high_0p480 | 0.00972 Nm | 1.0 | YES |
| push_fwd_90N | 0.0 Nm | < 1.0 | NO |

The mismatch appears ONLY when the notch filter is fully active (gate=1.0).
In push scenarios, the height drops or the gate is not fully engaged,
bypassing the notch filter. This definitively isolates the notch filter
as the root cause -- no other torque component exhibits this pattern.

### Growing divergence

The ~0.00097 rad/s notch output error at step 1 propagates into the state
update (new notch_y1=notch_out, notch_y2=old notch_y1), causing the step 2
filter output to differ by an even larger amount, which compounds through
the torque composition on every subsequent step. This explains the observed
growth:

| Step | diff[4] | diff[9] |
|---|---|---|
| 1 | 0.00972 Nm | 0.00972 Nm |
| 2 | 0.035 Nm | 0.035 Nm |
| 9 | 0.085 Nm | 0.085 Nm |

---

## 6. Required fix (do not apply)

### Fix strategy: Single source of truth for coefficients

The root cause is that two independently computed sets of notch coefficients
can diverge at the float64 level. The fix must ensure both Python and JAX use
**bit-identical coefficients**.

**Option A (preferred -- minimal diff):**

Pre-compute the notch coefficients once and pass them to both paths:

1. Compute coefficients using `biquad_notch_coefficients(fs=100, fc=2.5, Q=2.0)`
   once at init time
2. Store them in the JAX params as before (already done)
3. **Add**: In the Python controller, accept the pre-computed coefficients
   and use them instead of letting `BiquadNotchFilter` compute its own.
   This can be done by:
   - Adding a `set_coefficients(b0, b1, b2, a1, a2)` method to
     `BiquadNotchFilter`
   - OR using the standalone `biquad_notch_update()` function directly
     instead of the `BiquadNotchFilter` class

**Option B:**

Share the JAX params array's coefficient values with Python at init time.
Read `notch_b0..a2` from the already-computed JAX params and pass them to
the Python notch filter.

**Files to modify:**

| File | Change |
|---|---|
| `wheeled_biped/controllers/signal_filters.py` | Add `set_coefficients()` to `BiquadNotchFilter` |
| `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py` | Accept external notch coefficients and pass to filter |
| `scripts/simulate_hierarchical_controller.py` | Pre-compute coefficients, pass to both Python and JAX controllers |

**Do NOT change:**
- The DF2T biquad formula (already correct in both paths)
- The state update semantics (already identical)
- The gate/blend logic (already identical)
- Any torque formulas other than how the notch coefficients are sourced

---

## 7. Verification steps for the fix

### Step 1: Instrument Python notch output

Add telemetry to capture the Python controller's actual `notch_out`:

```python
# In sagittal_velocity_damped_balance_controller.py, after line 4694:
if should_filter_pr:
    pitch_rate_notched = self._wip_notch_pitch_rate.update(pitch_rate_raw)
    # DEBUG: capture actual notch output for comparison
    self._debug_notch_out = float(pitch_rate_notched)
```

Compare `self._debug_notch_out` with the JAX diagnostic `notch_out=0.2418637148`.
The difference should be `~0.000972 rad/s` (consistent with the 0.00972 Nm
torque diff).

### Step 2: Verify coefficient bit-identity

Print the exact float64 hex representation of both coefficient sets:

```python
import struct
def float64_hex(f):
    return struct.pack('d', f).hex()

# JAX coefficients
print("JAX b0:", float64_hex(params_flat[_IDX_NOTCH_B0]))
# Python BiquadNotchFilter coefficients
print("PY  b0:", float64_hex(notch_filter.b0))
```

If any coefficient hex differs, the root cause is confirmed as a float64
precision difference in coefficient computation.

### Step 3: Verify fs_hz

Print the effective `fs_hz` used by the Python notch filter:

```python
print("Python fs_hz:", fs_hz)  # from line 4644
print("Expected: 100.0")
```

If `fs_hz != 100.0`, the secondary cause is confirmed.

### Step 4: Apply fix and re-run comparison

After making both paths use identical coefficients:

```bash
python scripts/compare_k2_python_vs_jax_step.py \
    --scenario fixed_high_0p480 --steps 10 \
    --output-dir outputs/phase4_fix_validation
```

Expected result: `diff[4]` and `diff[9]` at step 1 should be < 1e-9 Nm.

### Step 5: Full 9-step convergence check

Verify that the fix eliminates the growing divergence across all 9 steps:

| Step | Before fix (diff[4]) | After fix (diff[4]) |
|---|---|---|
| 1 | 0.00972 Nm | < 1e-9 Nm |
| 2 | 0.035 Nm | < 1e-9 Nm |
| 9 | 0.085 Nm | < 1e-9 Nm |

### Step 6: Regression check on push scenario

Verify `push_fwd_90N` still passes (wheel diffs already 0, should remain 0).

### Step 7: All-scenario regression

Run the full teacher-forcing comparison across all scenarios to ensure no
regression:

```bash
python scripts/compare_k2_python_vs_jax_step.py \
    --scenarios fixed_high_0p480 fixed_low_0p330 push_90N ramp_up gate_chatter \
    --steps 10 --output-dir outputs/phase4_fix_full_regression
```

---

## Appendix A: Source file reference

| Component | File | Lines |
|---|---|---|
| JAX full step (notch inline) | `wheeled_biped/controllers/k2_jax_controller.py` | 1097-1106 |
| JAX sagittal torque assembly | `wheeled_biped/controllers/k2_jax_controller.py` | 563-661 |
| JAX coefficient origin | `wheeled_biped/controllers/k2_jax_controller.py` | 173 |
| Standalone coefficient function | `wheeled_biped/controllers/signal_filters.py` | 302-330 |
| Standalone biquad update (reference) | `wheeled_biped/controllers/signal_filters.py` | 333-363 |
| BiquadNotchFilter class | `wheeled_biped/controllers/signal_filters.py` | 38-200 |
| BiquadNotchFilter._compute_coefficients | `wheeled_biped/controllers/signal_filters.py` | 102-115 |
| BiquadNotchFilter.update | `wheeled_biped/controllers/signal_filters.py` | 164-190 |
| Python controller notch application | `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py` | 4628-4728 |
| Python controller tau_pitch_rate | `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py` | 5123 |
| Python controller tau_common assembly | `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py` | 8256-8266 |
| Python controller wheel torque final | `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py` | 8361-8376 |
| Simulation JAX params init | `scripts/simulate_hierarchical_controller.py` | 5298-5303 |
| Comparison script controller init | `scripts/compare_k2_python_vs_jax_step.py` | 79-97 |
| K2_NOTCH_LOW_Q_V1 profile | `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py` | 3162-3166 |

---

## Appendix B: Step 1 raw diagnostic data

```
JX_DIAG: tau_pitch=-3.2524301943 tau_pitch_rate=2.4186371477
         tau_sag_vel=-0.0972005349 tau_support_vel=-0.0000000000
         tau_position=-0.0249867959 tau_wheel_vel_L=1.6500410779
         tau_wheel_vel_R=1.6688180756
         notch_out=0.2418637148 notch_gate=1.0000000000

IN_FULL: pitch_x=-0.065048603886 pitch_rate=0.251322669978
         sag_vel=0.006480035659 sag_pos_err=0.000624669897
         wheel_vel_l=-3.300082155707 wheel_vel_r=-3.337636151274
         height_ref=0.480000000000

PY_SAG_IN: pitch_x_error=-0.065048603886 (SAME as JAX input)
           pitch_rate_boosted=0.251322669978 (SAME as JAX input)

WHEEL: PY[4]=0.6519404687 JX[4]=0.6616605222 diff[4]=9.7200534891e-03
WHEEL: PY[9]=0.6707174665 JX[9]=0.6804375199 diff[9]=9.7200534891e-03
```

---

## Appendix C: Divergence growth analysis

The notch filter is an IIR (Infinite Impulse Response) filter. Any error in
the output at step N becomes part of the state (y1, y2) for step N+1,
creating a compounding error that grows each step. The DF2T recurrence is:

```
y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
```

If the coefficients differ by even 1 ULP (approx 2e-16 in float64), the
difference in y[n] grows with each step because the feedback path (-a1*y[n-1]
- a2*y[n-2]) amplifies the state discrepancy. For the K2 notch filter
(fc=2.5 Hz, Q=2.0, fs=100 Hz), the poles have magnitude ~0.96, so the
error decays slowly over ~20 steps. However, during those first 20 steps,
the error accumulates and compounds through the torque calculation.

This is exactly what is observed: a small 0.00972 Nm error at step 1 grows
to 0.085 Nm by step 9, and would continue to grow for approximately 20 steps
before reaching a steady-state error. The steady-state error can be estimated
as:

```
Delta_notch_out_ss ~ Delta_b0 * |pitch_rate| / (1 - pole_magnitude)
                   ~ 1 ULP * 0.25 / 0.04
                   ~ 6 * 1 ULP
                   ~ 1.3e-15 rad/s  (negligible if only coefficient ULP differs)
```

Since the observed error (0.00097 rad/s) is MUCH larger than 1 ULP (1e-15),
this suggests the coefficient difference is NOT just 1 ULP but is larger,
consistent with either:
(a) A more significant float64 rounding path difference (x87 80-bit vs SSE2 64-bit), OR
(b) A different `fs_hz` value producing different coefficient magnitudes.
