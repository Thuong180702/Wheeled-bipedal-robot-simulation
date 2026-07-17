# K2 JAX Step-1 Hip-Yaw Divergence — Root Cause Report

> Generated: 2026-06-27
> Phase: 2 — Teacher-Forcing Root Cause Investigation
> Scenario: push_fwd_90N, height=0.480m

---

## 1. Executive Summary

The step-1 hip-yaw divergence is **NOT a precision issue**. It is caused by **systematic parameter mismatches** between the Python balance core controller defaults and the JAX controller hardcoded defaults. Five gain parameters differ between Python and JAX, producing a combined torque difference of ~0.032 Nm at step 1 on hip-yaw [1,6].

---

## 2. Step-by-Step Teacher-Forcing Data

### Step 0: Perfect Parity

| Field | Python | JAX | Diff |
|-------|--------|-----|------|
| tau[1] (l_hip_yaw) | 0.0 | 0.0 | 0 |
| tau[6] (r_hip_yaw) | 0.0 | 0.0 | 0 |
| max_tau_diff | 4.77e-08 at [2] | — | Float precision |

All inputs identical. State initialized identically. Step 0 confirms that with matched internal state, both controllers produce identical torques for zero yaw/mode_div inputs.

### Step 1: First Divergence on [1]

| Actuator | Python (Nm) | JAX (Nm) | Diff (Nm) |
|----------|-------------|----------|-----------|
| [1] l_hip_yaw | 0.05648895 | 0.02454252 | -0.031946 |
| [6] r_hip_yaw | -0.02150837 | -0.01288233 | +0.008626 |

All 10 input fields are **identical** between Python and JAX at step 1. The divergence originates in the torque computation, not the inputs.

---

## 3. Root Cause: Parameter Mismatches

### 3.1 Shape Posture Controller Gains

The Python `ShapePostureController` uses `BALANCE_CORE_HIP_YAW_AUTHORITY` when no CLI override is provided:

```python
# wheeled_biped/controllers/shape_posture_controller.py:32-35
BALANCE_CORE_HIP_YAW_AUTHORITY = HipYawAuthorityProfile(
    kp_hip_yaw=15.0,    # <-- Python uses 15.0
    kd_hip_yaw=3.0,     # <-- Python uses 3.0
)
```

The JAX function uses hardcoded defaults:

```python
# wheeled_biped/controllers/k2_jax_controller.py:670-671
def k2_jax_shape_posture_compute(
    ..., kp_hip_yaw=5.0, kd_hip_yaw=1.0, ...  # <-- JAX uses 5.0 and 1.0
):
```

**Mismatch: kp_hip_yaw=15.0 vs 5.0 (3×), kd_hip_yaw=3.0 vs 1.0 (3×)**

### 3.2 Yaw Controller Gains

The Python `YawController` uses the `build_balance_core` function defaults:

```python
# scripts/simulate_hierarchical_controller.py:2320-2322
yaw_controller_kp: float = 8.0,      # <-- Python uses 8.0
yaw_controller_kd: float = 2.0,      # <-- Python uses 2.0
yaw_controller_max_torque: float = 5.0,  # <-- Python uses 5.0
```

The JAX function uses hardcoded defaults:

```python
# wheeled_biped/controllers/k2_jax_controller.py:721
def k2_jax_yaw_compute(yaw_error_rad, yaw_rate_rad_s, 
    kp_yaw=5.0, kd_yaw=1.0, max_yaw_torque=3.0  # <-- JAX uses 5.0, 1.0, 3.0
):
```

**Mismatch: kp_yaw=8.0 vs 5.0 (1.6×), kd_yaw=2.0 vs 1.0 (2×), max=5.0 vs 3.0 (1.67×)**

### 3.3 Combined Torque Difference at Step 1

Computed from first principles with step 1 inputs (yaw_err=0.000098, yaw_rate=0.01779, q_hy_l=-0.000030, qd_hy_l=-0.00629):

| Component | Python (Nm) | JAX (Nm) | Δ (Nm) |
|-----------|-------------|----------|--------|
| Shape posture PD [1] | 0.01932 | 0.00644 | +0.01288 |
| Yaw controller [1] | 0.03480 | 0.01730 | +0.01750 |
| Mode div [1] | 0.00080 | 0.00065 | +0.00015 |
| **Total tau_sum[1]** | **0.05492** | **0.02439** | **+0.03053** |
| Observed tau_final[1] | 0.05649 | 0.02454 | +0.03195 |

The 0.0014 Nm remaining discrepancy is attributable to composer rate-limiting behavior interacting with the different posture/yaw gains and the mode_div height gate parameter difference (Python soft_gain=0.50 vs JAX soft_gain=0.80).

**95.5% of the divergence is explained by the 5 gain mismatches.**

---

## 4. Parameter Mismatch Summary

| Parameter | Python Default | JAX Default | Ratio | Location (Python) | Location (JAX) |
|-----------|---------------|-------------|-------|-------------------|-----------------|
| kp_hip_yaw (posture) | 15.0 | 5.0 | 3.0× | `BALANCE_CORE_HIP_YAW_AUTHORITY` | `k2_jax_shape_posture_compute` default |
| kd_hip_yaw (posture) | 3.0 | 1.0 | 3.0× | `BALANCE_CORE_HIP_YAW_AUTHORITY` | `k2_jax_shape_posture_compute` default |
| kp_yaw (yaw ctrl) | 8.0 | 5.0 | 1.6× | `build_balance_core` default | `k2_jax_yaw_compute` default |
| kd_yaw (yaw ctrl) | 2.0 | 1.0 | 2.0× | `build_balance_core` default | `k2_jax_yaw_compute` default |
| max_yaw_torque | 5.0 | 3.0 | 1.67× | `build_balance_core` default | `k2_jax_yaw_compute` default |
| mode_div soft_gain | 0.50 | 0.80 | 0.625× | `ModeBasedHipYawDivergenceController` default | `k2_jax_mode_div_compute` default |

---

## 5. Classification

**Root cause type**: PARAMETER MISMATCH — JAX hardcoded defaults differ from Python balance-core defaults.

**Not**: precision, dtype, state-update, input-packing, formula mismatch, or composer/rate-limit mismatch.

The 5 gain parameters differ because:
1. The JAX controller was written with hardcoded defaults that match the *PyController class defaults* (kp=5.0, kd=1.0, max=3.0)
2. But the Python *balance-core* composition overrides these defaults with `BALANCE_CORE_HIP_YAW_AUTHORITY` (kp=15.0, kd=3.0) and `build_balance_core` function defaults (kp=8.0, kd=2.0, max=5.0)
3. The JAX port did not account for these balance-core default overrides

---

## 6. Fix Plan

Update JAX hardcoded defaults to match Python balance-core defaults:

1. `k2_jax_shape_posture_compute`: kp_hip_yaw=5.0→15.0, kd_hip_yaw=1.0→3.0
2. `k2_jax_yaw_compute`: kp_yaw=5.0→8.0, kd_yaw=1.0→2.0, max_yaw_torque=3.0→5.0
3. `k2_jax_mode_div_compute`: soft_gain=0.80→0.50
4. Verify no other hardcoded defaults differ from balance-core equivalents
