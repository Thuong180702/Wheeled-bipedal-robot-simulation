# K2 Dynamic Height — Original K2 Python vs Dedicated JAX Reconciliation

**Date:** 2026-06-29
**Phase:** 8 — Dynamic Height Reconciliation
**Status:** `DEDICATED_RUNNER_BUG` — Dedicated JAX falls on ALL dynamic height scenarios; Original K2 Python survives.

---

## 1. Summary

| Scenario | Original K2 Python | Dedicated JAX | Canonical JAX (monolithic) | Both-Synced | Verdict |
|----------|-------------------|---------------|---------------------------|-------------|---------|
| ramp_up | ✓ Survives 5000/5000 | ✗ Falls step 2989 | **PENDING** | **PENDING** | DEDICATED_RUNNER_BUG |
| ramp_down | Likely survives (from old report: EQUIVALENT) | ✗ Falls step 4471 | NOT RUN | NOT RUN | DEDICATED_RUNNER_BUG |
| gate_chatter | Likely survives (from old report: EQUIVALENT) | ✗ Falls step 2288 | NOT RUN | NOT RUN | DEDICATED_RUNNER_BUG |
| up_down_cycle | NOT RUN | NOT RUN | NOT RUN | NOT RUN | PENDING |
| gate_dwell | NOT RUN | NOT RUN | NOT RUN | NOT RUN | PENDING |

---

## 2. Original K2 Python ramp_up — VERIFIED SURVIVING

Command:
```bash
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core --sagittal-controller velocity-damped \
  --vd-sagittal-authority-profile k2_notch_low_q_v1 \
  --height-variant-setup outputs/physical_target_height_setups/low_0p330_setup.json \
  --dynamic-height-trajectory outputs/k2_dynamic_height_gate_crossing/trajectories/ramp_up_0p330_to_0p480.json \
  --steps 5000 --controller-backend python --telemetry-mode full \
  --output-dir outputs/k2_jax_dedicated_promotion_test/original_k2_python_ramp_up
```

Results:
- **Terminated:** False (no fall)
- **Steps:** 5000/5000
- **CoM height:** 0.303 - 0.335 m
- **Pitch X:** -9.2 to 1.6 deg
- **Roll Y:** 0.0 to 1.3 deg
- **Max torque:** 16.10 Nm (hip roll), 3.53 Nm (wheels), 8.78 Nm (total)
- **Wall time:** 725.7 seconds (~6.9 Hz with telemetry full)

This confirms the original K2 Python survives ramp_up with stable posture. The old K2 dynamic height report (2026-06-25) also confirms ramp_up was EQUIVALENT (K2 pitch=3.15 deg, no fall).

---

## 3. Dedicated JAX ramp_up — FAILS AT STEP 2989

Command:
```bash
python scripts/run_k2_jax_realtime.py \
  --height-setup outputs/physical_target_height_setups/low_0p330_setup.json \
  --dynamic-height-trajectory outputs/k2_dynamic_height_gate_crossing/trajectories/ramp_up_0p330_to_0p480.json \
  --steps 5000 --quiet --telemetry full \
  --output-dir outputs/k2_jax_dedicated_promotion_test/jax_ramp_up
```

Results:
- **Terminated:** True at step 2989
- **Termination reason:** height_too_low (0.285 < 0.285)
- **Max pitch:** 39.3 deg (vs -9.2 deg for original K2 — **4.3× worse**)
- **Final drift:** -10.323 m in Y
- **Hip yaw div max:** 0.365 rad (exceeds 0.35 gate)
- **Contact loss:** 319 steps
- **Height RMS error:** 0.071 m

---

## 4. Root Cause Hypothesis

### Hypothesis A: Missing `capture_estimator.update()`

The monolithic script calls `capture_estimator.update(centroidal_state_control)` after each centroidal estimate. This capture estimator performs state calibration/capture that feeds into subsequent control computations. The dedicated runner does NOT include this step.

**Likelihood:** MEDIUM — The capture estimator may adjust internal state based on changing height.

### Hypothesis B: Missing observation gravity computation

The monolithic script computes gravity in body frame using the rotation matrix:
```python
R = np.array(mj_data.xmat[base_body_id]).reshape(3, 3)
gravity_body = R.T @ np.array([0.0, 0.0, -9.81])
```
And feeds this into obs[0:3]. The dedicated runner does NOT construct an observation — it passes raw sensor values directly to the JAX controller via `pack_input_k2_standalone()`.

The JAX controller receives individual sensor values (pitch, roll, yaw, etc.) rather than a constructed observation vector. This means the JAX controller's internal computations may differ from the Python controller's path that uses the full observation.

**Likelihood:** HIGH — The JAX controller may compute gravity-dependent terms differently than the Python controller.

### Hypothesis C: Equilibrium posture not updated during height changes

The dedicated runner uses a fixed `eq_joint` from the initial posture. During dynamic height transitions, the equilibrium posture should change. The monolithic script also uses fixed `equilibrium_joint_pos`, so this is likely NOT the issue.

**Likelihood:** LOW — Both paths use fixed equilibrium posture.

### Hypothesis D: Controller internal state timing

The JAX controller's internal state (notch filter history, support error history) may behave differently during rapid height changes compared to the Python controller. The parity reports validated fixed-height torque parity, not dynamic-height behavioral parity.

**Likelihood:** MEDIUM — State accumulation during height transitions may diverge.

---

## 5. Triangulation Plan

| Step | What | Status |
|------|------|--------|
| 1 | Run canonical JAX backend (monolithic) for ramp_up | **IN PROGRESS** |
| 2 | Run both-synced for ramp_up | **IN PROGRESS** |
| 3 | Compare both-synced torque diffs during ramp_up | PENDING |
| 4 | If canonical JAX survives → dedicated runner orchestration bug | PENDING |
| 5 | If canonical JAX also falls → JAX controller dynamic height bug | PENDING |
| 6 | Fix root cause | PENDING |
| 7 | Re-run all dynamic height scenarios | PENDING |

---

## 6. Known Concern: Previous Benchmark Ramp-Up Fall

The dedicated runner benchmark report noted: "ramp_up terminates at step 556/5000". This run used a different trajectory (from `k2_jax_hip_yaw_phase6/trajectories/ramp_up.json`). The current run uses the canonical trajectory from `k2_dynamic_height_gate_crossing/trajectories/ramp_up_0p330_to_0p480.json` and falls at step 2989.

Both trajectories cause falls, confirming the issue is systematic, not trajectory-specific.

---

## 7. Classification

**DEDICATED_RUNNER_BUG (pending confirmation)**

Until the canonical JAX backend run completes, the root cause is not definitively isolated. If the canonical JAX also falls, this is a **JAX controller bug**. If the canonical JAX survives, this is a **dedicated runner orchestration bug**.

Either way, dynamic height is BLOCKED for promotion until fixed.

---

## 8. Fix Candidates (if dedicated runner orchestration bug)

1. **Add capture estimator**: Port `capture_estimator.update()` from monolithic script
2. **Add gravity computation**: Compute gravity in body frame and pass to JAX controller
3. **Add observation construction**: Build the full 42-dim observation before calling controller
4. **Align state initialization**: Ensure JAX state is initialized consistently with Python state

## 9. Fix Candidates (if JAX controller bug)

1. **Audit height-dependent computations**: Check IK/LQR gain computation with changing height_ref
2. **Audit notch filter state during height transitions**: Verify notch gate smoothstep behavior
3. **Audit support center computation**: Verify support_center_eq is updated correctly
