# K2 JAX Actual Push Recovery Validation

> Generated: 2026-06-27
> Phase: 4 — Push Recovery with Real Push Forces
> Profile: `k2_notch_low_q_v1`

---

## 1. Test Configuration

| Parameter | Value |
|-----------|-------|
| Controller mode | balance-core |
| Sagittal controller | velocity-damped |
| Authority profile | k2_notch_low_q_v1 |
| Height setup | high_0p480 (0.480 m target CoM) |
| Push magnitude | ±90 N |
| Push direction | Sagittal-only |
| Push duration | 5 steps (default) |
| Steps | 500 |
| Mode-div enabled | Yes (kp=10.0, kd=0.50, max=7.5, soft_limit=0.30, soft_gain=0.80) |

---

## 2. Results

### Push Forward 90N

| Metric | Python | JAX | Result |
|--------|--------|-----|--------|
| Completed steps | 500 | 500 | ✅ |
| Fall | No | No | ✅ |
| Termination reason | Reached step limit | Reached step limit | ✅ |
| NaN count | 0 | 0 | ✅ |
| Total torque (max) | 14.02 Nm | 10.76 Nm | ✅ |

### Push Backward 90N

| Metric | Python | JAX | Result |
|--------|--------|-----|--------|
| Completed steps | 500 | 500 | ✅ |
| Fall | No | No | ✅ |
| Termination reason | Reached step limit | Reached step limit | ✅ |
| NaN count | 0 | 0 | ✅ |
| Total torque (max) | 15.65 Nm | 9.26 Nm | ✅ |

---

## 3. Acceptance Criteria

| Criterion | Status |
|-----------|--------|
| Python passes | ✅ |
| JAX passes | ✅ |
| No fall | ✅ |
| No NaN | ✅ |
| No actuator limit violations | ✅ |
| No hip-yaw safety violation | ✅ |
| JAX equivalent or better than Python | ✅ |

---

## 4. Commands Used

```bash
# Forward 90N, Python
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core --sagittal-controller velocity-damped \
  --vd-sagittal-authority-profile k2_notch_low_q_v1 \
  --controller-backend python \
  --enable-mode-hip-yaw-divergence ... \
  --height-variant-setup .../high_0p480_setup.json \
  --steps 500 --push-enabled --push-magnitude-n 90 --sagittal-push-only

# Forward 90N, JAX
(same, --controller-backend jax)

# Backward 90N, Python
(same, --push-magnitude-n -90)

# Backward 90N, JAX
(same, --controller-backend jax --push-magnitude-n -90)
```

---

## 5. Conclusion

**PASS — JAX backend survives actual push recovery (forward and backward 90N at 0.480m) without falling, NaN, or actuator violations.**
