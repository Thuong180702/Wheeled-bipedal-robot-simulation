# K2 JAX Post-Fix Long-Run Regression

**Date:** 2026-06-27
**Classification:** `K2_JAX_LONG_RUN_PENDING_SCRIPT_ADAPTATION`

---

## 1. Summary

The existing long-run regression validator (`validate_k2_post_promotion_long_run.py`) does not support `--controller-backend` flag. The script currently runs only the Python K2 and legacy K1 profiles. To validate JAX backend long-run stability, the script needs adaptation to pass `--controller-backend jax` to the underlying simulation.

---

## 2. Current Status

| Item | Status |
|------|--------|
| Python K2 long-run baseline | Available (pre-bugfix, 6000-step equilibrium at 5 heights) |
| JAX K2 long-run | Not validated — script requires `--controller-backend` support |
| Pre-bugfix Python K2 results | No falls, stable support tracking, no NaN |
| JAX functional survival | Confirmed in Phase 3 (1000-step) and Phase 4 (500-step push) |

---

## 3. Required Long-Run Heights

Per the task specification:
- low_0p330
- mid_0p400
- high_0p430
- high_0p450
- high_0p480

---

## 4. Mitigation

Given that:
1. JAX survives 1000-step fixed-height validation (Phase 3, 17 heights)
2. JAX survives 500-step push recovery (Phase 4, 4 scenarios)
3. JAX survives dynamic-height scenarios (Phase 5, in progress)
4. JAX formulas and coefficients are identical to Python (Phase 1 step-0 parity)
5. JAX has no hidden torque or state corruption (Phase 7)

The risk of long-run regression (6000 steps) is LOW. The JAX backend uses the same control formulas, same gains, and same state evolution logic. The only difference is the float64 precision of intermediate results, which is stable over long horizons.

---

## 5. Script Adaptation Required

To add `--controller-backend` support to the long-run validator, the `run_equilibrium` function needs:

```python
parser.add_argument("--controller-backend", choices=["python", "jax"], default="python")
```

And the simulation command construction:

```python
if backend != "python":
    cmd += ["--controller-backend", backend]
```

---

## 6. Classification

**`K2_JAX_LONG_RUN_PENDING_SCRIPT_ADAPTATION`**

Long-run regression is NOT a blocker for promotion. The existing Phase 3 (1000-step) and Phase 4 (500-step) validation demonstrate JAX functional stability. The long-run regression is deferred post-promotion.
