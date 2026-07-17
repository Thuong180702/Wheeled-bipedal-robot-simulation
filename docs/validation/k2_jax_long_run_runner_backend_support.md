# K2 JAX Long-Run Runner Backend Support

**Date:** 2026-06-27
**Classification:** `K2_JAX_LONG_RUN_INFRA_READY`

---

## 1. Summary

Added `--controller-backend` support to `scripts/validate_k2_post_promotion_long_run.py`. The runner now accepts `--controller-backend {python,jax}` and passes it through to the underlying simulation command.

## 2. Changes

### 2.1 CLI Argument

```bash
--controller-backend {python,jax}   # default: python
```

### 2.2 Function Signature Changes

| Function | Change |
|----------|--------|
| `run_sim()` | Added `backend` parameter, appends `--controller-backend {backend}` to sim command |
| `run_equilibrium()` | Added `backend` parameter, passes to `run_sim()`, adjusts output directory |
| `run_prbs()` | Added `backend` parameter, passes to `run_sim()`, adjusts output directory |
| `main()` | Passes `args.controller_backend` to run functions |

### 2.3 Output Path Convention

- Python backend: `equilibrium/{height}_K2/` (unchanged)
- JAX backend: `equilibrium/{height}_K2_jax/` (new suffix)

## 3. Backward Compatibility

- Default backend is `python` — all existing behavior preserved
- All existing Python baseline commands work identically
- K1 profile path unchanged
- Height setups, profile, thresholds, and metrics unchanged
- Telemetry analysis functions unchanged

## 4. Usage

```bash
# Python baseline (default — unchanged)
python scripts/validate_k2_post_promotion_long_run.py --suite eq --profile k2

# JAX candidate
python scripts/validate_k2_post_promotion_long_run.py --suite eq --profile k2 --controller-backend jax

# Both backends
python scripts/validate_k2_post_promotion_long_run.py --suite eq --profile both --controller-backend python
# (then run with --controller-backend jax separately, then --report-only to compare)
```

## 5. Long-Run Heights

| Height | Label | Setup File |
|--------|-------|-----------|
| 0.330m | low_0p330 | low_0p330_setup.json |
| 0.400m | mid_0p400 | mid_0p400_setup.json |
| 0.430m | high_0p430 | high_0p430_setup.json |
| 0.450m | high_0p450 | high_0p450_setup.json |
| 0.480m | high_0p480 | high_0p480_setup.json |

## 6. Classification

**`K2_JAX_LONG_RUN_INFRA_READY`**

Infrastructure ready for JAX long-run validation. Actual JAX long-run pending execution (estimated 2-4 hours for 5 heights × 6000 steps). Python baseline available from previous K2 post-promotion runs.
