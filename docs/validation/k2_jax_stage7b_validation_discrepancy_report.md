# Stage 7B Phase 1: Stage 6L vs Stage 7 Discrepancy Reconciliation

**Date:** 2026-06-27
**Status:** DISCREPANCY EXPLAINED — Stage 6L pass was Python backend only, JAX backend never tested for dynamic height

## Root Cause Summary

Stage 6L's claim of "Dynamic height: 5/5 PASS" was based on K2 **profile** validation running on the **Python backend**. The JAX backend was never tested for dynamic-height scenarios during Stage 6L. Stage 7 was the first time the JAX backend was tested with dynamic-height and push scenarios — and it failed on 5/9 scenarios.

## Detailed Evidence

### Evidence #1: Stage 6L script defaults to Python backend

File: [scripts/validate_k2_dynamic_height_gate_crossing.py:536-537](scripts/validate_k2_dynamic_height_gate_crossing.py#L536-L537)

```python
parser.add_argument("--controller-backend", choices=["python", "jax"], default="python",
                    help="Controller backend (default: python)")
```

The `run_dynamic_scenario()` function only adds `--controller-backend` when `backend != "python"` (line 221):

```python
if backend != "python":
    cmd += ["--controller-backend", backend]
```

The Stage 6L validation was comparing **K1 profile** vs **K2 profile** (different gain settings), both running on the Python backend. It was NOT comparing Python vs JAX backends.

### Evidence #2: dynamic_summary.json has empty paired conditions

File: [outputs/k2_dynamic_height_gate_crossing/dynamic_summary.json](outputs/k2_dynamic_height_gate_crossing/dynamic_summary.json)

```json
{
  "classification": "K2_POST_PROMOTION_LONG_RUN_STRONG_PASS",
  "class_counts": {},
  "k1_count": 0,
  "k2_count": 5,
  "conditions": []
}
```

- `k1_count: 0` — no K1 data was available for comparison
- `conditions: []` — zero paired K1 vs K2 comparisons were made
- The aggregate classifier with empty conditions defaults to `STRONG_PASS` because `better (0) >= equivalent (0)` evaluates to True

### Evidence #3: Fall detection threshold mismatch

The analyzer uses `min_h < 0.20` for fall detection (line 269):

```python
if min_h < 0.20:
    metrics["fell"] = True
```

But the simulation terminates with `height_too_low` before CoM reaches 0.20m:

| Scenario | JAX Fall Step | Min CoM (m) | Analyzer "fell"? | Actual termination |
|----------|--------------|-------------|------------------|--------------------|
| ramp_up | 3066 | 0.282 | False | height_too_low |
| ramp_down | 4159 | 0.429 | False | height_too_low |
| gate_chatter | 1549 | 0.279 | False | height_too_low |

All three failures had `min_h > 0.20`, so the analyzer classified them as "not fell" despite `termination_reason: "height_too_low"`.

### Evidence #4: Telemetry overwrite timeline

1. **Jun 25 22:21-23:28**: Stage 6L K2 Python backend runs completed successfully (telemetry_5000.summary.json shows `"survived_steps": 5000`, `"terminated": false`)
2. **Jun 27 00:26-01:10**: Stage 7 JAX backend runs failed and **overwrote** `telemetry_5000.csv` with truncated data (3067, 4159, 1549 rows)
3. **Jun 27 01:10**: `dynamic_summary.json` rewritten — still shows STRONG_PASS because the analyzer's fall threshold was never triggered

Early (Jun 25) ramp_down K2 summary: `"terminated": false, "survived_steps": 5000, "min": 0.480` ✅
Later (Jun 27) ramp_down K2 summary: `"terminated": true, "survived_steps": 4159, "min": 0.429` ❌

### Evidence #5: Stage 7 runs explicitly use `--controller-backend jax`

File: [scripts/stage7_run_benchmarks.py:124](scripts/stage7_run_benchmarks.py#L124)

```python
args.extend(["--controller-backend", backend])
```

Where `backend` is iterated as `["python", "jax"]` (line 643). The Stage 7 benchmark was the FIRST time the JAX backend was tested against dynamic-height and push scenarios.

## What Stage 6L Actually Validated

| Gate | What was tested | Backend | Result |
|------|----------------|---------|--------|
| Step C (fixed 7 heights) | K2 profile vs K1 torque parity | Python (both profiles) | ✅ |
| Step E (fixed 10 heights) | K2 profile extended | Python | ✅ |
| Step D (push matrix 6) | K2 profile push recovery | Python | ✅ |
| Single push (2) | K2 profile push | Python | ✅ |
| Dynamic height (5) | K1 vs K2 profiles | Python | ✅ |
| Step parity tests (128) | Component-level parity | N/A (no full sim) | ✅ |

**None of these gates tested JAX backend with dynamic-height or push scenarios in closed-loop simulation.**

## Stage 7 Was the First JAX Closed-Loop Dynamic Test

Stage 7 benchmark was the first time the JAX backend ran dynamic-height (ramp_up, ramp_down, gate_chatter) and push scenarios in closed loop. The JAX backend fails on 5/9 scenarios while the Python backend passes all 9.

## Conclusion

**The Stage 6L "5/5 PASS" for dynamic height was valid for what it tested (K2 profile on Python backend), but it was incorrectly interpreted as JAX backend validation. Stage 7 correctly identified a pre-existing JAX backend deficit that was never tested in Stage 6L.**
