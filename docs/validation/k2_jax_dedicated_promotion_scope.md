# K2 JAX Dedicated Realtime Runner — Promotion Scope

**Date:** 2026-06-29
**Phase:** 0 — Freeze Promotion Scope
**Classification:** PENDING (scope defined, verification in progress)

## What Is Being Promoted

| Artifact | Role |
|---|---|
| `scripts/run_k2_jax_realtime.py` | Production realtime K2 JAX simulation runner |
| K2 JAX balance-core realtime simulation | Fixed-height, push recovery, dynamic-height validation |
| Visual watching with realtime pacing | MuJoCo viewer with configurable realtime factor |
| Telemetry off/summary/decimated/full | Buffered write-once CSV output modes |

## Validated Scope (In-Scope for Promotion)

| Dimension | Value |
|---|---|
| Controller mode | `balance-core` only |
| Sagittal controller | `velocity-damped` only |
| Sagittal authority profile | `k2_notch_low_q_v1` / `K2_NOTCH_LOW_Q_V1` only |
| Backend | Standalone JAX (no Python controller, no WBC) |
| Robot model | `assets/robot/wheeled_biped_real.xml` |
| Control rate | 100 Hz (`control_dt = 0.01`) |
| JAX precision | float64 (`jax_enable_x64 = True`) |
| Scenarios | Fixed-height (0.330–0.480 m), push recovery (fwd/bwd), dynamic-height trajectories |
| Visual | MuJoCo viewer with realtime pacing, slow-motion, fast-forward |
| Telemetry | Off / summary / decimated / full — all buffered CSV write-once |

## Explicitly Out of Scope (NOT Promoted)

| Dimension | Reason |
|---|---|
| WBC (Whole-Body Control) | Not implemented in dedicated runner; debug-only in old script |
| Python controller fallback as realtime | Python sagittal is reference/debug only; 55-75 ms/step |
| Both-synced as realtime | Debug/validation mode; carries Python controller overhead |
| Non-K2 profiles | Only `K2_NOTCH_LOW_Q_V1` validated; other profiles not tested |
| Standing-balance controller mode | Not validated for K2 JAX |
| Baseline sagittal controller | Not validated for K2 JAX |
| Non-standalone JAX mode | Python-computed sagittal intermediates not supported |
| Hidden torque injection | Not supported |
| Hardware / sim-to-real | Not validated |
| Stand-up recovery | Not implemented or evaluated |
| Locomotion / stair climbing / rough terrain | Not implemented or evaluated |
| Calibrated outer loop variants | Not in validated K2 scope |
| `K1_*`, `K3_*`, or K-sweep profiles | Not validated |

## Old Script Preservation

`scripts/simulate_hierarchical_controller.py` remains the validation/debug reference:

- Python fallback (reference controller) — always available
- Both-synced mode (teacher-forcing) — debug parity validation
- Full 756-column telemetry — post-analysis
- WBC QP solver — debug/investigation
- All controller profiles — sweep/evaluation

## Target Classification

`K2_JAX_DEDICATED_REALTIME_PROMOTION_PASS`

Acceptance gates (must ALL pass):

1. Parameter parity with canonical K2 JAX path
2. Input contract and state timing match canonical
3. Trace-level torque parity for fixed-height and push scenarios
4. Dynamic-height behavior matches canonical or gap is documented
5. Old both-synced and Python fallback remain valid
6. Dedicated functional regression guard passes (no unexpected falls, no NaN)
7. Headless realtime >100 Hz target or >50 Hz minimum
8. Visual mode works without affecting simulation dynamics
9. Telemetry full writes one row per step to buffered CSV (write-once)
10. No per-step file writes or per-step print in quiet mode
11. Tests pass
12. README/docs updated with correct promotion claims

## Non-Negotiable Rules

- Do NOT claim WBC is supported by the dedicated runner
- Do NOT claim non-K2 profiles are promoted
- Do NOT claim dynamic-height is validated unless trace parity confirms
- Do NOT claim both-synced is a realtime production mode
- Do NOT claim Python fallback meets realtime targets
- If a gate fails, use `PROMOTION_PARTIAL` or `PROMOTION_BLOCKED`
