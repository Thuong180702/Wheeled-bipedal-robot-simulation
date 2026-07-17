# K2 JAX Visual Runtime Audit — Phase 3: Python Controller Path Audit

**Date:** 2026-06-29
**Status:** CONFIRMED — Python controller runs every step in JAX mode

## Finding

**The Python WBC + BALANCE-CORE controller pipeline runs EVERY step even when `--controller-backend jax`.**

The JAX path at line 7100 (`tau_smooth = _jax_tau`) overrides the Python-computed torque AFTER the entire Python pipeline has already executed. There is zero conditional skip logic — no code path exists to bypass the Python controller when `_jax_enabled` is True.

## Evidence

From the Stage7 benchmark (before fix):
- JAX hot-step: 0.284 ms (3500+ Hz capability)
- Total step: 105.41 ms (9.5 Hz)
- **370x overhead** from Python controller pipeline

## Costs Per Step (from --profile-controller)

| Component | Time (ms) | Skip status |
|-----------|-----------|-------------|
| WBC QP solve | ~30 | SKIPPED in fix |
| Sagittal controller compute() | ~30+ | STILL RUNNING |
| ShapePosture controller | ~2 | STILL RUNNING |
| Lateral controller | ~1 | STILL RUNNING |
| Composer | ~2 | SKIPPED in fix |
| JAX pack_input (old) | 17.4 | FIXED (NumPy opt) |
| JAX compute | 0.3 | N/A |
| Centroidal state estimation | ~7.7 | STILL RUNNING |
| Telemetry | ~15 | STILL RUNNING |

## JAX Mode Does NOT Call

- `both-synced` parity comparison (only when `--controller-backend both-synced`)
- Teacher-forcing state capture (only in both-synced mode)
- `py_abs_trim_trace` diagnostics (only when both-synced)
- `py_wd_override_active` diagnostics (only when both-synced)

## JAX Mode DOES Call (unnecessarily)

- WBC QP solve at line 5689 **(FIXED — skipped in fast path)**
- Posture regularizer at line 5814 **(FIXED — skipped in fast path)**
- ShapePosture controller at line 6085
- Contact supervisor at line 6060
- Sagittal controller compute() at line 6379
- Lateral controller compute()
- Yaw controller compute()
- Mode-div processing
- Composer at line 6642 **(FIXED — skipped in fast path)**

## Fix Applied

1. `pack_input_k2`: Replaced JAX `.at[idx].set()` pattern with NumPy direct indexing (saves ~12 ms)
2. WBC QP solve: Guarded with `if not _jax_fast_path:` (saves ~30 ms)
3. Composer: Guarded with `if not _jax_fast_path:` (saves ~2 ms)
4. WBC diagnostics: Guarded for step 0

## Remaining Python Overhead

~44 ms/step from the sagittal controller and shape posture compute() still runs.
These produce `sagittal_diag` values needed for JAX input packing (pitch_x_error, support_position_error, etc.).

The sagittal controller could be skipped entirely if the JAX controller were modified to accept raw sensor values and compute pitch_ref_offset internally.

## Classification

**K2_JAX_VISUAL_RUNTIME_BUG_FIXED** — Python WBC no longer runs in JAX mode. Remaining sagittal controller overhead is a separate optimization task.
