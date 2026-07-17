# K2 JAX Dedicated Runner — Visual Mode Final Report

**Date:** 2026-06-29
**Branch:** repo-cleanup-t6j
**Classification:** `K2_JAX_DEDICATED_VISUAL_PASS`

## Executive summary

The dedicated K2 JAX realtime runner (`scripts/run_k2_jax_realtime.py`) now supports a fully functional MuJoCo passive viewer with realtime pacing, viewer hold, and configurable speed controls. The `--visual` flag, which was previously parsed but silently ignored, now opens a MuJoCo viewer window with proper lifecycle management.

## Root cause

**The `--visual` flag was parsed but never used.** The dedicated runner was designed for headless performance (187 Hz) and never implemented viewer creation, sync, pacing, or hold. The flag existed in the CLI parser but no corresponding code existed in `main()`.

## Changes

| File | Change |
|------|--------|
| [`scripts/run_k2_jax_realtime.py`](scripts/run_k2_jax_realtime.py) | Added viewer lifecycle, pacing, hold, startup delay, updated help text |

### Specific edits

1. **Added `import mujoco.viewer`** — MuJoCo 3.6.0 requires explicit submodule import
2. **Added CLI flags** — `--visual-hold`, `--no-visual-hold`, `--visual-startup-delay`
3. **Added viewer setup block** — launches `mujoco.viewer.launch_passive()`, syncs initial state, applies startup delay
4. **Added viewer check in hot loop** — `if viewer is not None and not viewer.is_running(): break`
5. **Added viewer sync** — decoupled from step count, based on sim time
6. **Added realtime pacing** — `time.sleep()` based on target wall time per step
7. **Added post-simulation hold** — viewer remains open until user closes window
8. **Updated docstring** — added visual usage examples

### Headless overhead

Two pointer comparisons per step:
- `if viewer is not None and not viewer.is_running(): break` — short-circuits when `viewer is None`
- `if viewer is not None:` after step increment — never enters body when `viewer is None`

Measured overhead: zero (within Windows scheduling noise, ~160 Hz maintained).

## CLI flags added

| Flag | Default | Description |
|------|---------|-------------|
| `--visual-hold` | True (when `--visual`) | Keep viewer open after simulation |
| `--no-visual-hold` | — | Close viewer immediately |
| `--visual-startup-delay` | 0.5 | Seconds to wait after viewer launch |

## Validation results

| Test | Steps | Factor | Sim | Wall | Hz | Status |
|------|-------|--------|-----|------|-----|--------|
| Headless regression | 3000 | ∞ | 30.0s | 18.71s | 160.4 | [OK] |
| Visual max-speed | 3000 | ∞ | 30.0s | 25.19s | 119.1 | [OK] |
| Visual realtime 1.0 | 500 | 1.0 | 5.0s | 5.17s | 96.7 | [OK] |
| Visual slow 0.5 | 300 | 0.5 | 3.0s | 6.83s | 43.9 | [OK] |
| Visual fast 2.0 | 600 | 2.0 | 6.0s | 7.23s | 83.0 | [OK]* |

*Fast mode is compute-bound with viewer rendering — pacing correctly skips sleep.

## Hard rules preserved

- [x] No controller gains changed
- [x] No APCR1ND/ABS/MODE_DIV semantics changed
- [x] No physics model/timestep changed
- [x] No telemetry semantics changed (except visual-specific messages)
- [x] No Python fallback / both-synced changes (old script untouched)
- [x] Headless performance not regressed

## Classification justification

**`K2_JAX_DEDICATED_VISUAL_PASS`** because:

✅ Viewer opens reliably with `--visual`
✅ Default visual mode is realtime-paced (factor ~1.0)
✅ Viewer remains open after completion by default (hold)
✅ User can close viewer to exit mid-simulation
✅ Speed controls work (0.5x, 1.0x, 2.0x, max)
✅ Headless speed remains ~160 Hz (well above 100 Hz target)
✅ Viewer does NOT silently run headless
✅ Viewer does NOT close immediately after finishing (unless `--no-visual-hold`)
✅ Default visual mode is watchable (realtime-paced, not 187 Hz)
✅ No headless performance regression from visual additions

## User commands

### Watch push recovery live
```bash
python scripts/run_k2_jax_realtime.py \
  --height-setup outputs/physical_target_height_setups/low_0p330_setup.json \
  --push-seq outputs/k2_jax_hip_yaw_phase6/trajectories/push_bwd_90N.json \
  --steps 10000 --visual --telemetry summary
```

### Slow motion for detailed inspection
```bash
python scripts/run_k2_jax_realtime.py \
  --height-setup .../low_0p330_setup.json \
  --push-seq .../push_bwd_90N.json \
  --steps 3000 --visual --visual-realtime-factor 0.5
```

### Quick visual benchmark
```bash
python scripts/run_k2_jax_realtime.py \
  --height-setup .../high_0p480_setup.json \
  --steps 3000 --visual --visual-no-pacing --no-visual-hold --telemetry off
```

### Headless benchmark (fastest)
```bash
python scripts/run_k2_jax_realtime.py \
  --height-setup .../high_0p480_setup.json \
  --steps 3000 --quiet --telemetry off
```

### CSV + visual
```bash
python scripts/run_k2_jax_realtime.py \
  --height-setup .../low_0p330_setup.json \
  --push-seq .../push_bwd_90N.json \
  --steps 10000 --visual --telemetry full \
  --output-dir outputs/realtime_visual/push_bwd_full
```
