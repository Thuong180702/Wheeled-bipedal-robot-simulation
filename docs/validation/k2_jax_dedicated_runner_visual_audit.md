# K2 JAX Dedicated Runner — Phase 0 Visual Audit

**Date:** 2026-06-29
**Branch:** repo-cleanup-t6j
**Phase:** 0 — Audit Current Visual Implementation

## Root cause

**The `--visual` flag is parsed but silently ignored.** No viewer code exists in `main()`.

### What exists

| Feature | CLI parsed? | Implemented in `main()`? |
|---------|-------------|-------------------------|
| `--visual` | Yes (line 112) | **NO** — dead flag |
| `--visual-sync-hz` | Yes (line 114) | **NO** |
| `--visual-realtime-factor` | Yes (line 116) | **NO** |
| `--visual-no-pacing` | Yes (line 118) | **NO** |
| `--visual-hold` / `--no-visual-hold` | **NO** | **NO** |
| `--visual-startup-delay` | **NO** | **NO** |
| `mujoco.viewer` import | **NO** | **NO** |
| `mujoco.viewer.launch_passive()` call | — | **NO** |
| `viewer.sync()` | — | **NO** |
| `viewer.is_running()` check | — | **NO** |
| Realtime pacing | — | **NO** |
| Viewer hold after sim | — | **NO** |
| Startup delay | — | **NO** |

### Confirmation run

```bash
python scripts/run_k2_jax_realtime.py \
  --height-setup outputs/physical_target_height_setups/low_0p330_setup.json \
  --push-seq outputs/k2_jax_hip_yaw_phase6/trajectories/push_bwd_90N.json \
  --steps 10000 --visual --telemetry summary
```

**Result:**
- No viewer window opened
- Simulation ran headless at 162.2 Hz
- Completed in 22.09s wall time
- Robot fell at step 3582 (height_too_low)
- Script exited immediately

### Missing pieces

1. **No `import mujoco.viewer`** — the submodule is never imported, so `mujoco.viewer.launch_passive` is unavailable (MuJoCo 3.6.0 requires explicit `import mujoco.viewer`)

2. **No viewer object created** — the `--visual` flag value is never read after parsing

3. **No viewer lifecycle** — the hot loop (lines 363–485) has no viewer sync, no `is_running()` check, no viewer launch/close

4. **No realtime pacing** — the simulation runs as fast as possible (~162 Hz) regardless of `--visual`

5. **No post-simulation hold** — the script exits immediately after the loop finishes

6. **Missing CLI flags** — `--visual-hold`, `--no-visual-hold`, `--visual-startup-delay` not defined

### Reference: Monolithic script visual implementation

The monolithic script (`simulate_hierarchical_controller.py`, lines 8848–8963) has a complete viewer lifecycle:

1. Imports `mujoco.viewer` (line 30)
2. Parses pacing flags (8850–8852)
3. Clamps sync Hz to [5, 120] (8859)
4. Computes pacing from `control_dt` (8861–8865)
5. Prints startup banner (8878–8891)
6. Uses context manager: `with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:` (8897)
7. Loop driven by `viewer.is_running()` (8898)
8. Viewer sync decoupled from step count — syncs every `1.0/visual_sync_hz` wall seconds (8912–8921)
9. Realtime pacing with sleep-debt management (8924–8948)
10. Collects visual timing stats (8952–8963)

## Acceptance

- [x] Root cause is identified: `--visual` flag parsed but never used; no viewer code exists
- [x] Confirmed via actual run: viewer does not open
- [x] All missing pieces enumerated
- [x] Reference implementation identified in monolithic script
