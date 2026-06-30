# K2 JAX Dedicated Runner — Phase 1 Visual Viewer Lifecycle

**Date:** 2026-06-29
**Branch:** repo-cleanup-t6j
**Phase:** 1 — Add Visual Viewer Lifecycle

## Changes

### Added `import mujoco.viewer`

MuJoCo 3.6.0 requires explicit `import mujoco.viewer` to access `launch_passive()`. The submodule is not auto-loaded.

### Viewer setup block (before hot loop)

```python
viewer = None
visual_sync_interval_s = 1.0 / 30.0
visual_realtime_factor = 1.0
visual_disable_pacing = True
visual_hold = False
last_sync_sim_time = -999.0
sim_start_time = 0.0

if args.visual:
    # ... clamp and configure ...
    viewer = mujoco.viewer.launch_passive(mj_model, mj_data)
    viewer.sync()
    if _vstartup_delay > 0:
        time.sleep(_vstartup_delay)
    sim_start_time = time.perf_counter()
```

Design decisions:
- All viewer vars initialized to safe defaults (no viewer, no pacing, no hold)
- `viewer = None` means headless mode — the loop uses `if viewer is not None` guards
- Viewer launched with `launch_passive()` (not context manager) to enable post-simulation hold
- First `viewer.sync()` renders the initial state before any simulation steps
- Startup delay gives the OS window manager time to display the window

### Viewer check in hot loop

```python
while step < max_steps and not terminated:
    if viewer is not None and not viewer.is_running():
        break
```

One pointer comparison per step. Short-circuit ensures `viewer.is_running()` is never called in headless mode.

### Post-simulation viewer hold

```python
if viewer is not None and visual_hold:
    print("Simulation complete. Close viewer to exit.")
    while viewer.is_running():
        viewer.sync()
        time.sleep(0.016)
    viewer.close()
elif viewer is not None:
    viewer.close()
```

- Hold: viewer stays open until user closes the window. Syncs at ~60 Hz.
- No-hold: viewer closes immediately after simulation.
- Headless: skip entirely.

### Error handling

If viewer launch fails (e.g., headless server), MuJoCo raises an exception. The script prints the error and exits — it does NOT silently fall back to headless mode, because the user explicitly requested `--visual`.

## Acceptance

- [x] `--visual` launches a MuJoCo passive viewer window
- [x] Viewer is synced at least once before simulation loop starts
- [x] Viewer remains open after simulation ends (default `--visual-hold`)
- [x] `--no-visual-hold` closes viewer immediately after simulation
- [x] User closing the viewer mid-simulation stops the loop
- [x] Headless mode has zero viewer overhead (just `None` checks)
- [x] Failed viewer launch raises exception (no silent headless fallback)
