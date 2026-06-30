# K2 JAX Dedicated Runner — Phase 3 Visual Startup Delay

**Date:** 2026-06-29
**Branch:** repo-cleanup-t6j
**Phase:** 3 — Add Visual Startup Delay / Initial Sync

## Implementation

### Startup sequence

```python
viewer = mujoco.viewer.launch_passive(mj_model, mj_data)
viewer.sync()                              # render initial state
if _vstartup_delay > 0:
    time.sleep(_vstartup_delay)            # let OS window manager catch up
sim_start_time = time.perf_counter()       # start pacing clock AFTER delay
```

### Rationale

On Windows (and some Linux WMs), `launch_passive()` returns before the window is fully mapped to the display. Without a startup delay, the first few simulation steps execute while the window is still being created, resulting in:

1. A blank/flashing window for the first ~200ms
2. The robot "jumping" to an intermediate state when the window finally renders
3. First `viewer.sync()` potentially being a no-op

The default 0.5s delay gives the window manager adequate time.

### Startup messages

```python
if visual_disable_pacing:
    print("\nLaunching MuJoCo viewer...")
    print("Close the viewer window to end simulation.")
    print(f"Viewer sync: {_vsync_hz:.0f} Hz | Realtime pacing: DISABLED")
else:
    print("\nLaunching MuJoCo viewer...")
    print("Close the viewer window to end simulation.")
    print(f"Viewer sync: {_vsync_hz:.0f} Hz | Realtime factor: {visual_realtime_factor:.1f}")
```

Only printed when `--quiet` is not set. No per-step print spam.

### CLI flag

```
--visual-startup-delay FLOAT   Delay in seconds after viewer launch before
                               advancing simulation (default: 0.5)
```

## Acceptance

- [x] Viewer window has time to appear before simulation advances
- [x] No per-step print spam
- [x] Startup delay is configurable
- [x] Pacing clock starts AFTER delay (correct timing)
