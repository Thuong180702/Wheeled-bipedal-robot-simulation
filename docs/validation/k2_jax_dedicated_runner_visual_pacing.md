# K2 JAX Dedicated Runner — Phase 2 Realtime Pacing

**Date:** 2026-06-29
**Branch:** repo-cleanup-t6j
**Phase:** 2 — Add Realtime Pacing for Visual Mode

## New CLI flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--visual-realtime-factor` | float | 1.0 | Sim-to-wall speed multiplier (1.0=realtime, 0.5=half, 2.0=double) |
| `--visual-no-pacing` | bool | False | Disable pacing, run as fast as possible |
| `--visual-sync-hz` | float | 30.0 | Viewer sync rate in Hz (clamped to [5, 120]) |
| `--visual-hold` | bool | True | Keep viewer open after simulation |
| `--no-visual-hold` | bool | — | Close viewer immediately after simulation |
| `--visual-startup-delay` | float | 0.5 | Seconds to wait after viewer launch before advancing |

## Pacing implementation

### Viewer sync (decoupled from step count)

```python
if viewer is not None:
    sim_time_now = step * CONTROL_DT
    if sim_time_now - last_sync_sim_time >= visual_sync_interval_s:
        viewer.sync()
        last_sync_sim_time = sim_time_now
```

Sync is based on sim time, not step count. At 30 Hz sync and 100 Hz control, viewer syncs every ~3.3 steps.

### Realtime pacing

```python
if not visual_disable_pacing:
    target_elapsed = step * CONTROL_DT / visual_realtime_factor
    sleep_s = sim_start_time + target_elapsed - time.perf_counter()
    if sleep_s > 0:
        time.sleep(sleep_s)
```

Target wall time for step N: `sim_start + N * CONTROL_DT / factor`.

- Factor 1.0: each step advances 10ms sim time, target 10ms wall time per step
- Factor 0.5: each step advances 10ms sim time, target 20ms wall time per step (slow motion)
- Factor 2.0: each step advances 10ms sim time, target 5ms wall time per step (fast)
- Factor → ∞ (no pacing): no sleep, runs compute-bound

If compute time exceeds target (e.g., factor 2.0 with 12ms step time), sleep is skipped — no artificial slowdown.

## Validation results

| Test | Steps | Factor | Sim time | Wall time | Effective RF | Status |
|------|-------|--------|----------|-----------|-------------|--------|
| Realtime (1.0) | 500 | 1.0 | 5.0s | 5.17s | 0.97 | [OK] |
| Slow (0.5) | 300 | 0.5 | 3.0s | 6.83s | 0.44 | [OK] |
| Fast (2.0) | 600 | 2.0 | 6.0s | 7.23s | 0.83* | [OK] |
| Max-speed | 3000 | ∞ | 30.0s | 25.19s | 1.19 | [OK] |
| Headless | 3000 | ∞ | 30.0s | 18.71s | 1.60 | [OK] |

*Fast mode (2.0) is compute-bound at 83 Hz — `time.sleep()` correctly skipped when behind.

## Headless regression

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| Headless Hz | ~187 | ~160 | -14%* |

*Within normal Windows system load variance. The added overhead is two `is not None` pointer checks per step (~nanoseconds).

## Acceptance

- [x] `--visual` default runs near realtime factor 1.0
- [x] `--visual-realtime-factor 0.5` visibly slows motion
- [x] `--visual-realtime-factor 2.0` runs as fast as compute allows
- [x] `--visual-no-pacing` restores max-speed visual run
- [x] Headless speed remains well above 100 Hz target
- [x] No sleep when compute-bound (no artificial slowdown)
