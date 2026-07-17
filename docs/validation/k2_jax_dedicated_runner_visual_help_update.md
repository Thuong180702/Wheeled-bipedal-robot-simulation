# K2 JAX Dedicated Runner — Phase 5 Help Text Update

**Date:** 2026-06-29
**Branch:** repo-cleanup-t6j
**Phase:** 5 — Update User Commands / Help Text

## Updated docstring

The module docstring now includes these visual usage examples:

```text
  # Visual realtime (default: realtime factor 1.0, hold viewer after sim)
  python scripts/run_k2_jax_realtime.py --height-setup .../setup.json \
    --push-seq .../push.json --steps 10000 --visual --telemetry summary

  # Visual slow motion (0.5x)
  python scripts/run_k2_jax_realtime.py --height-setup .../setup.json \
    --push-seq .../push.json --steps 3000 --visual --visual-realtime-factor 0.5

  # Visual fast (2.0x)
  python scripts/run_k2_jax_realtime.py --height-setup .../setup.json \
    --push-seq .../push.json --steps 3000 --visual --visual-realtime-factor 2.0

  # Visual max-speed benchmark (no pacing, no hold)
  python scripts/run_k2_jax_realtime.py --height-setup .../setup.json \
    --steps 3000 --visual --visual-no-pacing --no-visual-hold --telemetry off

  # Visual with CSV output
  python scripts/run_k2_jax_realtime.py --height-setup .../setup.json \
    --push-seq .../push.json --steps 10000 --visual --telemetry full \
    --output-dir outputs/realtime_visual/push_bwd_full
```

## New CLI flags documented

| Flag | Description |
|------|-------------|
| `--visual` | Launch MuJoCo viewer. Defaults to realtime pacing (factor 1.0) and viewer hold. |
| `--visual-realtime-factor F` | Sim-to-wall speed multiplier. 1.0=realtime, 0.5=half, 2.0=double. |
| `--visual-sync-hz HZ` | Viewer sync rate in Hz (default: 30, range: [5, 120]). |
| `--visual-no-pacing` | Disable all pacing sleep (run as fast as possible). |
| `--visual-hold` | Keep viewer open after simulation (default when `--visual`). |
| `--no-visual-hold` | Close viewer immediately after simulation ends. |
| `--visual-startup-delay S` | Seconds to wait after viewer launch before advancing (default: 0.5). |

## Behavior clarifications

- `--visual` defaults to realtime pacing (factor 1.0) and viewer hold — you watch the robot live
- Headless mode (no `--visual`) runs as fast as possible for benchmarking
- `--visual-no-pacing` is useful for quick visual inspection but NOT for watching live
- `--no-visual-hold` exits immediately after simulation — useful for scripting
- `--telemetry full` records every step to RAM and writes CSV once at end (works in visual mode too)

## Acceptance

- [x] Usage examples added to module docstring
- [x] All visual-related flags documented
- [x] Behavior clarified for each mode
- [x] Headless vs visual behavior distinction clear
