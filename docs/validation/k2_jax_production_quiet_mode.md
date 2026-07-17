# K2 JAX Production Quiet Mode

**Date:** 2026-06-29
**Branch:** repo-cleanup-t6j
**Phase:** 1

## CLI flags added/modified

### `--quiet` (new)
Suppresses all non-essential terminal output. No per-step progress prints, no per-step diagnostics. Only startup summary and final summary are printed.

In production `backend=jax` mode, `--quiet` is **implied by default** — production runs are silent unless `--verbose` is explicitly set.

### `--verbose` (new)
Overrides `--quiet`. All per-step diagnostic output is shown. Use for debugging only.

### `--telemetry-mode` (new)
Choices: `off`, `summary`, `decimated`, `full`

| Mode | Behavior | Default for |
|------|----------|-------------|
| `off` | No per-step telemetry. Only final wall-clock summary. | — |
| `summary` | ~123 essential fields per step (CoM, orientation, torques, height, status). | — |
| `decimated` | Full 756-column telemetry every N steps, skipped on others. N from `--telemetry-decimation`. | Production JAX (`backend=jax`, not `both-synced`) |
| `full` | Every 756 columns every step. Legacy behavior. | Python fallback, `both-synced`, debug modes |

### `--telemetry-buffered` (new, placeholder)
Collect telemetry rows in memory and write CSV once at end. Already the default behavior — this flag is a no-op for backward compatibility.

### `--telemetry-decimation` (existing, enhanced)
Controls both CSV write frequency AND telemetry population frequency when `--telemetry-mode decimated`.

Default: 10 for production JAX, 1 for everything else.

## Resolution logic

```
if --verbose:
    _quiet = False
elif --quiet:
    _quiet = True
elif backend=jax (not both-synced):
    _quiet = True  # production default
else:
    _quiet = False  # legacy default

if --telemetry-mode:
    use explicit value
elif backend=jax (not both-synced):
    _telemetry_mode = "decimated"
else:
    _telemetry_mode = "full"
```

## `--output-dir` (existing, unchanged)

Already existed. Supports `--output-dir none` to skip all file output.

## Acceptance

- [x] `--quiet` suppresses all per-step prints
- [x] Production JAX defaults to quiet
- [x] Telemetry mode supports off/summary/decimated/full
- [x] `--telemetry-decimation` controls decimation rate
- [x] Existing debug workflows preserved via `--verbose`
- [x] `--output-dir none` skips file writes
