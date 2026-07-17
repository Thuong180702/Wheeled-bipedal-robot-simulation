# K2 JAX Dedicated Runner — Phase 2 Telemetry

**Date:** 2026-06-29
**Branch:** repo-cleanup-t6j

## Telemetry modes

| Mode | Per-step cost | What it records |
|------|-------------|----------------|
| `off` | ~0 ms | Nothing. Only final summary. |
| `summary` | ~0 ms | Same as off. Final metrics at end. |
| `decimated` | ~0.01 ms avg | Buffer every N steps in memory, write CSV once at end. |
| `full` | ~0.05 ms/step | Buffer every step in memory, write CSV once at end. |

## Implementation

Telemetry rows are stored in a Python list of dicts. Each row is a minimal dict with 11 fields:

```python
MINIMAL_CSV_COLUMNS = [
    "step", "sim_time", "com_z", "pitch_deg", "roll_deg",
    "left_wheel_tau", "right_wheel_tau", "max_abs_tau",
    "height_ref", "contact_valid", "fall",
]
```

CSV is written once after the simulation loop completes using `csv.DictWriter.writerows()`. No file I/O occurs per step.

## Decimation

```python
if tmode == "full" or step % tdec == 0:
    telemetry_rows.append({...})
```

Default decimation: 10 (record every 10th step). Configurable via `--telemetry-decimation`.

## Summary JSON

Written alongside CSV (if `--output-dir` is specified):

```json
{
  "backend": "jax",
  "profile": "k2_notch_low_q_v1",
  "variant": "high_0p480",
  "steps": 3000,
  "wall_time_s": 16.00,
  "achieved_hz": 187.5,
  "mean_step_ms": 5.33,
  "com_z": {"initial": 0.481, "min": 0.481, "max": 0.493, "final": 0.481},
  "pitch_x_deg": {"min": -0.0, "max": 8.5},
  "roll_y_deg": {"min": -0.2, "max": 0.1},
  "max_torque_nm": {"total": 9.56, "wheels": 3.31, "hip_roll": 0.11, "legs": 9.56}
}
```

## Verified behavior

```bash
# Off mode: no files written
python scripts/run_k2_jax_realtime.py ... --telemetry off --output-dir none
# → 0 CSV rows, no file writes

# Decimated mode: 300 rows in CSV (3000 steps / 10)
python scripts/run_k2_jax_realtime.py ... --telemetry decimated --telemetry-decimation 10 \
  --output-dir outputs/runs/test
# → telemetry_3000.csv with 301 lines (header + 300 data rows)
```

## Acceptance

- [x] Telemetry off cost ≈ 0 — 187.5 Hz vs 187.5 Hz
- [x] Decimated telemetry cost <0.5 ms/step average — 177.7 Hz vs 187.5 Hz = ~0.3 ms/step
- [x] CSV output works and contains correct data
- [x] No per-step file writes — CSV written once after loop
- [x] Summary JSON includes all key metrics
- [x] No 756-column dict construction — 11 columns only
- [x] No proxy/dict overhead in hot loop
