# Step E Height-Variant Position-Hold Audit

## Verdict

- Overall audit verdict: **INCONCLUSIVE**
- Step E nominal remains valid: **true**
- Step E across true height variants passes: **false**
- Controller behavior changed: `false`
- WBC applied: `false` in available historical telemetry; fresh 5000-step audit unavailable

## Root cause for inconclusive official audit

Fresh simulation cannot start because the current working tree has an IndentationError in `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py:43`. The audit did not modify controller files or tune gains.

## Per-variant official verdicts

| Variant | Official 5000-step verdict | Supplemental rows | Support max abs (m) | Support final (m) | Pitch max abs (rad) | Roll max abs (rad) | Height final vs achieved (m) | WBC/ownership |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| nominal | INCONCLUSIVE | 1000 | n/a | n/a | 0.0355427012 | 0.00296542672 | 0.00451205852 | wbc=False, hidden=0, owner=0 |
| low_tiny | INCONCLUSIVE | 1000 | n/a | n/a | 0.0268600845 | 0.00276068189 | 0.00396607007 | wbc=False, hidden=0, owner=0 |
| high_tiny | INCONCLUSIVE | 1000 | n/a | n/a | 0.0441337514 | 0.00318391403 | 0.00495657093 | wbc=False, hidden=0, owner=0 |
| low_small | INCONCLUSIVE | 1000 | n/a | n/a | 0.0337665927 | 0.00295639811 | 0.00384224116 | wbc=False, hidden=0, owner=0 |
| high_small | INCONCLUSIVE | 1000 | n/a | n/a | 0.0371549343 | 0.00298106541 | 0.00491881233 | wbc=False, hidden=0, owner=0 |

## Interpretation

- The requested 5000-step Step E height-variant hold verdict is **INCONCLUSIVE**, not FAIL, because no fresh official telemetry could be produced.
- Historical Step B 1000-step telemetry exists for all variants and is copied as supplemental evidence only; it cannot satisfy the required 5000-step final-verdict criterion.
- No Step C DONE claim is made.

## Recommended next action

Resolve the separate uncommitted syntax error in the ongoing Step C work, then rerun the same diagnostic-only audit to produce fresh 5000-step telemetry for all variants.
