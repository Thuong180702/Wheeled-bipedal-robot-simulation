# Step C Preparation Notes

Planning only. Step C implementation should not start from this note alone.

## Objective

Step C should recover to the commanded target height after a height disturbance or a height-variant initial condition.

Step C must start from the validated Step E controller and extend it for height recovery without breaking nominal standing-position hold.

## Required invariants

Step C must preserve:

- Step E position hold
- hip-yaw posture validity
- WBC-off invariant in the applied torque path
- four-source balance-core ownership
- legacy torque paths off
- ownership violations zero
- hidden torque zero

Do not introduce WBC at Step C start. Only consider WBC later if diagnostics prove that the four-source architecture is insufficient for height recovery after passing E, C, D, and F.

## Initial Step C diagnostic goals

Before any gain change or recovery logic change, Step C should begin with diagnostics and validation gates.

Initial diagnostics should identify:

- target height reference source
- allowed height error band
- recovery time threshold
- position drift limit during recovery
- pitch/roll/yaw safety limits
- contact validity requirements

These should be derived from telemetry and failure modes, not assumed from fixed thresholds alone.

## Starting point

Step C should begin from:

- the validated Step E controller state
- official Step E archival documentation
- current roadmap status in `docs/validation/current_balance_core_roadmap_status.md`

## Constraints

- Step C should begin with diagnostics and validation gates, not immediate gain tuning.
- Do not change Step E gains unless Step C diagnostics prove a specific root cause.
- Do not flip sagittal axis.
- Do not add WBC.
- Do not modify hip-roll.
- Do not change position/sagittal/height logic before defining target and recovery acceptance criteria.
- Do not delete or overwrite any Step E artifacts.
