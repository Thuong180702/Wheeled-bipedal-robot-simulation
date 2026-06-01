# Current Balance-Core Roadmap Status

Updated: 2026-06-01

## Roadmap status

| Step | Status | Notes |
|---|---|---|
| Step A — Long-run nominal balance | DONE | Completed before Step E validation. |
| Step B — True standing-height variants | DONE | Completed before Step E validation. |
| Step E — Position hold / position return | DONE | Official production validation v2 passed on `outputs/hierarchical_controller_sim/telemetry_1780289121.csv`. |
| Step C — Height recovery to target height | CURRENT | Next roadmap step. Begin with diagnostics and validation gates; do not implement yet from this status update. |
| Step D | BLOCKED | Still blocked until Step C passes. |
| Step F | BLOCKED | Still blocked until Step C passes. |

## Step E completion reference

- Archive: `docs/validation/step_e_done_2026-06-01.md`
- Validation output directory: `outputs/step_e_official_validation_v2/`
- Final decision: `STEP_E_DONE`

## WBC status

WBC has not been added to the applied balance-core torque path.

The official Step E telemetry contains nonzero raw `tau_wbc_norm`, but v2 validation shows this is diagnostic-only:

- Applied WBC contribution norm max: `0.0`
- WBC applied: `false`
- WBC contributed to `tau_total_raw_per_joint`: `false`
- `active_torque_owner_per_joint` includes WBC: `false`

WBC should still not be added at Step C start. Only reconsider WBC after balance-core passes E, C, D, and F, or after verified authority insufficiency with diagnostics proving that the four-source balance-core architecture cannot satisfy the requirement.

## Guardrails for current work

- Start Step C from the validated Step E controller.
- Preserve Step E position hold.
- Preserve hip-yaw posture validity.
- Preserve WBC-off invariant in the applied torque path.
- Preserve four-source balance-core ownership.
- Do not do further Step E tuning unless future Step C, Step D, or Step F regressions identify a Step E root cause.
