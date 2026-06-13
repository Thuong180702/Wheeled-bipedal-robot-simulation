# Physical Standing Height Envelope Design

## Goal

Find the robot's **physical/kinematic standing-height envelope** independently of current controller dynamic success, using reusable support-geometry validation based on actual wheel-floor contact positions in world XY. Freeze the lowest and highest statically feasible standing poses first, then use those frozen extrema later for dynamic Step E and Step C validation without shrinking the physical envelope if the controller fails.

## Scope boundary

This design covers only:
- reusable static physical-feasibility geometry and gating,
- static standing-pose search across a broader height range than the existing Step B variants,
- artifact generation for physical extrema,
- static reload/revalidation of the selected extrema,
- documentation of the physical/kinematic envelope.

This design explicitly does **not** cover:
- controller behavior changes,
- torque logic changes,
- WBC changes or enablement,
- dynamic Step E / Step C execution logic,
- shrinking the physical envelope based on controller performance.

## Required files

Create:
- `wheeled_biped/validation/physical_standing_height_envelope.py`
- `scripts/search_physical_standing_height_envelope.py`
- `tests/test_physical_standing_height_envelope.py`
- `docs/validation/physical_standing_height_envelope_definition.md`
- `docs/validation/physical_standing_height_envelope_validation.md`

Do not modify in this phase:
- `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py`
- `scripts/simulate_hierarchical_controller.py`
- `scripts/run_step_c_height_recovery.py`

## Key distinction

### A. Physical / kinematic standing envelope

Selected from **static feasibility only**:
- both wheels in valid floor contact,
- no non-wheel floor contact,
- valid wheel-contact support geometry,
- CoM projection on the wheel support segment,
- small orthogonal support offset,
- near-equilibrium pitch/roll/yaw,
- symmetric standing posture,
- hip/knee within joint limits with margin,
- root_z calibrated from wheel-ground contact,
- not root-z-only.

### B. Controller-stable operational envelope

A later dynamic subset of A, based on whether the current controller can stabilize those statically valid poses. Dynamic failure does **not** redefine the physical envelope.

## Reusable utility design

Primary owner:
- `wheeled_biped/validation/physical_standing_height_envelope.py`

This module is the single owner of:
- wheel contact geometry extraction,
- support segment construction,
- CoM projection onto the support segment,
- lateral / orthogonal offset calculation,
- static feasibility gate evaluation,
- JSON-serializable structured results.

### Separation of concerns

The utility must not know about:
- Step E,
- Step C,
- dynamic pass/fail,
- WBC,
- controller profiles,
- recovery metrics,
- time-series telemetry.

It evaluates one static candidate pose only.

### Contact extraction

Use actual MuJoCo wheel-floor contacts from the current `MjModel` / `MjData` state after `mj_forward` or equivalent forward/contact update.

The utility must explicitly identify:
- left wheel floor contact point,
- right wheel floor contact point,
- non-wheel floor contacts.

If actual wheel-floor contact geometry cannot be found, reject or mark inconclusive with:
- `missing_wheel_floor_contact_geometry`

Do not silently estimate wheel contact points from body positions in the main path. Any fallback must remain separate and explicitly diagnostic.

### Support geometry definition

For the current model, the two wheels are arranged left/right at nearly the same sagittal coordinate. Therefore, “CoM between the wheels” means:
- the CoM projection lies within the left-right wheel contact segment,
- the fore-aft support error is the horizontal offset perpendicular to that segment.

The implementation must avoid hardcoded X/Y semantics when possible.

Given actual wheel contact points in world XY:
- `wheel_line_segment = [left_wheel_contact_xy, right_wheel_contact_xy]`
- `support_center_xy = midpoint(left_wheel_contact_xy, right_wheel_contact_xy)`
- `wheel_line_direction_xy = normalized(right - left)`
- `support_error_direction_xy = normalized(horizontal_perpendicular(wheel_line_direction_xy))`

Then record:
- `support_center_xy`
- `wheel_line_direction_xy`
- `support_error_direction_xy`
- `com_projection_fraction_on_wheel_segment`
- `com_projection_inside_wheel_segment`
- `com_lateral_offset_from_support_center_m`
- `com_sagittal_offset_from_support_center_m`

Where:
- projection fraction is the CoM projection location along the wheel segment,
- lateral offset is the signed displacement along the wheel-line direction relative to the support midpoint,
- sagittal offset is the signed displacement along the perpendicular support-error direction.

Pass/fail uses:
- projection containment,
- `abs(com_sagittal_offset_from_support_center_m)`.

The sign of sagittal offset is diagnostic-only unless a stable convention is explicitly available and documented.

### Degenerate support geometry guard

If the two wheel contact points are too close to define a valid segment:
- do not divide by a near-zero length,
- reject with `degenerate_wheel_support_segment`,
- also record support geometry invalidity.

### Numerical tolerance

Containment should allow a small tolerance around the segment endpoints:
- inside if fraction is within `[-projection_tolerance, 1 + projection_tolerance]`

Still record:
- raw projection fraction,
- endpoint margin if implemented.

Borderline candidates should remain auditable rather than silently normalized.

### Threshold configuration

Use a small threshold/config object or top-level evaluator arguments so thresholds are not buried in the implementation.

Configurable thresholds should include at least:
- `projection_tolerance`
- `preferred_sagittal_offset_m`
- `max_sagittal_offset_m`
- `max_pitch_abs_rad`
- `max_roll_abs_rad`
- `max_yaw_abs_rad`
- `min_joint_limit_margin_rad`

Default expectations for this phase:
- preferred sagittal offset: `<= 0.01 m`
- maximum sagittal offset: `<= 0.02 m`

### Root-z-only provenance

The geometry utility may receive provenance metadata from the search layer because the final pose alone may not reveal whether the candidate was root-z-only.

The static-feasibility result should be able to record:
- `candidate_source`
- `candidate_is_root_z_only`
- whether the candidate came from calibrated hip/knee/root_z search

If root-z-only provenance is disallowed, reject with:
- `root_z_only_candidate_not_allowed`

### Result format

The top-level result should be JSON-serializable through `to_dict()` or equivalent.

It must include:
- `setup_valid`
- `static_feasible`
- `rejection_reasons`
- support geometry metrics
- contact validity metrics
- posture metrics
- joint-limit margins if available
- candidate provenance metadata

## Search script design

Primary owner:
- `scripts/search_physical_standing_height_envelope.py`

This script orchestrates:
- candidate generation,
- root_z calibration,
- calling the shared utility,
- artifact writing,
- static physical envelope selection,
- static reload/revalidation.

It must not redefine support geometry locally.

### Candidate generation

The search must be broader than the current Step B variants and continue below `low_small` and above `high_small` until static feasibility fails for physical reasons.

Each candidate must:
- use symmetric leg posture search,
- set `l_hip_pitch == r_hip_pitch`,
- set `l_knee == r_knee`,
- keep hip-yaw near neutral/reference,
- keep hip-roll near neutral/reference unless explicitly justified,
- calibrate `root_z` from wheel-ground contact after setting posture,
- never be generated by root-z-only shifting.

Each candidate must record:
- `candidate_source`
- `candidate_is_root_z_only`
- `calibrated_root_z_m`
- `hip_pitch_ref`
- `knee_ref`
- `achieved_com_z_m`
- `requested_target_com_z_m`

### Search breadth and stopping behavior

The script should search sufficiently broadly to reveal the static/kinematic range, not merely confirm controller-ready variants.

Validity should fail only for physical reasons such as:
- joint limit margin too small,
- non-wheel contact,
- missing or invalid wheel contact,
- projection outside support segment,
- sagittal offset too large,
- pitch/roll/yaw out of bounds,
- degenerate support geometry.

### Selection rule

Select:
- `physical_min_height = lowest static_feasible candidate`
- `physical_max_height = highest static_feasible candidate`

This selection must depend only on static feasibility.

The script must not use:
- Step E results,
- Step C results,
- controller hold success,
- wheel velocity behavior,
- recovery behavior.

### Tie-breaking within the same height

If multiple candidates exist at the same achieved height, choose the one with the best static quality using a deterministic order such as:
1. smaller absolute sagittal offset,
2. better endpoint margin / projection margin,
3. larger joint-limit margin,
4. smaller orientation error.

Alternatives may still be recorded in JSON artifacts for auditability.

### Artifacts

Write:
- `outputs/physical_standing_height_envelope_search/physical_height_search_grid.csv`
- `outputs/physical_standing_height_envelope_search/physical_height_valid_candidates.json`
- `outputs/physical_standing_height_envelope_search/physical_height_invalid_candidates.json`
- `outputs/physical_standing_height_envelope_search/physical_height_envelope_summary.json`
- `outputs/physical_standing_height_envelope_search/physical_height_envelope_report.md`
- `outputs/physical_standing_height_envelope_search/physical_min_height_setup.json`
- `outputs/physical_standing_height_envelope_search/physical_max_height_setup.json`
- `outputs/physical_standing_height_envelope_search/static_physical_extrema_validation.json`

The report must clearly separate:
- search coverage,
- valid static envelope,
- invalid candidate rejection reasons,
- selected physical extrema,
- static revalidation result,
- dynamic validation not yet performed in Part 3.

### Explicit rejection reasons

Use clear machine-readable reasons, including:
- `missing_wheel_floor_contact_geometry`
- `degenerate_wheel_support_segment`
- `projection_outside_wheel_segment`
- `sagittal_support_offset_too_large`
- `non_wheel_floor_contact`
- `joint_limit_margin_too_small`
- `root_z_only_candidate_not_allowed`
- `pitch_roll_yaw_out_of_bounds`
- `support_geometry_invalid`

### Static revalidation

After selecting `physical_min_height_setup.json` and `physical_max_height_setup.json`, reload and revalidate both setups using the same shared utility.

If reload/revalidation fails:
- mark `PHYSICAL_ENVELOPE_INCONCLUSIVE`
- do not proceed to dynamic validation.

## Documentation requirements

### `docs/validation/physical_standing_height_envelope_definition.md`

This document should define:
- the distinction between the physical/kinematic envelope and the controller-stable envelope,
- the corrected support-geometry interpretation for the current left/right wheel layout,
- the morphology-independent projection method,
- static feasibility criteria,
- provenance/root-z-only rejection rule,
- why dynamic controller failure must not shrink the physical envelope.

### `docs/validation/physical_standing_height_envelope_validation.md`

This document should report:
- search method,
- search coverage,
- physical min/max CoM heights found,
- root_z / hip_pitch / knee at each extreme,
- support geometry metrics at each extreme,
- joint-limit margins,
- static revalidation result,
- explicit statement that dynamic Step E / Step C is not part of Part 3,
- later dynamic results must be reported separately without redefining the physical extrema.

## Tests

Create `tests/test_physical_standing_height_envelope.py`.

Required coverage:
1. Current left/right wheel geometry is handled correctly.
2. CoM projection inside the wheel segment passes.
3. CoM projection outside the segment fails.
4. Sagittal / orthogonal offset above threshold fails.
5. The utility does not assume hardcoded X/Y axes.
6. Synthetic front/back wheel geometry is still interpretable by the same projection method.
7. Root-z-only candidates are rejected.
8. Search-layer static extrema selection depends only on static feasibility.
9. Dynamic failure metadata cannot shrink physical extrema.
10. Selected extrema can be serialized, reloaded, and revalidated.
11. Rejection reasons are preserved in artifacts.
12. The search script uses the shared utility rather than duplicating support geometry logic.

## Success criteria for Part 3

Part 3 is complete when:
- the shared validation utility exists and owns the geometry/gating logic,
- the search script uses that utility for all static support evaluation,
- physical extrema are selected from static feasibility only,
- selected extrema are serialized and successfully revalidated,
- required artifacts are written,
- controller behavior remains unchanged,
- WBC remains unchanged and unapplied in this phase,
- dynamic Step E / Step C are not run yet.

## Non-goals

This phase must not claim:
- absolute mechanical limits unless the search exhaustively proves them,
- controller-stable extrema as physical extrema,
- dynamic success at the selected physical extrema,
- Step E / Step C completion for these extrema.

## Planned stop point

After Part 3 implementation, stop and report:
- files created/updated,
- tests run and results,
- `physical_min_height` found,
- `physical_max_height` found,
- whether static revalidation passed,
- artifacts created,
- `controller behavior changed: false`,
- `WBC added/applied: false`.
