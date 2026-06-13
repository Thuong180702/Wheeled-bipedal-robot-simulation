# Physical Standing Height Envelope Definition

**Date:** 2026-06-03  
**Status:** Definition

## Purpose

This document defines the **physical standing height envelope** for the wheeled biped robot: the range of torso CoM heights that are statically feasible given the robot's kinematic and contact constraints, independent of controller stability.

## Physical vs Operational vs Controller-Stable Height

The robot has three nested height envelopes:

1. **Physical standing height envelope** (this document)
   - Pure kinematic and static-equilibrium feasibility
   - Both wheels on floor, no non-wheel collisions
   - CoM projection over support polygon
   - Joint limits respected
   - Symmetric hip-pitch/knee posture with calibrated root_z
   - **No controller constraints applied**
   - **No dynamic stability requirements**

2. **Operational height envelope** (see `operational_height_extreme_validation.md`)
   - Physical envelope + controller readiness constraints
   - Limited to height band where balance-core controller can initialize
   - Applies controller-specific min/max CoM bounds
   - Conservative selection with joint-margin safety buffer

3. **Controller-stable height envelope** (dynamic validation)
   - Operational envelope + dynamic hold/recovery success
   - Validated via Step E position hold and Step C recovery
   - Depends on controller tuning, disturbance profile, and convergence

## Critical principle: dynamic failure does not shrink the physical envelope

The physical envelope is defined by **static feasibility only**.

If a height within the physical envelope fails Step E or Step C validation:
- The height remains **physically valid**
- The failure indicates **controller limitation**, not kinematic impossibility
- The operational/controller-stable envelope may be narrower, but the physical envelope is unchanged

This separation is essential for:
- Controller performance benchmarking
- Future controller improvements that may expand the stable range
- Hardware design evaluation independent of control software

## Static feasibility criteria

A standing height `h` (torso CoM z-coordinate) is **physically feasible** if there exists a symmetric hip-pitch/knee posture such that:

### Contact requirements
- Both left and right wheel geoms achieve floor contact (contact dist < 0)
- No non-wheel body parts contact the floor
- Total vertical wheel contact force > minimum threshold (e.g., 1.0 N)

### Posture requirements
- Hip-pitch and knee joints are symmetric: `left == right`
- Hip-roll near zero (±0.02 rad max)
- Hip-yaw near zero (±0.02 rad max)
- Root z-coordinate is **calibrated from wheel geometry**, not manually set
- The pose is **not root-z-only**: hip/knee must differ from nominal by at least 1e-3 rad when height differs from nominal

### Balance requirements
- CoM (x, y) projection error from support polygon center < threshold (e.g., 0.010 m)
- Pitch/roll angles near equilibrium (|pitch|, |roll| < 0.03 rad)
- Yaw near reference (|yaw| < 0.03 rad)

### Joint limit requirements
- All hip-pitch, knee joints maintain minimum margin from joint limits (e.g., 0.02 rad)

### Structural invariants
- WBC remains off (no hidden torques, no ownership violations)
- Equilibrium joint positions and CoM can be captured from the static pose

## Physical envelope boundaries

The **physical minimum height** `physical_min_height` is the lowest CoM height satisfying all static feasibility criteria.

The **physical maximum height** `physical_max_height` is the highest CoM height satisfying all static feasibility criteria.

These boundaries are:
- **Conservative estimates**, not absolute mechanical limits (we use safety margins in selection)
- **Independent of controller behavior**
- **Independent of Step E / Step C dynamic results**
- **Broader than or equal to** the operational/controller-stable envelopes

## Search method

The physical envelope search uses:

1. **Coarse grid search**
   - Nominal pose from keyframe
   - Coarse height targets from `nominal_com_z - 0.15 m` to `nominal_com_z + 0.15 m` with 0.01 m steps
   - For each target height, search symmetric hip-pitch/knee pairs
   - Calibrate root_z from wheel-floor contact geometry
   - Validate static feasibility

2. **Refinement near boundaries**
   - Once coarse boundary found, refine with 0.002 m steps
   - Ensure boundary precision sufficient for later controller work

3. **Conservative selection**
   - Select extrema with joint-margin safety buffer (e.g., 0.05 rad minimum margin)
   - Do not select the first invalid point; select the last valid point with margin

4. **Artifact generation**
   - CSV grid of all evaluated candidates
   - JSON lists of valid/invalid candidates
   - Setup files for physical_min_height and physical_max_height
   - Static validation summary
   - Human-readable report

## What this envelope does NOT claim

- **Not** claiming absolute mechanical min/max from joint range analysis
- **Not** claiming dynamic stability at the extrema
- **Not** claiming controller readiness across the full envelope
- **Not** validated on hardware
- **Not** incorporating actuator torque limits or power constraints
- **Not** incorporating terrain variation or disturbances

## Relationship to operational envelope

The operational envelope search (already complete) applied controller constraints:
- `controller_min_com_z_m = 0.38 m`
- `controller_max_com_z_m = 0.43 m`

These constraints artificially limited the search range.

The physical envelope search removes these constraints to discover the broader kinematic range.

Expected outcome:
- `physical_min_height <= min_operational_height`
- `physical_max_height >= max_operational_height`

The gap between physical and operational extrema quantifies the controller's height operating margin.

## Use cases

- **Controller benchmarking**: measure what fraction of the physical envelope is controller-stable
- **Future controller work**: identify height ranges where current controller fails but kinematics permit
- **Hardware design**: evaluate whether joint ranges are limiting or whether control is limiting
- **Sim-to-real planning**: understand kinematic workspace before attempting hardware validation

## Validation artifacts

The search produces:
- `physical_height_search_grid.csv` — all evaluated candidates
- `physical_height_valid_candidates.json` — candidates passing static checks
- `physical_height_invalid_candidates.json` — candidates failing static checks with reasons
- `physical_height_envelope_summary.json` — selected min/max and metadata
- `physical_height_envelope_report.md` — human-readable summary
- `physical_min_height_setup.json` — equilibrium setup for min height
- `physical_max_height_setup.json` — equilibrium setup for max height
- `static_physical_extrema_validation.json` — static validation record

If the search cannot confidently identify extrema (e.g., search hits implementation limits before finding invalid poses), the verdict will be `PHYSICAL_ENVELOPE_INCONCLUSIVE` and artifacts will explain why.
