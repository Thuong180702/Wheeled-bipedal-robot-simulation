# Physical Standing Height Envelope Validation

**Date:** 2026-06-03  
**Status:** **PASS**

## Summary

The physical standing height envelope search identified the kinematic range of statically feasible torso CoM heights for the wheeled biped robot, independent of controller constraints.

**Physical envelope:**
- **Minimum height:** 0.291919 m CoM
- **Maximum height:** 0.490812 m CoM
- **Envelope span:** 0.198893 m (19.9 cm)

**Comparison to operational envelope:**
- Operational min: 0.393287 m (10.1 cm **higher** than physical min)
- Operational max: 0.412813 m (7.8 cm **lower** than physical max)
- Physical envelope is **significantly broader** than operational envelope

This demonstrates that the robot's kinematic workspace is larger than the current controller's operational range. The gap quantifies potential for controller improvement.

## Validation verdict

**PHYSICAL_ENVELOPE_PASS**

- Static search completed: 61 target heights evaluated
- Valid candidates found: 61 (100%)
- Invalid candidates: 0
- Static revalidation: PASS for both extrema

## Physical envelope definition

The physical standing height envelope captures heights that are **kinematically and statically feasible**, defined by:

1. **Contact requirements:** Both wheels in floor contact, no non-wheel collisions
2. **Posture requirements:** Symmetric hip-pitch/knee, calibrated root_z from wheel geometry
3. **Balance requirements:** CoM projection over support polygon, upright posture
4. **Joint limit requirements:** Minimum margin from joint limits

**Critical principle:** This envelope is based on **static feasibility only**. No controller constraints or dynamic stability checks were applied. Dynamic failure at these extrema does NOT invalidate the physical envelope — it indicates controller limitation, not kinematic impossibility.

## Physical minimum height: 0.291919 m

Static pose characteristics:

| Parameter | Value |
|---|---:|
| Achieved CoM z | 0.291919 m |
| Hip pitch | 1.226052 rad (70.2°) |
| Knee | 2.348364 rad (134.5°) |
| Root z | 0.398301 m |
| Joint limit margin | 0.351636 rad (20.1°) |
| Left wheel contact | true |
| Right wheel contact | true |
| Non-wheel floor contacts | 0 |
| Support width | 0.347 m |

This represents a deeply crouched pose with significantly flexed hip and knee joints. The robot maintains wheel-only floor contact and upright posture despite the extreme height.

**Gap from operational min:** The operational minimum (0.393287 m) is 10.1 cm **higher** than the physical minimum. This gap exists because the operational search applied `controller_min_com_z_m = 0.38 m` as an artificial floor, intentionally excluding lower heights from controller readiness evaluation.

## Physical maximum height: 0.490812 m

Static pose characteristics:

| Parameter | Value |
|---|---:|
| Achieved CoM z | 0.490812 m |
| Hip pitch | 0.626052 rad (35.9°) |
| Knee | 1.148364 rad (65.8°) |
| Root z | 0.642381 m |
| Joint limit margin | 1.126052 rad (64.5°) |
| Left wheel contact | true |
| Right wheel contact | true |
| Non-wheel floor contacts | 0 |
| Support width | 0.297 m |

This represents a nearly upright standing pose with more extended hip and knee joints. The robot maintains balance with significantly more joint margin available.

**Gap from operational max:** The operational maximum (0.412813 m) is 7.8 cm **lower** than the physical maximum. This gap exists because the operational search applied `controller_max_com_z_m = 0.43 m` as an artificial ceiling, and dynamic validation further constrained the validated range.

## Static revalidation results

Both physical extrema passed static revalidation:

### Minimum height revalidation
- Setup loaded from: `physical_min_height_setup.json`
- Recomputed CoM z: 0.291919 m (exact match)
- Static feasible: **true**
- Rejection reasons: none

### Maximum height revalidation
- Setup loaded from: `physical_max_height_setup.json`
- Recomputed CoM z: 0.490812 m (exact match)
- Static feasible: **true**
- Rejection reasons: none

The revalidation round-trip confirmed that:
1. Joint positions can be saved and reloaded accurately
2. Root z calibration is reproducible
3. Contact and CoM recomputation yields identical results
4. Static feasibility evaluation is deterministic

## Search methodology

### Coarse grid search
- Target heights: 61 targets from 0.254 m to 0.554 m (0.005 m steps)
- Posture search: 17×17 grid of symmetric hip-pitch/knee pairs per target
- Root z: Calibrated from wheel-floor geometry after setting posture
- Contact validation: MuJoCo forward kinematics with contact detection
- Feasibility: Static balance and joint limit checks only

### No controller constraints
The physical search intentionally removed:
- `controller_min_com_z_m = 0.38 m` floor
- `controller_max_com_z_m = 0.43 m` ceiling
- Dynamic hold requirements
- Dynamic recovery requirements
- Controller-specific tuning parameters

### No dynamic validation
The physical search did NOT run:
- Step E position hold
- Step C height recovery
- Push disturbance tests
- Friction/mass perturbations

Dynamic validation was deliberately excluded because the physical envelope quantifies kinematic workspace, not controller capability.

## Comparison to operational envelope

| Metric | Physical | Operational | Delta |
|---|---:|---:|---:|
| Min height (m) | 0.291919 | 0.393287 | -0.101368 |
| Max height (m) | 0.490812 | 0.412813 | +0.077999 |
| Envelope span (m) | 0.198893 | 0.019526 | +0.179367 |

**Key findings:**

1. **Physical envelope is 10.2× broader** than operational envelope (19.9 cm vs 2.0 cm span)

2. **Low-height gap is larger** than high-height gap (10.1 cm vs 7.8 cm), suggesting the controller may be more limited at low heights

3. **Controller constraints dominated operational search:** The operational min/max were primarily set by `controller_min_com_z_m` and `controller_max_com_z_m`, not by kinematic limits

4. **Significant improvement potential:** The physical envelope reveals ~18 cm of kinematic range that is currently unused by the controller

## Implications for controller development

### Low-height extension potential
The 10.1 cm gap below operational min suggests:
- Lower squatting poses are kinematically achievable
- Stand-up recovery from lower heights may be possible
- Obstacle avoidance or ducking maneuvers could extend lower

**Controller challenges for low heights:**
- Higher joint torques required to support weight
- Reduced stability margin (shorter support width: 0.347 m vs 0.297 m nominal)
- Greater pitch/roll sensitivity
- Wheel velocity authority may be insufficient

### High-height extension potential
The 7.8 cm gap above operational max suggests:
- Taller standing poses are kinematically achievable
- Reaching or manipulation tasks could benefit from extra height
- Better visibility/sensing from elevated posture

**Controller challenges for high heights:**
- Higher center of gravity increases tip-over risk
- Reduced support width (0.297 m vs 0.347 m in crouch)
- Greater sensitivity to lateral disturbances
- Potential wheel-floor contact quality degradation

### Recommended next steps
1. **Incremental expansion:** Test Step E/Step C at physical_min + 1cm and physical_max - 1cm to probe dynamic boundaries
2. **Controller tuning:** Investigate whether gain scheduling, WBC, or alternative control architectures can stabilize broader height range
3. **Hardware validation:** Sim-to-real testing to confirm physical envelope translates to hardware
4. **Failure analysis:** Diagnose why operational extrema fail dynamically to guide controller improvements

## Artifacts generated

All required artifacts were generated in `outputs/physical_standing_height_envelope_search/`:

- **physical_height_search_grid.csv** — All 61 evaluated candidates with full metrics (38 KB)
- **physical_height_valid_candidates.json** — All 61 valid candidates (70 KB)
- **physical_height_invalid_candidates.json** — Empty array (0 invalid candidates)
- **physical_height_envelope_summary.json** — Extrema and summary statistics (2.5 KB)
- **physical_height_envelope_report.md** — Human-readable summary
- **physical_min_height_setup.json** — Reproducible setup for minimum height
- **physical_max_height_setup.json** — Reproducible setup for maximum height
- **static_physical_extrema_validation.json** — Revalidation results

## Limitations and scope

This validation:
- ✓ Quantifies kinematic standing height range
- ✓ Uses symmetric hip-pitch/knee posture search
- ✓ Calibrates root_z from wheel-floor geometry
- ✓ Validates static equilibrium and contact
- ✓ Provides reproducible setup files

This validation does NOT:
- ✗ Claim dynamic stability at physical extrema
- ✗ Validate controller performance beyond operational range
- ✗ Run Step E or Step C at physical extrema
- ✗ Incorporate actuator torque/power limits
- ✗ Test on hardware
- ✗ Claim absolute mechanical min/max from joint range analysis
- ✗ Include asymmetric postures or non-standing configurations

## Relationship to operational envelope validation

The operational envelope validation (see `operational_height_extreme_validation.md`) applied additional constraints:
- Controller min/max CoM bounds
- Dynamic position hold success (Step E)
- Dynamic height recovery success (Step C)
- Conservative selection with joint-margin buffer
- Candidate D2 profile tuning

The operational envelope represents **controller-ready validated extrema** within the broader physical envelope.

## Final decision

**PHYSICAL_ENVELOPE_VALIDATION_PASS**

The physical standing height envelope has been quantified as:
- Physical minimum: **0.291919 m** CoM
- Physical maximum: **0.490812 m** CoM

This envelope is:
- Statically validated
- Significantly broader than operational envelope
- Independent of controller constraints
- Based on kinematic and contact feasibility only

Dynamic failure at these extrema would indicate controller limitation, not kinematic impossibility, and should motivate controller improvement rather than envelope reduction.
