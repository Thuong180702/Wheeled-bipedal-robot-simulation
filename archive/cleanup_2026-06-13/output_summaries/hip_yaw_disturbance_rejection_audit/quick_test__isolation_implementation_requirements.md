# Hip-Yaw Isolation Experiments - Implementation Requirements

## Current Blocker

The simulation script (`scripts/simulate_hierarchical_controller.py`) does not
support runtime override of shape posture controller gains (kp_hip_yaw, kd_hip_yaw).

## Required CLI Arguments

Add to `simulate_hierarchical_controller.py` argument parser:

```python
parser.add_argument(
    '--shape-kp-hip-yaw',
    type=float,
    default=None,
    help='Override shape posture controller kp_hip_yaw (default: 15.0 for balance-core)'
)
parser.add_argument(
    '--shape-kd-hip-yaw',
    type=float,
    default=None,
    help='Override shape posture controller kd_hip_yaw (default: 3.0 for balance-core)'
)
```

## Controller Instantiation Modification

In the section where `ShapePostureController` is instantiated:

```python
# Current (around line 1400-1500)
shape_posture_controller = ShapePostureController(
    kp_hip_yaw=BALANCE_CORE_HIP_YAW_AUTHORITY.kp_hip_yaw,  # 15.0
    kd_hip_yaw=BALANCE_CORE_HIP_YAW_AUTHORITY.kd_hip_yaw,  # 3.0
    # ...
)

# Modified
shape_kp_hip_yaw = (
    args.shape_kp_hip_yaw
    if args.shape_kp_hip_yaw is not None
    else BALANCE_CORE_HIP_YAW_AUTHORITY.kp_hip_yaw
)
shape_kd_hip_yaw = (
    args.shape_kd_hip_yaw
    if args.shape_kd_hip_yaw is not None
    else BALANCE_CORE_HIP_YAW_AUTHORITY.kd_hip_yaw
)

shape_posture_controller = ShapePostureController(
    kp_hip_yaw=shape_kp_hip_yaw,
    kd_hip_yaw=shape_kd_hip_yaw,
    # ...
)
```

## Telemetry Addition

Add effective kp/kd values to telemetry for verification:

```python
telemetry['effective_shape_kp_hip_yaw'].append(shape_kp_hip_yaw)
telemetry['effective_shape_kd_hip_yaw'].append(shape_kd_hip_yaw)
```

## Planned Experiments After Implementation

### Experiment D: Damping Sweep

Test kd values: [3, 5, 7, 9, 12]
Keep kp=15.0 fixed
Run at: low_0p300, high_0p480, nominal

```bash
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --height-variant-setup outputs/physical_target_height_setups/low_0p300_setup.json \
  --shape-kd-hip-yaw 5.0 \
  --num-steps 1000
```

### Experiment E: kp/kd Matrix

Test matrix (low_0p300 only):
- kp: [15, 20, 25]
- kd: [3, 5, 7, 9]
- Total: 12 combinations

```bash
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --height-variant-setup outputs/physical_target_height_setups/low_0p300_setup.json \
  --shape-kp-hip-yaw 20.0 \
  --shape-kd-hip-yaw 7.0 \
  --num-steps 1000
```

## Expected Outcomes

### If damping helps:
- Hip-yaw error decreases monotonically with kd
- Some kd value achieves hip_yaw_abs_max <= 0.07 rad
- Support drift does not worsen significantly
- Conclusion: Implement continuous kd schedule (HY-D candidate)

### If damping insufficient:
- Hip-yaw error plateaus or only reduces marginally
- No kd value passes hip-yaw gate
- Conclusion: Need kp increase (HY-PD) or feedforward (HY-FF)

### If increased authority helps:
- Higher kp reduces hip-yaw peak error
- Some (kp, kd) combination passes hip-yaw gate
- Conclusion: Implement continuous kp+kd schedule (HY-PD candidate)

### If neither helps:
- No parameter combination passes hip-yaw gate
- Disturbance magnitude exceeds available authority
- Conclusion: Need feedforward compensation (HY-FF) or sagittal fix first

## Alternative: Boundary Profile Approach

If CLI arguments are undesirable, create test profiles:

```python
# In simulate_hierarchical_controller.py
BOUNDARY_HIP_YAW_TEST_PROFILES = {
    'test_kd5': (15.0, 5.0),
    'test_kd7': (15.0, 7.0),
    'test_kd9': (15.0, 9.0),
    'test_kp20_kd7': (20.0, 7.0),
    # ...
}

parser.add_argument(
    '--boundary-hip-yaw-test-profile',
    type=str,
    default=None,
    choices=list(BOUNDARY_HIP_YAW_TEST_PROFILES.keys()),
    help='Test profile for hip-yaw isolation experiments'
)
```

Then select (kp, kd) based on profile if variant is low_0p300 or high_0p480.

## Recommendation

**Implement CLI arguments (--shape-kp-hip-yaw, --shape-kd-hip-yaw)**

Pros:
- Clean, flexible
- No profile proliferation
- Easy to script sweeps
- Self-documenting in telemetry

Cons:
- Requires simulation script modification

This is the cleanest path forward for Phase 2 isolation experiments.