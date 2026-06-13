# Balance-Core True Height Variant Setup Report (Multi-Objective Search)

## Search Method

**Multi-objective CoM-calibrated posture search** with scoring weights:
- Height error: 100.0
- CoM Y centering (sagittal): 50.0
- CoM X centering (lateral): 20.0
- Pitch error: 10.0
- Roll error: 10.0
- Yaw error: 5.0
- Joint distance from nominal: 1.0

## HeightIK Metric Audit

**CRITICAL**: HeightIK uses torso/root height (qpos[2]), NOT CoM height.
Cannot be used directly with target_com_z. Use multi-objective CoM-calibrated search instead.

## Gate Enforcement Status

- **CoM centering gate**: ✓ ENFORCED
- **Static-balance gate**: ✓ ENFORCED
- **Nominal reference comparison**: ✓ ENFORCED

## Summary

- **Nominal CoM**: 0.403999 m
- **Nominal support center**: (-0.000001, -0.013404) m
- **Nominal CoM support error**: (0.000001, -0.000132) m
- **Valid variants**: 5/5 (nominal, high_tiny, high_small, low_tiny, low_small)
- **Invalid variants**: 0/5 (none)
- **Ready for B5-B10**: yes

## Variant Details

### nominal (target=0.403999m)

**Status**: ✓ VALID

**Height**:
- Target CoM Z: 0.403999 m
- Achieved CoM Z: 0.403999 m
- Height error: 0.000000 m
- Calibrated root Z: 0.535742 m

**Posture**:
- Method: keyframe_baseline
- Hip pitch: 0.9261 rad (53.1°)
- Knee: 1.7484 rad (100.2°)
- Hip roll (L/R): 0.0000 / 0.0000 rad
- Hip yaw (L/R): 0.0000 / 0.0000 rad

**CoM Centering**:
- Support center: (-0.000001, -0.013404) m
- CoM position: (-0.000000, -0.013535, 0.403999) m
- CoM support error: (0.000001, -0.000132) m
- CoM support error norm: 0.000132 m

**Orientation**:
- Pitch X: 0.0000 rad (0.0°)
- Roll Y: 0.0000 rad (0.0°)
- Yaw Z: 0.0000 rad (0.0°)

**Contact**:
- Wheel floor contacts: 2
- Left wheel contact: True
- Right wheel contact: True
- Non-wheel floor contacts: 0

**Equilibrium References** (captured):
- Joint pos: ['0.0000', '0.0000', '0.9261', '1.7484', '0.0000', '0.0000', '0.0000', '0.9261', '1.7484', '0.0000']
- CoM pos: [-0.000000, -0.013535, 0.403999] m
- Pitch X: 0.0000 rad
- Roll Y: 0.0000 rad
- Yaw Z: 0.0000 rad

### high_tiny (target=0.408999m)

**Status**: ✓ VALID

**Height**:
- Target CoM Z: 0.408999 m
- Achieved CoM Z: 0.409381 m
- Height error: 0.000382 m
- Calibrated root Z: 0.542188 m

**Posture**:
- Method: multiobjective_com_calibrated_search
- Hip pitch: 0.9155 rad (52.5°)
- Knee: 1.7168 rad (98.4°)
- Hip roll (L/R): 0.0000 / 0.0000 rad
- Hip yaw (L/R): 0.0000 / 0.0000 rad

**CoM Centering**:
- Support center: (-0.000001, -0.015804) m
- CoM position: (-0.000000, -0.013627, 0.409381) m
- CoM support error: (0.000001, 0.002177) m
- CoM support error norm: 0.002177 m

**Orientation**:
- Pitch X: 0.0000 rad (0.0°)
- Roll Y: 0.0000 rad (0.0°)
- Yaw Z: 0.0000 rad (0.0°)

**Contact**:
- Wheel floor contacts: 2
- Left wheel contact: True
- Right wheel contact: True
- Non-wheel floor contacts: 0

**Candidate Search Statistics**:
- Total evaluated: 400
- Passed contact gate: 400
- Passed height gate: 57
- Passed CoM centering gate: 102
- Passed orientation gate: 400
- Passed all gates: 15

- Best by height: hip_pitch=1.0839, knee=1.7168, error=0.000026m
- Best by CoM centering: hip_pitch=0.9576, knee=1.8010, error=0.000031m

**Equilibrium References** (captured):
- Joint pos: ['0.0000', '0.0000', '0.9155', '1.7168', '0.0000', '0.0000', '0.0000', '0.9155', '1.7168', '0.0000']
- CoM pos: [-0.000000, -0.013627, 0.409381] m
- Pitch X: 0.0000 rad
- Roll Y: 0.0000 rad
- Yaw Z: 0.0000 rad

### high_small (target=0.413999m)

**Status**: ✓ VALID

**Height**:
- Target CoM Z: 0.413999 m
- Achieved CoM Z: 0.412813 m
- Height error: 0.001186 m
- Calibrated root Z: 0.546680 m

**Posture**:
- Method: multiobjective_com_calibrated_search
- Hip pitch: 0.8945 rad (51.3°)
- Knee: 1.6957 rad (97.2°)
- Hip roll (L/R): 0.0000 / 0.0000 rad
- Hip yaw (L/R): 0.0000 / 0.0000 rad

**CoM Centering**:
- Support center: (-0.000001, -0.012423) m
- CoM position: (-0.000000, -0.012794, 0.412813) m
- CoM support error: (0.000001, -0.000371) m
- CoM support error norm: 0.000371 m

**Orientation**:
- Pitch X: 0.0000 rad (0.0°)
- Roll Y: 0.0000 rad (0.0°)
- Yaw Z: 0.0000 rad (0.0°)

**Contact**:
- Wheel floor contacts: 2
- Left wheel contact: True
- Right wheel contact: True
- Non-wheel floor contacts: 0

**Candidate Search Statistics**:
- Total evaluated: 400
- Passed contact gate: 400
- Passed height gate: 57
- Passed CoM centering gate: 102
- Passed orientation gate: 400
- Passed all gates: 14

- Best by height: hip_pitch=1.1050, knee=1.6747, error=0.000019m
- Best by CoM centering: hip_pitch=0.9576, knee=1.8010, error=0.000031m

**Equilibrium References** (captured):
- Joint pos: ['0.0000', '0.0000', '0.8945', '1.6957', '0.0000', '0.0000', '0.0000', '0.8945', '1.6957', '0.0000']
- CoM pos: [-0.000000, -0.012794, 0.412813] m
- Pitch X: 0.0000 rad
- Roll Y: 0.0000 rad
- Yaw Z: 0.0000 rad

### low_tiny (target=0.398999m)

**Status**: ✓ VALID

**Height**:
- Target CoM Z: 0.398999 m
- Achieved CoM Z: 0.398632 m
- Height error: 0.000367 m
- Calibrated root Z: 0.529297 m

**Posture**:
- Method: multiobjective_com_calibrated_search
- Hip pitch: 0.9366 rad (53.7°)
- Knee: 1.7799 rad (102.0°)
- Hip roll (L/R): 0.0000 / 0.0000 rad
- Hip yaw (L/R): 0.0000 / 0.0000 rad

**CoM Centering**:
- Support center: (-0.000001, -0.011071) m
- CoM position: (-0.000000, -0.013450, 0.398632) m
- CoM support error: (0.000001, -0.002379) m
- CoM support error norm: 0.002379 m

**Orientation**:
- Pitch X: 0.0000 rad (0.0°)
- Roll Y: 0.0000 rad (0.0°)
- Yaw Z: 0.0000 rad (0.0°)

**Contact**:
- Wheel floor contacts: 2
- Left wheel contact: True
- Right wheel contact: True
- Non-wheel floor contacts: 0

**Candidate Search Statistics**:
- Total evaluated: 400
- Passed contact gate: 400
- Passed height gate: 53
- Passed CoM centering gate: 102
- Passed orientation gate: 400
- Passed all gates: 14

- Best by height: hip_pitch=0.9576, knee=1.7799, error=0.000005m
- Best by CoM centering: hip_pitch=0.9576, knee=1.8010, error=0.000031m

**Equilibrium References** (captured):
- Joint pos: ['0.0000', '0.0000', '0.9366', '1.7799', '0.0000', '0.0000', '0.0000', '0.9366', '1.7799', '0.0000']
- CoM pos: [-0.000000, -0.013450, 0.398632] m
- Pitch X: 0.0000 rad
- Roll Y: 0.0000 rad
- Yaw Z: 0.0000 rad

### low_small (target=0.393999m)

**Status**: ✓ VALID

**Height**:
- Target CoM Z: 0.393999 m
- Achieved CoM Z: 0.395234 m
- Height error: 0.001235 m
- Calibrated root Z: 0.524805 m

**Posture**:
- Method: multiobjective_com_calibrated_search
- Hip pitch: 0.9576 rad (54.9°)
- Knee: 1.8010 rad (103.2°)
- Hip roll (L/R): 0.0000 / 0.0000 rad
- Hip yaw (L/R): 0.0000 / 0.0000 rad

**CoM Centering**:
- Support center: (-0.000001, -0.014268) m
- CoM position: (-0.000000, -0.014237, 0.395234) m
- CoM support error: (0.000001, 0.000031) m
- CoM support error norm: 0.000031 m

**Orientation**:
- Pitch X: 0.0000 rad (0.0°)
- Roll Y: 0.0000 rad (0.0°)
- Yaw Z: 0.0000 rad (0.0°)

**Contact**:
- Wheel floor contacts: 2
- Left wheel contact: True
- Right wheel contact: True
- Non-wheel floor contacts: 0

**Candidate Search Statistics**:
- Total evaluated: 400
- Passed contact gate: 400
- Passed height gate: 49
- Passed CoM centering gate: 102
- Passed orientation gate: 400
- Passed all gates: 15

- Best by height: hip_pitch=0.9155, knee=1.8010, error=0.000029m
- Best by CoM centering: hip_pitch=0.9576, knee=1.8010, error=0.000031m

**Equilibrium References** (captured):
- Joint pos: ['0.0000', '0.0000', '0.9576', '1.8010', '0.0000', '0.0000', '0.0000', '0.9576', '1.8010', '0.0000']
- CoM pos: [-0.000000, -0.014237, 0.395234] m
- Pitch X: 0.0000 rad
- Roll Y: 0.0000 rad
- Yaw Z: 0.0000 rad
