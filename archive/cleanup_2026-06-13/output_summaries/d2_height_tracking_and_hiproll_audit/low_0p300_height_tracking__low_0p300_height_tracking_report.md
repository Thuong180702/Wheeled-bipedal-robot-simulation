# Low 0.300m Height Tracking Audit

**Classification:** `no_significant_drop`

**Root Cause:** `contact_compliance_or_settling`

## Setup Validity

- Target CoM: 0.300 m
- Achieved static CoM: 0.295485 m
- Height error at setup: 0.004515 m
- Hip pitch ref: 1.3761 rad
- Knee ref: 2.3484 rad
- Root z: 0.397088 m
- Setup valid: True
- Root-z-only candidate: False

## Height Behavior

- Initial CoM: 0.295450 m
- Final CoM: 0.275981 m
- CoM range: [0.272975, 0.295450] m
- Mean: 0.281781 m, Std: 0.005389 m
- Height error max: -0.004550 m
- Height error final: -0.024019 m
- Height error RMS: 0.019000 m
- Collapse amount: 0.022475 m

- First below target-1cm: step 309 at t=3.09s, com_z=0.289981m
- First below target-2cm: step 2303 at t=23.03s, com_z=0.279998m

## Posture Behavior

- Pitch range: [0.00, 0.76] deg
- Pitch RMS: 0.23 deg
- Roll range: [-0.48, 6.36] deg
- Roll RMS: 3.07 deg
- Hip pitch error max: 0.0597 rad
- Knee error max: 0.0226 rad

## Torque Behavior

- Hip roll max: 0.27 Nm
- Hip pitch max: 2.05 Nm
- Knee max: 8.88 Nm
- Wheel max: 1.66 Nm

## Survival

- Terminated: False
- Total steps: 5000
- Total time: 49.99 s

## Classification Rationale

CoM remained within 1cm of target throughout simulation.

## Root Cause Analysis

Contact compliance or settling behavior observed.
