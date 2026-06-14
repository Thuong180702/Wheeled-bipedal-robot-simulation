# Step E-HV Sagittal Schedule Fix Report

- Selected candidate: `candidate_D2_wheel_velocity_damping_light`
- Final decision: **STEP_E_HEIGHT_VARIANT_HOLD_PASS**
- Controller behavior changed: `true` only when selected candidate is non-baseline
- WBC remains disabled; hidden torque and ownership are checked per case.

| Candidate | Variant | Steps | Verdict | Support max abs | HipYaw max | Pitch max | Wheel max | Required fails |
|---|---|---:|:---:|---:|---:|---:|---:|---|
| candidate_D1_support_velocity_light | high_small | 1000 | FAIL | 0.1500130875681833 | 0.02664091065526 | 0.0981123182582587 | 5.12576150894165 | support_max_abs=0.150013 > 0.15; wheel_vel_max=5.125762 > 5.0 |
| candidate_D2_wheel_velocity_damping_light | high_small | 1000 | PASS | 0.1296857785390591 | 0.0259768739342689 | 0.0941172833917718 | 4.566095590591431 |  |
| candidate_D2_wheel_velocity_damping_light | high_small | 5000 | PASS | 0.1354918956940817 | 0.0296155605465173 | 0.0960037677193006 | 4.770036935806274 |  |
| candidate_D2_wheel_velocity_damping_light | high_tiny | 5000 | PASS | 0.1241913834844777 | 0.0382067896425724 | 0.0917569964071727 | 4.118019104003906 |  |
| candidate_D2_wheel_velocity_damping_light | nominal | 5000 | PASS | 0.1060617172825227 | 0.0564832426607608 | 0.0711304926597348 | 3.867526888847351 |  |
| candidate_D2_wheel_velocity_damping_light | low_tiny | 5000 | PASS | 0.109590027603286 | 0.0420027114450931 | 0.0728116248189794 | 4.036306142807007 |  |
| candidate_D2_wheel_velocity_damping_light | low_small | 5000 | PASS | 0.1062003425457684 | 0.0573853142559528 | 0.0712435957923427 | 3.989667296409607 |  |