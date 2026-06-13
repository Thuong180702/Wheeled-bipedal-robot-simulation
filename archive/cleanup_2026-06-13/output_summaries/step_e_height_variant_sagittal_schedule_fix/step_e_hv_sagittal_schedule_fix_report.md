# Step E-HV Sagittal Schedule Fix Report

- Selected candidate: `None`
- Final decision: **STEP_E_HEIGHT_VARIANT_ROBUSTNESS_GAP**
- Controller behavior changed: `true` only when selected candidate is non-baseline
- WBC remains disabled; hidden torque and ownership are checked per case.

| Candidate | Variant | Steps | Verdict | Support max abs | HipYaw max | Pitch max | Wheel max | Required fails |
|---|---|---:|:---:|---:|---:|---:|---:|---|
| candidate_A_position_cap | high_tiny | 1000 | PASS | 0.1386544651213209 | 0.0226624440401792 | 0.0941353530196226 | 4.354282140731812 |  |
| candidate_A_position_cap | high_tiny | 5000 | PASS | 0.1390517074299906 | 0.0226624440401792 | 0.0942438242959596 | 4.376364231109619 |  |
| candidate_A_position_cap | high_small | 1000 | FAIL | 0.1519078740134757 | 0.0226932317018508 | 0.0984017956931302 | 5.243177652359009 | support_max_abs=0.151908 > 0.15; wheel_vel_max=5.243178 > 5.0 |
| candidate_A2_height_staged | high_tiny | 1000 | PASS | 0.1386544651213209 | 0.0226624440401792 | 0.0941353530196226 | 4.354282140731812 |  |
| candidate_A2_height_staged | high_tiny | 5000 | PASS | 0.1390517074299906 | 0.0226624440401792 | 0.0942438242959596 | 4.376364231109619 |  |
| candidate_A2_height_staged | high_small | 1000 | FAIL | 0.1775534171871721 | 0.023465858772397 | 0.1126263796786364 | 6.772815465927124 | support_max_abs=0.177553 > 0.15; pitch_max=0.112626 > 0.1; wheel_vel_max=6.772815 > 5.0 |