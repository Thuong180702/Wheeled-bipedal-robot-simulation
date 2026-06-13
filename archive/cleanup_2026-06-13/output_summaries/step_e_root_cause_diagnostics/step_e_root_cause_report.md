# Step E Root Cause Diagnostics Report

## 1. Executive summary

- H1 sagittal sign/frame mismatch: **partially_confirmed**.
- H2 velocity-frame mismatch: **partially_confirmed** (not_observed).
- H3 hip-roll posture ownership: **rejected**.
- H4 hip-yaw posture: **confirmed**.

Exact next-step recommendation: **Fix sagittal axis/sign first**.

It is safe to proceed to a fix only for hypotheses marked confirmed or partially confirmed, preserving the safety constraints below.

## 2. Test environment

- Commit hash: `9971615447f36e0127482db28c5b4139b742bc3b`
- Date/time UTC: `2026-06-01T03:03:22.669585+00:00`
- Python version: `3.10.2 (tags/v3.10.2:a58ebcc, Jan 17 2022, 14:12:15) [MSC v.1929 64 bit (AMD64)]`
- MuJoCo version: `3.6.0`
- Platform: `Windows-10-10.0.26200-SP0`
- Command lines used: ['python scripts/diagnose_step_e_root_causes.py']
- Simulation steps: 1000-step current/flipped axis ablation; 5000-step gate result: {
  "current_5000": {
    "requested_steps": 5000,
    "survived_steps": 5000,
    "terminated": false,
    "termination_reason": "",
    "final_sim_time_s": 50.00000000001514,
    "support_position_error_m": {
      "min": -0.038304423693072395,
      "max": 0.5433812576321346,
      "final": -0.0059573465213673266,
      "rms": 0.12058955624157076,
      "max_abs": 0.5433812576321346
    },
    "sagittal_position_error_m": {
      "min": -0.038304423693072395,
      "max": 0.5433812576321346,
      "final": -0.0059573465213673266,
      "rms": 0.12058955624157076,
      "max_abs": 0.5433812576321346
    },
    "com_position_error_sagittal_m": {
      "min": -0.017446594312787056,
      "max": 0.5242135860025883,
      "final": 0.00566528644412756,
      "rms": 0.11808957260598976,
      "max_abs": 0.5242135860025883
    },
    "pitch_x_rad": {
      "min": -0.029177128786793814,
      "max": 0.1254399667535688,
      "final": -0.012886520122854742,
      "rms": 0.04278403412546174,
      "max_abs": 0.1254399667535688
    },
    "pitch_x_error_rad": {
      "min": -0.029177128786793814,
      "max": 0.1254399667535688,
      "final": -0.014071479958099509,
      "rms": 0.042783645982647375,
      "max_abs": 0.1254399667535688
    },
    "roll_y_rad": {
      "min": -0.04496472382335462,
      "max": 0.0065163428576250565,
      "final": 0.0009647156842792328,
      "rms": 0.004713211015129573,
      "max_abs": 0.04496472382335462
    },
    "yaw_z_rad": {
      "min": -0.20663638023742736,
      "max": 0.07060047910879022,
      "final": 0.06781696574066179,
      "rms": 0.057603107223895354,
      "max_abs": 0.20663638023742736
    },
    "com_z_m": {
      "min": 0.36227157711982727,
      "max": 0.4085761606693268,
      "final": 0.36718514561653137,
      "rms": 0.37980137180370765,
      "max_abs": 0.4085761606693268
    },
    "wheel_vel_mean_rad_s": {
      "min": -7.0356128215789795,
      "max": 3.3868978023529053,
      "final": -3.174791932106018,
      "rms": 2.007600205908222,
      "max_abs": 7.0356128215789795
    },
    "tau_position": {
      "min": -3.0,
      "max": 1.5321769714355469,
      "final": 0.23829385638237,
      "rms": 2.0145189828803884,
      "max_abs": 3.0
    },
    "tau_pitch": {
      "min": -1.4588564393396908,
      "max": 6.27199833767844,
      "final": -0.7035739979049754,
      "rms": 2.139182299132369,
      "max_abs": 6.27199833767844
    },
    "tau_sagittal_velocity": {
      "min": -5.258247256278992,
      "max": 2.597227692604065,
      "final": -1.8768563121557236,
      "rms": 1.3802222529649661,
      "max_abs": 5.258247256278992
    },
    "torque_saturation_fraction": {
      "min": 0.0,
      "max": 0.0,
      "final": 0.0,
      "rms": 0.0,
      "max_abs": 0.0
    },
    "torque_rate_saturation_fraction": {
      "min": 0.0,
      "max": 0.2,
      "final": 0.0,
      "rms": 0.004,
      "max_abs": 0.2
    },
    "ownership_violation_count_max": 0,
    "hidden_torque_norm_max": 0.0,
    "tau_wbc_norm_max": 0.0,
    "csv": "F:\\ROBOTCUATAO\\Wheeled-bipedal-robot-simulation\\outputs\\step_e_root_cause_diagnostics\\axis_ablation_current_5000.csv"
  },
  "flipped_5000": {
    "requested_steps": 5000,
    "survived_steps": 5000,
    "terminated": false,
    "termination_reason": "",
    "final_sim_time_s": 50.00000000001514,
    "support_position_error_m": {
      "min": -20.667120526506178,
      "max": 0.005792708048510134,
      "final": -20.667120526506178,
      "rms": 11.801773995199007,
      "max_abs": 20.667120526506178
    },
    "sagittal_position_error_m": {
      "min": -20.667120526506178,
      "max": 0.005792708048510134,
      "final": -20.667120526506178,
      "rms": 11.801773995199007,
      "max_abs": 20.667120526506178
    },
    "com_position_error_sagittal_m": {
      "min": -20.667057905346155,
      "max": 0.00032572541385889053,
      "final": -20.667057905346155,
      "rms": 11.801807197347784,
      "max_abs": 20.667057905346155
    },
    "pitch_x_rad": {
      "min": -0.0074536729231057695,
      "max": 0.03801245563131311,
      "final": 0.035559001321246855,
      "rms": 0.03491796798796323,
      "max_abs": 0.03801245563131311
    },
    "pitch_x_error_rad": {
      "min": -0.0074536729231057695,
      "max": 0.03801245563131311,
      "final": 0.03555935069129028,
      "rms": 0.034914346619884386,
      "max_abs": 0.03801245563131311
    },
    "roll_y_rad": {
      "min": 2.5593164454837064e-05,
      "max": 0.0030658096479634364,
      "final": 0.0016058887230649974,
      "rms": 0.0016572334713107883,
      "max_abs": 0.0030658096479634364
    },
    "yaw_z_rad": {
      "min": -0.08537137116168525,
      "max": 0.022362941505331305,
      "final": -0.08537137116168525,
      "rms": 0.04867803730746595,
      "max_abs": 0.08537137116168525
    },
    "com_z_m": {
      "min": 0.40383487939834595,
      "max": 0.4087391793727875,
      "final": 0.4082396626472473,
      "rms": 0.40828741245770866,
      "max_abs": 0.4087391793727875
    },
    "wheel_vel_mean_rad_s": {
      "min": -7.360750675201416,
      "max": 1.2347928285598755,
      "final": -7.08056902885437,
      "rms": 6.962929028169349,
      "max_abs": 7.360750675201416
    },
    "tau_position": {
      "min": -0.2317083179950714,
      "max": 3.0,
      "final": 3.0,
      "rms": 2.9711889347718854,
      "max_abs": 3.0
    },
    "tau_pitch": {
      "min": -0.3726836461552885,
      "max": 1.9006227815656556,
      "final": 1.7779675345645138,
      "rms": 1.745717330994219,
      "max_abs": 1.9006227815656556
    },
    "tau_sagittal_velocity": {
      "min": -6.542086601257324,
      "max": 0.10812515392899513,
      "final": -6.3457489013671875,
      "rms": 6.25235690303277,
      "max_abs": 6.542086601257324
    },
    "torque_saturation_fraction": {
      "min": 0.0,
      "max": 0.0,
      "final": 0.0,
      "rms": 0.0,
      "max_abs": 0.0
    },
    "torque_rate_saturation_fraction": {
      "min": 0.0,
      "max": 0.2,
      "final": 0.0,
      "rms": 0.004,
      "max_abs": 0.2
    },
    "ownership_violation_count_max": 0,
    "hidden_torque_norm_max": 0.0,
    "tau_wbc_norm_max": 0.0,
    "csv": "F:\\ROBOTCUATAO\\Wheeled-bipedal-robot-simulation\\outputs\\step_e_root_cause_diagnostics\\axis_ablation_flipped_5000.csv"
  }
}
- Controller flags: standalone balance-core velocity-damped diagnostic loop; WBC applied torque off; legacy torque paths off.
- WBC remained off: `True`
- Ownership violation count max: `0`

## 3. Hypothesis H1 report: sagittal sign/frame mismatch

XML convention states the robot front is `-Y`, while the current diagnostic axis at zero yaw is `+Y`.

Wheel torque sign audit:

- Positive wheel torque mean delta support Y: `-0.043570391` m
- Current-axis max abs drift: `0.112218396` m
- Flipped-axis max abs drift: `3.769110538` m
- Current-axis final drift: `0.034222158` m
- Flipped-axis final drift: `-3.769110538` m
- Improvement max abs: `-3258.728` %
- Improvement final: `-10913.655` %

Numerical conclusion: **partially_confirmed**.

## 4. Hypothesis H2 report: velocity-frame mismatch

Code-path inspected in the standalone diagnostic loop records the call-site value before calling the controller.

- Projected velocity vs actual passed max abs difference: `0.000000000` m/s
- Projected velocity vs actual passed RMS difference: `0.000000000` m/s
- Actual passed value source: raw `com_vel[1]` / raw `com_vy`
- Dominance classification: `not_observed`

Numerical conclusion: **partially_confirmed**.

## 5. Hypothesis H3 report: hip-roll posture ownership

- Max abs hip-roll error: `0.050877981` rad
- RMS hip-roll error: `0.019813321` rad
- Percent time abs hip-roll error > 0.10 rad: `0.000` %
- Percent time abs hip-roll error > 0.15 rad: `0.000` %
- Percent time abs hip-roll error > 0.10 rad while abs roll < 0.05 rad: `0.000` %
- Roll-to-stance torque ratio median: `0.903763210`
- Roll-to-stance torque ratio max: `9.025182649`
- Shape posture hip-roll torque is zero/inactive: `True`

Posture validity conclusion: **rejected**.

## 6. Hypothesis H4 report: hip-yaw differential diagnosis

- Max abs hip-yaw error: `0.109224766` rad
- RMS hip-yaw error: `0.041609009` rad
- Final yaw drift: `-0.005102121` rad
- Yaw range: `0.025606182` rad

Hip-yaw conclusion: **confirmed**.

## 7. Final decision matrix

| Hypothesis | Evidence | Key metrics | Verdict | Recommended next action |
|-----------|----------|-------------|---------|--------------------------|
| H1 sagittal sign/frame | Wheel pulse sign and current/flipped ablation | max drift 0.1122 vs 3.7691 m | partially_confirmed | Fix sagittal axis/sign first |
| H2 velocity frame | Call-site actual value vs projected velocity | max 0.0000 m/s, RMS 0.0000 m/s | partially_confirmed | Do not fix first |
| H3 hip-roll posture | Hip-roll errors, ownership, roll/stance torque | max error 0.0509 rad | rejected | Do not fix first |
| H4 hip-yaw posture | Hip-yaw error and yaw drift | max error 0.1092 rad | confirmed | Collect more telemetry before changing code |

## 8. Exact next-step recommendation

**Fix sagittal axis/sign first**

## 9. Safety constraints for the next fix

- WBC remains off.
- No blind gain tuning.
- No legacy torque path reintroduction.
- No controller ownership violation.
- Fix only the confirmed root cause.

## Missing artifacts

None
