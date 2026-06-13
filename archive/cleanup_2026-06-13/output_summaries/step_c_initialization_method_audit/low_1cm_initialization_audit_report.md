# Step C low_1cm Initialization Method Audit

## Verdict

- Classification: **physically_inconsistent_height_perturbation**
- Root-z-only perturbation confirmed: `true`
- Root-z-only perturbation is suspected invalid as the official Step C height-change method: `true`
- Controller behavior changed: `false`
- WBC applied: `false`
- Step E invariants preserved: `true`

## Key evidence

- `scripts/simulate_hierarchical_controller.py` applies `--initial-root-z-perturbation` by changing `data.qpos[2]` only, zeroing velocities/accelerations, and calling `mj_forward`.
- The perturbation is applied after equilibrium capture, so controller references remain the nominal Step E references rather than a height-variant equilibrium.
- low_1cm invalid contact rows are startup rows only: steps `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]` at times `[0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]`.
- low_1cm raw/adjusted contact validity: `98.0%` / `98.0%`.
- low_1cm wheel velocity max abs: `21.34300994873047` rad/s, consistent with startup settling after the root-z-only displacement.
- First 20 low_1cm rows keep geometric wheel contacts true: wheel contacts `False`, wheel-floor contacts `False`, double-contact supervisor `False`, non-wheel floor contacts zero `True`.
- First 20 low_1cm body/support metrics are safe: pitch safe `True`, roll safe `True`, support-position safe `True`.
- Nominal only had the single startup contact-force artifact already handled by the validator; low_1cm has ten consecutive startup invalid rows.

## Step B reuse evidence

- Existing Step B setup report: `outputs\balance_core_true_height_variants\true_height_variant_setup_report.json`
- Search method: `multiobjective_com_calibrated`
- Ready for B5-B10: `True`
- Valid variants: `nominal, high_tiny, high_small, low_tiny, low_small`
- Step B variants are true posture variants: non-nominal variants change hip_pitch/knee and use calibrated root_z; they are explicitly not root-z-only offsets.

| Variant | Target CoM Z | Achieved CoM Z | Root Z | Hip pitch | Knee | Valid |
|---|---:|---:|---:|---:|---:|---|
| nominal | 0.403999 | 0.403999 | 0.535742 | 0.926052 | 1.748364 | True |
| high_tiny | 0.408999 | 0.409381 | 0.542188 | 0.915526 | 1.716785 | True |
| high_small | 0.413999 | 0.412813 | 0.546680 | 0.894473 | 1.695732 | True |
| low_tiny | 0.398999 | 0.398632 | 0.529297 | 0.936578 | 1.779943 | True |
| low_small | 0.393999 | 0.395234 | 0.524805 | 0.957631 | 1.800996 | True |

## Recommended Step C initialization design

1. Do not use root-z-only perturbation as the official Step C height-change method unless a later static validation proves it physically valid.
2. Prefer reusing existing Step B true height-variant setup JSONs for available ?5 mm / ?10 mm target heights.
3. For any missing target height, add a diagnostic-only symmetric hip_pitch/knee CoM-calibrated pose generator, then calibrate root_z for wheel-floor contact before simulation begins.
4. Before dynamic rollout, require static validation: both wheel contacts valid, no non-wheel ground contact, positive/reasonable contact force, pitch/roll/yaw near equilibrium, CoM height near target, support center near body projection, joint limits respected, and left/right symmetry preserved.
5. Capture equilibrium references from the valid height-variant pose before running the dynamic Step C case.

## Artifacts

- `low_1cm_initialization_audit.json`
- `low_1cm_initialization_audit_report.md`
- `low_1cm_startup_window.csv`
- `nominal_vs_low_1cm_startup_comparison.csv`
