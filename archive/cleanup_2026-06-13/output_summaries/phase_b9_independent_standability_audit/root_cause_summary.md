# Independent root-cause summary

Decision: **B. RESET_EQUILIBRIUM_BUG**

Key evidence:

1. The current audited reset/static state is physically invalid.
   - Full balanced-root h=0.60 t0 wheel clearances are `-0.021007` / `-0.020972` m with contact forces `3124.8` / `3118.8` N. This is ~21 mm wheel penetration and multi-kN contact impulse, not a static equilibrium for an ~8 kg robot.
   - Full balanced-root h=0.60 drifts to roll `-20.598` deg by 100 ms.
   - Step5 joint-only h=0.60 starts with no wheel contact (`force=nan`) because the evaluator applies leg joints but not the table root pose.
   - h=0.40 shows the same issue: full-root t0 forces `96.9` / `93.2` N and Step5 joint-only root starts with no contact.

2. The Step 5 best-controller evaluation path does not apply the balanced-root initialization it claims to use.
   - `scripts/phase_b9_step5_lqr_gain_strengthening.py::apply_balanced_root_init` writes only hip_pitch/knee (`qpos[9]`, `qpos[10]`, `qpos[14]`, `qpos[15]`).
   - It ignores `root_x`, `root_z`, `root_roll`, and `root_pitch` from `configs/controllers/b9_balanced_root_init_table.yaml`.
   - Earlier Step 3/4 scripts contain a full-root initializer, so this is a localized evaluation/control setup regression.

3. Model/control path basics are valid enough to transmit commands.
   - Model has 10 actuators and 10 controlled joints.
   - PID path sends normalized leg targets to position PID and wheel targets to velocity PI.
   - Control trace at h=0.60/full-root: hip-roll action is `0.000/0.000` and PID torque is `0.000/0.000` Nm because the controller requests zero hip-roll correction.

4. The active B9 controller also lacks a lateral closed loop, but this is secondary to the reset bug for the final decision.
   - `configs/controllers/dual_rate_balance_controller_b9.yaml` sets roll kp/kd/max_correction to zero.
   - `DualRateBalanceController.compute_action()` only activates hip-roll correction when roll gains are nonzero; otherwise both hip-roll actions are zero.
   - The wheel LQR is symmetric left/right, so it controls pitch/forward dynamics, not roll/lateral dynamics.
   - Minimal roll probes show lateral correction is not part of the current controller.
   - +2 deg perturbation, no correction: roll after 10 physics steps `-1.514` deg.
   - +2 deg perturbation, bounded hip-roll correction probe: roll after 10 physics steps `-1.511` deg.
   - This verifies hip-roll commands can reach actuators, but the current controller never commands them.

Hypothesis classification:

- H1 MODEL/MJCF bug: not primary from current evidence; axes/actuators/contact are plausible and commands produce actuator torques.
- H2 RESET bug: supported as primary; reset either starts in no-contact state or in excessive penetration/contact impulse state depending on initializer.
- H3 CONTROL PATH/PID bug: secondary localized bug in the Step 5 initializer/evaluation path, not the PID math itself.
- H4 CONTACT/FRICTION problem: not primary; bad contact measurements are caused by invalid initial geometry, not proven friction failure.
- H5 MISSING LATERAL BALANCE CONTROLLER: real missing controller term, but diagnose after reset is physically valid.
- H6 MISSING VMC/WHOLE-BODY LAYER: likely needed later for robust standing, not the first root cause.
- H7 ACTUATOR AUTHORITY LIMIT: not established; current reset invalidity prevents authority conclusions.
- H8 ARCHITECTURE LIMIT: premature; a concrete reset/static-equilibrium bug exists.
