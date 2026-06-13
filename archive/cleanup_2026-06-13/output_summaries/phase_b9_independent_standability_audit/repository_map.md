# Repository map for independent standability audit

- Robot MJCF/model: `assets/robot/wheeled_biped_real.xml`
- Active model loader: `wheeled_biped/utils/config.py::get_model_path`
- Joint/action order: `wheeled_biped/envs/base_env.py::JOINT_NAMES`, `wheeled_biped/controllers/action_codec.py`
- Low-level PID: `wheeled_biped/sim/low_level_control.py::pid_control`
- Balance environment reset/step/termination: `wheeled_biped/envs/base_env.py`, `wheeled_biped/envs/balance_env.py`
- Classical B9 controller: `wheeled_biped/controllers/dual_rate_balance_controller.py`
- Base B9 controller config: `configs/controllers/dual_rate_balance_controller_b9.yaml`
- Reported best gain multipliers: `outputs/phase_b9_lqr_gain_strengthening/best_lqr_config.yaml`
- Balanced-root initialization table: `configs/controllers/b9_balanced_root_init_table.yaml`
- Step 3 full-root initializer reference: `scripts/phase_b9_step3_fast_only.py`
- Step 5 joint-only initializer under audit: `scripts/phase_b9_step5_lqr_gain_strengthening.py`
