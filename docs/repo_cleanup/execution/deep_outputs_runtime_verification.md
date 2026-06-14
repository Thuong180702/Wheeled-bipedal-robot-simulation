# Deep Outputs Cleanup — Runtime Verification

**Date:** 2026-06-13
**Branch:** `repo-cleanup-t6j`
**Phase:** 6

---

## 1. Compile checks

```
python -m py_compile scripts/simulate_hierarchical_controller.py                              → OK
python -m py_compile scripts/train.py                                                         → OK
python -m py_compile scripts/evaluate.py                                                      → OK
python -m py_compile scripts/validate_checkpoint.py                                           → OK
python -m py_compile wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py → OK
```

**Result: COMPILE OK**

---

## 2. Key tests

```
tests/test_t6j_centering_bias_trim.py
tests/test_simulation_telemetry_csv_writer.py
tests/test_low_height_setup_initialization.py
tests/test_step_e_wbc_gate_validator.py
```

**Result: 47 passed, 1 skipped.** The T6J runtime path, telemetry CSV writer, height-variant setup loader, and Step E gate validator all pass after the bulk output deletion.

---

## 3. CLI help smoke

```
python scripts/train.py --help                          → OK
python scripts/evaluate.py --help                       → OK
python scripts/validate_checkpoint.py --help            → OK
python scripts/simulate_hierarchical_controller.py --help → OK
```

---

## 4. T6J smoke simulation (preserved setup JSON)

```
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core --sagittal-controller velocity-damped \
  --vd-sagittal-authority-profile T6J_centering_bias_trim \
  --height-variant-setup outputs/physical_target_height_setups/high_0p480_setup.json \
  --steps 100 --telemetry-decimation 1 --failure-window-steps 100 --write-run-summary-sidecar
```

Result:
- **Status: [OK] Completed full simulation without falling** (100/100 steps).
- CoM height 0.481–0.491 m; pitch_x −0.0…5.7°; roll_y −0.0…0.1°.
- Telemetry written to a freshly recreated `outputs/hierarchical_controller_sim/` (~1.1 M) — confirms the preserved setup JSON loads and the T6J path runs end-to-end after cleanup.

---

## 5. Protected assets re-confirmed post-smoke

All 3 seed final checkpoints, `physical_target_height_setups/`, and `backup_checkpoints/` present after the smoke run.

---

## Classification

**DEEP_OUTPUTS_RUNTIME_VERIFICATION_PASS**

Compile clean, key tests green, all CLI entrypoints load, T6J smoke runs cleanly off the preserved setup JSON, and every protected asset survived.
