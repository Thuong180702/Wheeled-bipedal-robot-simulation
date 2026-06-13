# Balance Core Cleanup — Verification

**Date:** 2026-06-13
**Branch:** `repo-cleanup-t6j`
**Phase:** 6

---

## 1. Compile checks

`simulate_hierarchical_controller.py`, `train.py`, `evaluate.py`,
`validate_checkpoint.py`, `sagittal_velocity_damped_balance_controller.py`,
`audit_outputs_bulk.py`, `clean_outputs_bulk.py`, `extract_output_summaries.py`,
`delete_balance_core_dirs.py` → **COMPILE OK**

---

## 2. Key tests

```
test_t6j_centering_bias_trim.py          → passed
test_simulation_telemetry_csv_writer.py → passed
test_low_height_setup_initialization.py  → passed
test_step_e_wbc_gate_validator.py         → passed (1 skipped)
```
**Result: 47 passed, 1 skipped.**

---

## 3. CLI help smoke

`train --help`, `evaluate --help`, `validate_checkpoint.py --help`,
`simulate_hierarchical_controller.py --help` → all OK.

---

## 4. T6J smoke simulation (100 steps, preserved setup JSON)

```
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --vd-sagittal-authority-profile T6J_centering_bias_trim \
  --height-variant-setup outputs/physical_target_height_setups/high_0p480_setup.json \
  --steps 100 --telemetry-decimation 1 --failure-window-steps 100 \
  --write-run-summary-sidecar
```

- **Status: [OK] Completed full simulation without falling** (100/100 steps)
- Terminated: False
- T6J path, setup JSON loader, and telemetry writer all intact after cleanup.

---

## Classification

**BALANCE_CORE_CLEANUP_PASS**