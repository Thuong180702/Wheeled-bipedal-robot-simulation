# K2 JAX Release Hardening — Baseline Lock

**Date:** 2026-06-28
**Phase:** 0 — Reproducibility and Baseline Lock
**Purpose:** Record exact repository state before release hardening begins

---

## Git Baseline

| Field | Value |
|-------|-------|
| **Branch** | `repo-cleanup-t6j` |
| **Commit** | `0e1c7135e22b4cb852f71a795426cd3d3f19753a` |
| **Commit message** | `Stage 6K: Dynamic runner extended, JAX ramp_up terminates at step 556/5000` |
| **Parent commit** | `4c19803 Stage 6I: Fixed-height 25/25 PASS, dynamic height needs transition fix` |

### Recent commits (last 5)

```
0e1c713 Stage 6K: Dynamic runner extended, JAX ramp_up terminates at step 556/5000
4c19803 Stage 6I: Fixed-height 25/25 PASS, dynamic height needs transition fix
9e81267 Stage 6H: Full validation gate — 25/25 PASS, 151 tests
b338e09 Stage 6G-B: Support feedforward fix — ALL 17/17 PASS, 151 tests
f7d8d71 Stage 6G: Height mismatch bug found and fixed
```

### Working tree diff summary

```
 docs/validation/k2_post_promotion_long_run_and_dynamic_height_regression_report.md |   61 +-
 scripts/simulate_hierarchical_controller.py                                        |  704 ++++++++++-
 scripts/validate_k2_dynamic_height_gate_crossing.py                                |    7 +-
 scripts/validate_k2_post_promotion_long_run.py                                     |   30 +-
 tests/test_k2_jax_backend_cli.py                                                   |   97 ++
 tests/test_k2_jax_component_parity.py                                              |    3 +-
 tests/test_k2_jax_step_parity.py                                                   |   29 +-
 wheeled_biped/controllers/k2_jax_controller.py                                     | 1228 ++++++++++++++++++--
 wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py           |   29 +
 wheeled_biped/controllers/signal_filters.py                                        |   19 +-
 10 files changed, 2008 insertions(+), 199 deletions(-)
```

### Changed files (unstaged)

1. `docs/validation/k2_post_promotion_long_run_and_dynamic_height_regression_report.md`
2. `scripts/simulate_hierarchical_controller.py`
3. `scripts/validate_k2_dynamic_height_gate_crossing.py`
4. `scripts/validate_k2_post_promotion_long_run.py`
5. `tests/test_k2_jax_backend_cli.py`
6. `tests/test_k2_jax_component_parity.py`
7. `tests/test_k2_jax_step_parity.py`
8. `wheeled_biped/controllers/k2_jax_controller.py`
9. `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py`
10. `wheeled_biped/controllers/signal_filters.py`

---

## Backend Status

| Field | Value |
|-------|-------|
| **Python backend** | **Default** (all scripts default to `--controller-backend python`) |
| **JAX backend** | **Opt-in** (via `--controller-backend jax`) |
| **Backend CLI flag** | `--controller-backend {python, jax}` |

---

## K2 Profile

| Field | Value |
|-------|-------|
| **Profile name** | `k2_notch_low_q_v1` |
| **Python constant** | `K2_NOTCH_LOW_Q_V1` (from `wheeled_biped.controllers.sagittal_velocity_damped_balance_controller`) |
| **Profile type** | `SagittalVelocityDampedBalanceController` authority schedule |
| **Controller mode** | `balance-core` |
| **Sagittal controller** | `velocity-damped` |

---

## Controller Configuration (Key Flags)

| Mechanism | Status |
|-----------|--------|
| `enable_unified_sagittal_state_feedback` | **False** |
| `enable_wip_notch_filter` | **True** |
| `outer_loop_enabled` | **True** |
| `calibrated_outer_loop_enabled` | **True** (v2) |
| `low_band_support_outer_loop_enabled` | **True** |
| `physics_equilibrium_feedforward_enabled` | **True** (v1.0) |
| `recenter_priority_enabled` | **True** |
| `recenter_priority_direct_enabled` | **True** |
| `vd_wheel_damping_recenter_override_enabled` | **True** |
| `position_cap_recenter_boost_enabled` | **True** |
| `apcr1nd_tuned_enabled` | **True** (`adaptive_support_centering_trim`) |
| `arch_fix_enabled` | **True** (`budget_cap_raise`) |
| `t6i_enabled` | **True** |
| `adaptive_bias_trim_enabled` | **True** (`adaptive_bias_trim_replace_t6j: True`) |
| `WBC / hidden torque` | **Inactive / not applicable** (not a `dual_rate_balance_controller` profile) |
| `torque_wbc_enabled` | **False** (implied by controller architecture) |

---

## Test Inventory

| Metric | Value |
|--------|-------|
| **Total tests collected** | 131 |
| **Total tests passed** | **131** |
| **Tests failed** | 0 |
| **Tests skipped** | 0 |
| **Tests xfailed** | 0 |
| **Test runtime** | 498.78s (~8 min) |
| **Test files** | `test_k2_jax_backend_cli.py`, `test_k2_jax_component_parity.py`, `test_k2_jax_step_parity.py`, `test_k2_jax_branch_activity_audit.py` |

### Previous test count discrepancy explained

- Previous reports mentioned 125/125 (111 component + 14 backend CLI)
- Current count: 131/131 — the difference is `test_k2_jax_branch_activity_audit.py` tests were previously excluded or not counted
- 109/109 count from another report likely used `-m "not slow"` which deselects slow tests
- No silent test removal detected. Full suite with no markers = 131, all passing.

---

## Exact Command Set for Validations

### Both-synced parity (Phase 1)
```bash
# Primary: compare_k2_python_vs_jax_step.py (synthetic inputs, full controller pipeline)
python scripts/compare_k2_python_vs_jax_step.py --scenario <name> --steps 200 --output-dir outputs/k2_jax_parity

# Extended 9-scenario runner (to be extended for missing scenarios)
python scripts/compare_k2_python_vs_jax_step.py --all-scenarios --steps 200 --output-dir outputs/k2_jax_parity
```

### Functional validation (Phase 2)
```bash
python scripts/validate_k2_jax_stage6h_full.py

# Individual scenarios:
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --vd-sagittal-authority-profile k2_notch_low_q_v1 \
  --controller-backend jax \
  --height-variant-setup outputs/physical_target_height_setups_centered/<variant>_setup.json \
  --steps <steps> \
  --wbc-quiet
```

### Long-run validation (Phase 3)
```bash
python scripts/validate_k2_post_promotion_long_run.py \
  --profile k2 \
  --controller-backend jax \
  --mode equilibrium
# or with PRBS:
python scripts/validate_k2_post_promotion_long_run.py \
  --profile k2 \
  --controller-backend jax \
  --mode prbs
```

### Test suite (Phase 4)
```bash
pytest tests/test_k2_jax_*.py -v
```

### Performance sanity (Phase 6)
```bash
python -c "
import jax, timeit
from wheeled_biped.controllers.k2_jax_controller import *
# ... hot-step benchmarking
"
```

---

## Previous Parity Status

From `k2_jax_full_both_synced_parity_matrix.md` (2026-06-28):

| Scenario | Status |
|----------|--------|
| fixed_high_0p480 | PASS (<1e-7) |
| fixed_low_0p330 | PASS (prior gate) |
| ramp_up | DEGRADED (0.57 Nm diff) |
| ramp_down | Pending |
| gate_chatter | Pending |
| push_fwd_90N | IMPROVED (0.98 Nm, +67%) |
| push_bwd_90N | IMPROVED (1.2 Nm, +63%) |

**Prior classification:** `K2_JAX_FULL_BOTH_SYNCED_PARITY_FAIL_WITH_ROOT_CAUSE`

**Fixes applied since (per user):**
1. push_fwd_90N composer/rate-limit parity resolved
2. effective_max_position_tau runtime state added
3. ABS trim scheduling corrected to use actual measured com_z
4. ABS trim state/timing verified correct
5. two-stage tau_position clipping fixed
6. dynamic/fixed/push both-synced parity passes in the checked 7-scenario set
7. Tests pass: 131/131

---

## Classification

**K2_JAX_RELEASE_HARDENING_BASELINE_LOCK_RECORDED**

**Expected pre-hardening state confirmed:**
- K2 profile: `K2_NOTCH_LOW_Q_V1` / `k2_notch_low_q_v1` ✅
- Controller mode: `balance-core` ✅
- Sagittal controller: `velocity-damped` ✅
- WBC/hidden torque: disabled/inactive ✅
- Python default preserved ✅
- JAX opt-in preserved ✅
- 131/131 tests pass ✅
