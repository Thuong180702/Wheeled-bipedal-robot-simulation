# K2 JAX Dedicated Realtime Runner — Blocked State Freeze

**Date:** 2026-06-29
**Phase:** 0 — FREEZE CURRENT BLOCKED STATE
**Classification:** `K2_JAX_DEDICATED_REALTIME_ORIGINAL_K2_PROMOTION_BLOCKED`

---

## 1. Repository State

| Field | Value |
|---|---|
| Commit | `0e1c7135e22b4cb852f71a795426cd3d3f19753a` |
| Short hash | `0e1c713` |
| Commit message | `Stage 6K: Dynamic runner extended, JAX ramp_up terminates at step 556/5000` |
| Branch | `repo-cleanup-t6j` |
| Previous commits | `4c19803` Stage 6I, `9e81267` Stage 6H, `b338e09` Stage 6G-B, `f7d8d71` Stage 6G |

---

## 2. Commands Used for 39/39 Measured Validation

```bash
# Full validation matrix (all 5 scopes, 39 scenarios)
python scripts/validate_k2_jax_dedicated_promotion.py \
  --scope all \
  --output-dir outputs/k2_jax_dedicated_promotion_validation

# Performance benchmarks
python scripts/run_k2_jax_realtime.py \
  --height-setup outputs/physical_target_height_setups/high_0p480_setup.json \
  --steps 3000 --quiet --telemetry off \
  --output-dir outputs/k2_jax_dedicated_promotion_validation/performance/fixed_high_0p480

# Tests
pytest tests/test_k2_strict_promotion_classifier.py \
       tests/test_k2_jax_dedicated_runner_guards.py -v
# 64/64 PASS

pytest tests/test_k2_jax_component_parity.py -v
# 71/72 PASS, 1 FAIL (params_size_consistent)
```

---

## 3. Output Directory

```
outputs/k2_jax_dedicated_promotion_validation/
├── step_c/          # 7 scenarios
├── step_e/          # 10 scenarios
├── step_d/          # 12 scenarios
├── dynamic_height/  # 5 scenarios
├── long_run/        # 5 scenarios
└── performance/     # 7 benchmark runs
```

Baseline reference: `outputs/k2_original_promoted_baseline/k2_original_metrics.json`

---

## 4. SAFETY_FAIL Cases (3 in dynamic_height)

| Scenario | Failure | Detail |
|---|---|---|
| `ramp_up_0p330_to_0p480` | FELL at step 1509 | height_too_low — CoM stuck at 0.33m, never rises to 0.48m |
| `up_down_cycle_0p330_0p480_0p330` | FELL at step 1186 | height_too_low — same root cause |
| `gate_dwell_0p420_0p450_0p480` | hip_yaw=0.537 rad | exceeds 0.35 rad absolute safety gate |

**Root cause of dynamic falls:** `original-k2-exact` mode uses STATIC q_ref from initial height setup. When starting from `low_0p330_setup.json`, the equilibrium posture is frozen at ~0.33m CoM. Height-dependent LQR gains alone cannot generate sufficient feedforward action to raise torso to 0.48m. CoM stays at 0.33-0.335m throughout ramp_up.

---

## 5. SAFE_BUT_WORSE Scopes (all non-dynamic scopes)

### Step C: 2/7 SAFE_BUT_WORSE
- C5_long_random: hy=0.1823 vs 0.0851, pitch=4.51 vs 3.63
- focused_low_0p320: hy=0.0821 vs 0.0502, pitch=3.69 vs 2.83

### Step E: 7/10 SAFE_BUT_WORSE
- low_0p300, low_0p320, low_0p340, low_0p360, low_0p380, high_0p430, high_0p450
- Systematic hip_yaw and pitch elevation vs original

### Step D: 8/12 SAFE_BUT_WORSE
- All mid_0p400 and low_0p330 conditions have elevated hip_yaw vs original (hy_max=0.000 in original baseline)
- ⚠️ Metric mismatch: candidate uses full-episode pitch_rms, original uses post-push 500-step

### Long-Run: 5/5 SAFE_BUT_WORSE
- All 5 heights show elevated hip_yaw vs original

---

## 6. Failed Test

```
tests/test_k2_jax_component_parity.py::TestParamsPackUnpackStage2::test_params_size_consistent FAILED

AssertionError: assert (54,) == (41,)
  K2_JAX_PARAMS_SIZE_STAGE2 = 41
  pack_params_stage2() returns shape (54,)

Location: k2_jax_controller.py
  Line 149: K2_JAX_PARAMS_SIZE_STAGE2 = len(K2_JAX_PARAMS_FIELDS_STAGE2)  # = 41
  Line 153: K2_JAX_PARAMS_SIZE_STAGE2_EXT_STANDALONE = K2_JAX_PARAMS_SIZE_STAGE2_EXT + 6  # = 54
  Line 253: _param_size = K2_JAX_PARAMS_SIZE_STAGE2_EXT_STANDALONE  # 54

pack_params_stage2() uses _param_size = 54 (EXT_STANDALONE) but constant says 41.
```

---

## 7. Current Dynamic q_ref Modes

### `original-k2-exact` (default, used for promotion)
- **Behavior:** STATIC q_ref from initial equilibrium_joint_pos captured at initialization
- **Claimed purpose:** "matches canonical K2 JAX path in simulate_hierarchical_controller.py"
- **Problem:** Static q_ref prevents CoM from following dynamic height commands — falls on ramp_up/up_down_cycle

### `setup-interp-debug` (debug-only)
- **Behavior:** APPROXIMATE linear interpolation from height setup files
- **Known defect:** Produces WORSE hip-yaw divergence (ramp_down hy=0.3728 vs 0.0977 static)
- **Usage rule:** NEVER for promotion validation

---

## 8. Known Suspicious Findings (from user brief)

1. `original-k2-exact` static q_ref causes dynamic height failure
2. `setup-interp-debug` interpolation was approximate and previously made ramp_down unsafe
3. Step C validation is not equivalent: original Step C used dynamic height patterns, but dedicated currently runs fixed height for C1-C5
4. Step D comparison has metric window mismatch: candidate pitch_rms is full episode, original is post-push 500-step
5. Step E and long-run show systematic support_rms and hip_yaw regressions
6. One test still fails: `K2_JAX_PARAMS_SIZE_STAGE2` says 41 while `pack_params_stage2()` returns 54
7. Some historical reports contradict each other about whether dynamic height canonical/static q_ref passes

---

## 9. Non-Negotiable Checks (from last report)

| Check | Status |
|---|---|
| No "expected outcome" as evidence | ✅ |
| No "infrastructure complete = promotion pass" | ✅ |
| No "tests-only success = behavioral pass" | ✅ |
| No "survival-only = pass" | ✅ |
| No "gate-only success = pass" | ✅ |
| No Step D full pass with partial conditions | ✅ |
| No Step C full pass with missing cases | ✅ |
| No dynamic height pass with missing scenarios | ✅ |
| No `setup-interp-debug` used in promotion | ✅ |
| No hidden missing files or failed commands | ✅ |

---

## 10. Reproducibility

To reproduce this blocked state:

```bash
git checkout 0e1c7135e22b4cb852f71a795426cd3d3f19753a

# Run full validation
python scripts/validate_k2_jax_dedicated_promotion.py \
  --scope all \
  --output-dir outputs/k2_jax_dedicated_promotion_validation

# Verify failing test
pytest tests/test_k2_jax_component_parity.py::TestParamsPackUnpackStage2::test_params_size_consistent -v

# Expected: 39/39 scenarios run, 3 SAFETY_FAIL in dynamic_height,
# SAFE_BUT_WORSE in all non-dynamic scopes, 1 test failure
```

---

## 11. Acceptance

- [x] Current BLOCKED state is documented and reproducible
- [x] Exact commit recorded
- [x] Exact commands recorded
- [x] All SAFETY_FAIL cases documented
- [x] All SAFE_BUT_WORSE scopes documented
- [x] Failed test documented
- [x] Dynamic q_ref modes documented
- [x] No code changes in this phase
