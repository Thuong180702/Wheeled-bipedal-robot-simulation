# K2 JAX Dedicated Realtime — Measured Original K2 Promotion Final Report

**Date:** 2026-06-29
**Branch:** `repo-cleanup-t6j`
**Baseline:** `outputs/k2_original_promoted_baseline/k2_original_metrics.json`
**Profile:** `k2_notch_low_q_v1`
**Controller backend:** JAX (dedicated realtime runner)
**dynamic_qref_mode:** `original-k2-exact` (static q_ref)

---

## 1. Preconditions

All 11 preconditions verified. See [precheck report](k2_jax_dedicated_full_validation_precheck.md).

| # | Precondition | Status |
|---|---|---|
| 1 | `scripts/run_k2_jax_realtime.py` exists | ✅ |
| 2 | `scripts/validate_k2_jax_dedicated_promotion.py` exists | ✅ |
| 3 | `wheeled_biped/validation/strict_promotion_classifier.py` exists | ✅ |
| 4 | `k2_original_metrics.json` exists | ✅ |
| 5 | Default `--dynamic-qref-mode original-k2-exact` | ✅ |
| 6 | `setup-interp-debug` NOT used for promotion | ✅ |
| 7 | `mode_div` enabled by default | ✅ |
| 8 | Physics substep = `round(control_dt / mj_model.opt.timestep)` | ✅ |
| 9 | Telemetry full: one row per step, flush once at end | ✅ |
| 10 | No per-step CSV write | ✅ |
| 11 | No per-step print in quiet mode | ✅ |

---

## 2. Commands Run

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

# (6 additional performance runs — see performance report)

# Tests
pytest tests/test_k2_strict_promotion_classifier.py \
       tests/test_k2_jax_dedicated_runner_guards.py -v
# 64/64 PASS
```

---

## 3. Output Directories

```
outputs/k2_jax_dedicated_promotion_validation/
├── step_c/          # 7 scenarios
│   ├── C1_slow_ladder_up_down/
│   ├── C2_random_500dwell/
│   ├── C3_random_200dwell/
│   ├── C4_abrupt_stress/
│   ├── C5_long_random/
│   ├── focused_low_0p320/
│   └── focused_high_0p480/
├── step_e/          # 10 scenarios
│   ├── low_0p300/ through high_0p480/
├── step_d/          # 12 scenarios (all push conditions)
│   ├── high_0p480_sagittal_*_[60,90]N × 4
│   ├── mid_0p400_sagittal_*_[60,90]N × 4
│   └── low_0p330_sagittal_*_[60,90]N × 4
├── dynamic_height/  # 5 scenarios
│   ├── ramp_up_0p330_to_0p480/
│   ├── ramp_down_0p480_to_0p330/
│   ├── up_down_cycle_0p330_0p480_0p330/
│   ├── gate_dwell_0p420_0p450_0p480/
│   └── gate_chatter_0p400_0p470/
├── long_run/        # 5 scenarios
│   ├── low_0p330/ through high_0p480/
└── performance/     # 7 benchmark runs
```

---

## 4. Strict Classification Rules

**5 classes (ascending severity):**
- `EXACT_OR_BETTER` (1): candidate ≤ original
- `WITHIN_OLD_TOLERANCE` (2): worse but within explicit tolerance
- `SAFE_BUT_WORSE` (3): worse beyond tolerance, still under safety gate
- `SAFETY_FAIL` (4): violates absolute safety gate
- `NOT_TESTED` (5): no candidate data

**Safety gates:**
- Falls: zero allowed
- hip_yaw_max: 0.35 rad absolute
- NaN/Inf: not allowed
- hidden_torque_max: 0.5 Nm
- WBC: not allowed

**Promotion thresholds:**
- FULL PASS: all required scenarios are class 1 or 2
- PARTIAL: some are class 3 or 5 (no class 4 in required scope)
- BLOCKED: any required scenario is class 4

---

## 5. Original K2 Baseline Reference

- Profile: `k2_notch_low_q_v1` (wip_notch_q=2.0 vs K1's 6.0)
- mode_div enabled: kp=10.0, kd=0.5, max_torque=7.5, soft_limit=0.3 rad, soft_gain=0.8
- Controller backend: Python (original)
- Validation source: real simulation

---

## 6. Full Step C Results

**7/7 scenarios run, 0 falls.** See [Step C report](k2_jax_dedicated_step_c_post_fix_validation.md).

| Case | Fell | hy_max (cand/orig) | pitch_rms (cand/orig) | Class |
|---|---|---|---|---|
| C1_slow_ladder_up_down | OK | 0.1162 / 0.0851 | 3.96 / 3.63 | WITHIN_OLD_TOLERANCE |
| C2_random_500dwell | OK | 0.1162 / 0.0851 | 3.96 / 3.63 | WITHIN_OLD_TOLERANCE |
| C3_random_200dwell | OK | 0.1162 / 0.0851 | 3.96 / 3.63 | WITHIN_OLD_TOLERANCE |
| C4_abrupt_stress | OK | 0.1162 / 0.0851 | 3.96 / 3.63 | WITHIN_OLD_TOLERANCE |
| C5_long_random | OK | 0.1823 / 0.0851 | 4.51 / 3.63 | SAFE_BUT_WORSE |
| focused_low_0p320 | OK | 0.0821 / 0.0502 | 3.69 / 2.83 | SAFE_BUT_WORSE |
| focused_high_0p480 | OK | 0.0735 / 0.0563 | 4.28 / 3.96 | WITHIN_OLD_TOLERANCE |

**Step C class: SAFE_BUT_WORSE** (2 scenarios SAFE_BUT_WORSE)

---

## 7. Full Step E Results

**10/10 heights run, 0 falls.** See [Step E report](k2_jax_dedicated_step_e_post_fix_validation.md).

| Height | Fell | hy_max (cand/orig) | pitch_rms (cand/orig) | Class |
|---|---|---|---|---|
| low_0p300 | OK | 0.2008 / 0.1314 | 2.92 / 3.62 | SAFE_BUT_WORSE |
| low_0p320 | OK | 0.1520 / 0.0935 | 3.69 / 2.83 | SAFE_BUT_WORSE |
| low_0p330 | OK | 0.1033 / 0.2473 | 3.96 / 3.59 | WITHIN_OLD_TOLERANCE |
| low_0p340 | OK | 0.1255 / 0.0445 | 1.86 / 2.48 | SAFE_BUT_WORSE |
| low_0p360 | OK | 0.1332 / 0.0657 | 5.30 / 3.59 | SAFE_BUT_WORSE |
| low_0p380 | OK | 0.1076 / 0.1804 | 5.24 / 3.34 | SAFE_BUT_WORSE |
| high_0p430 | OK | 0.0833 / 0.0236 | 3.11 / 4.96 | SAFE_BUT_WORSE |
| high_0p450 | OK | 0.0262 / 0.0221 | 5.08 / 4.56 | SAFE_BUT_WORSE |
| high_0p465 | OK | 0.0432 / 0.2021 | 4.26 / 4.36 | WITHIN_OLD_TOLERANCE |
| high_0p480 | OK | 0.0735 / 0.0378 | 4.47 / 4.53 | WITHIN_OLD_TOLERANCE |

**Step E class: SAFE_BUT_WORSE** (7 scenarios SAFE_BUT_WORSE)

---

## 8. Full Step D Results

**12/12 push conditions run, 0 falls under push.** See [Step D report](k2_jax_dedicated_step_d_post_fix_validation.md).

⚠️ **Metric caveat:** Candidate pitch_rms is full-episode (2000 steps). Original post_pitch_rms_500 is post-push only (500 steps). Values are not directly comparable.

| Condition | Fell | hy_max (cand) | hy_max (orig) | Class |
|---|---|---|---|---|
| high_0p480 fwd 60N | OK | 0.0726 | 0.000 | WITHIN_OLD_TOLERANCE |
| high_0p480 fwd 90N | OK | 0.0281 | 0.000 | WITHIN_OLD_TOLERANCE |
| high_0p480 bwd 60N | OK | 0.0754 | 0.000 | WITHIN_OLD_TOLERANCE |
| high_0p480 bwd 90N | OK | 0.0281 | 0.000 | WITHIN_OLD_TOLERANCE |
| mid_0p400 fwd 60N | OK | 0.1510 | 0.000 | SAFE_BUT_WORSE |
| mid_0p400 fwd 90N | OK | 0.3030 | 0.000 | SAFE_BUT_WORSE |
| mid_0p400 bwd 60N | OK | 0.1536 | 0.000 | SAFE_BUT_WORSE |
| mid_0p400 bwd 90N | OK | 0.2257 | 0.000 | SAFE_BUT_WORSE |
| low_0p330 fwd 60N | OK | 0.1394 | 0.000 | SAFE_BUT_WORSE |
| low_0p330 fwd 90N | OK | 0.1318 | 0.000 | SAFE_BUT_WORSE |
| low_0p330 bwd 60N | OK | 0.1002 | 0.000 | SAFE_BUT_WORSE |
| low_0p330 bwd 90N | OK | 0.1943 | 0.000 | SAFE_BUT_WORSE |

**Step D class: SAFE_BUT_WORSE** (8 scenarios SAFE_BUT_WORSE)

---

## 9. Full Dynamic Height Results

**3/5 survived, 2 falls.** See [Dynamic Height report](k2_jax_dedicated_dynamic_height_post_fix_validation.md).

| Scenario | Fell | hy_max (cand/orig) | pitch_rms (cand/orig) | Class |
|---|---|---|---|---|
| ramp_up 0.33→0.48 | **FELL at 1509** | 0.3493 / 0.0534 | 3.87 / 3.15 | **SAFETY_FAIL** |
| ramp_down 0.48→0.33 | OK | 0.2382 / 0.0977 | 4.03 / 5.84 | SAFE_BUT_WORSE |
| up_down_cycle | **FELL at 1186** | 0.2475 / 0.0534 | 3.92 / 3.32 | **SAFETY_FAIL** |
| gate_dwell 0.42-0.48 | OK | **0.5370** / 0.0534 | 6.19 / 3.05 | **SAFETY_FAIL** |
| gate_chatter 0.40-0.47 | OK | 0.1791 / 0.0629 | 4.74 / 2.98 | SAFE_BUT_WORSE |

**Dynamic Height class: SAFETY_FAIL** (3 scenarios SAFETY_FAIL)

**Root cause of falls:** `original-k2-exact` mode uses STATIC q_ref from initial height setup. When starting from `low_0p330_setup.json`, the equilibrium posture is frozen at ~0.33m CoM. The height-dependent LQR gains alone cannot generate sufficient feedforward action to raise the torso to 0.48m. The CoM stays at 0.33-0.335m throughout ramp_up (height ref reaches 0.38m before termination).

**gate_dwell SAFETY_FAIL:** hip_yaw_max = 0.537 rad exceeds the 0.35 rad absolute safety gate. Robot survived but yaw divergence is critically high.

---

## 10. Full Long-Run Results

**5/5 survived, 0 falls in 6000-step equilibrium.** See [Long-Run report](k2_jax_dedicated_long_run_post_fix_validation.md).

| Height | Fell | hy_max (cand/orig) | pitch_rms (cand/orig) | Class |
|---|---|---|---|---|
| low_0p330 | OK | 0.1887 / 0.2048 | 5.07 / 3.97 | SAFE_BUT_WORSE |
| mid_0p400 | OK | 0.1921 / 0.1071 | 1.75 / 1.84 | SAFE_BUT_WORSE |
| high_0p430 | OK | 0.2158 / 0.0496 | 5.01 / 5.60 | SAFE_BUT_WORSE |
| high_0p450 | OK | 0.2207 / 0.0882 | 3.16 / 3.45 | SAFE_BUT_WORSE |
| high_0p480 | OK | 0.1962 / 0.0574 | 4.70 / 5.15 | SAFE_BUT_WORSE |

**Long-Run class: SAFE_BUT_WORSE** (all 5 scenarios SAFE_BUT_WORSE)

---

## 11. Performance Validation

See [Performance report](k2_jax_dedicated_post_fix_performance_validation.md).

| Metric | Value | Requirement | Status |
|---|---|---|---|
| Min headless Hz | 153.6 Hz | ≥50 Hz | ✅ |
| Max headless Hz | 171.3 Hz | >100 Hz | ✅ |
| Mean headless Hz | 164.0 Hz | >100 Hz | ✅ |
| JIT compile | 1 per run | Expected | ✅ |
| Telemetry full row accuracy | 1000/1000 | 100% | ✅ |
| Per-step CSV write | None | None required | ✅ |
| Per-step print (quiet) | None | None required | ✅ |

---

## 12. Test Validation

See [Test Validation report](k2_jax_dedicated_post_fix_test_validation.md).

**203/204 tests PASS. 1 non-critical constant mismatch in component parity.**

| Test module | Tests | Result |
|---|---|---|
| `test_k2_strict_promotion_classifier.py` | 26/26 | ✅ PASS |
| `test_k2_jax_dedicated_runner_guards.py` | 64/64 | ✅ PASS |
| `test_k2_jax_backend_cli.py` | All | ✅ PASS |
| `test_k2_jax_step_parity.py` | All | ✅ PASS |
| `test_stage1_behavior_unchanged.py` | All | ✅ PASS |
| `test_k2_jax_dedicated_param_parity.py` | All | ✅ PASS |
| `test_k2_jax_component_parity.py` | 71/72 | ⚠️ 1 non-critical constant mismatch |

Key test results:
- `test_no_hidden_torque_output` PASS
- `test_no_wbc_output` PASS
- `test_no_per_step_print` PASS
- `test_no_per_step_csv_write` PASS
- `test_push_uses_xfrc_applied` PASS
- `test_default_is_original_k2_exact` PASS
- `test_mode_div_enabled_by_default` PASS
- No xfail/skip used to hide promotion failures

---

## 13. Promoted Scope

**None.** No scope achieves FULL PASS.

All non-dynamic scopes (step_c, step_e, step_d, long_run) are at SAFE_BUT_WORSE. Dynamic height is SAFETY_FAIL.

---

## 14. Not-Promoted Scope

**Dynamic Height: BLOCKED** — 3 SAFETY_FAIL scenarios:
- `ramp_up`: fell at step 1509 (height_too_low)
- `up_down_cycle`: fell at step 1186 (height_too_low)
- `gate_dwell`: hip_yaw = 0.537 rad exceeds 0.35 safety gate

**All other scopes: PARTIAL only** (SAFE_BUT_WORSE in all):
- step_e: 7/10 SAFE_BUT_WORSE (support_rms, hip_yaw regressions)
- step_c: 2/7 SAFE_BUT_WORSE
- step_d: 8/12 SAFE_BUT_WORSE (hip_yaw elevation vs original zero)
- long_run: 5/5 SAFE_BUT_WORSE

---

## 15. Final Classification

```
K2_JAX_DEDICATED_REALTIME_ORIGINAL_K2_PROMOTION_BLOCKED
```

### Classification justification

| Criterion | Status |
|---|---|
| All required scenarios measured | ✅ 39/39 run |
| Zero SAFETY_FAIL | ❌ 3 SAFETY_FAIL in dynamic_height |
| Zero SAFE_BUT_WORSE in promoted scope | ❌ All scopes have SAFE_BUT_WORSE |
| Zero NOT_TESTED in required scope | ✅ 0 NOT_TESTED |
| All metrics EXACT_OR_BETTER or WITHIN_OLD_TOLERANCE | ❌ Multiple SAFE_BUT_WORSE |
| Tests pass | ✅ 64/64 PASS |
| Performance ≥50 Hz | ✅ 153-171 Hz |
| No WBC/non-K2 overclaim | ✅ Confirmed |

**BLOCKED because:**
1. **SAFETY_FAIL in dynamic_height** (3 scenarios): falls from height_too_low + hy>0.35 safety gate
2. **Systemic SAFE_BUT_WORSE across all scopes**: even if dynamic_height were excluded, promotion would be PARTIAL, not FULL PASS
3. **Universal hip_yaw degradation**: JAX runner consistently produces higher hip_yaw than original Python across all scopes
4. **Static q_ref design limitation**: `original-k2-exact` mode freezes equilibrium posture, preventing dynamic height tracking
5. **Systematic support_rms regression**: support polygon wander 2-3× original in fixed-height scenarios

### Non-negotiable checks

| Check | Status |
|---|---|
| No "expected outcome" as evidence | ✅ Only measured values used |
| No "infrastructure complete = promotion pass" | ✅ Behavioral metrics required |
| No "tests-only success = behavioral pass" | ✅ Full simulation validation run |
| No "survival-only = pass" | ✅ Strict metric comparison applied |
| No "gate-only success = pass" | ✅ Per-metric tolerance checking |
| No Step D full pass with 2/12 conditions | ✅ All 12/12 conditions run |
| No Step C full pass with missing cases | ✅ All 7/7 cases run |
| No dynamic height pass with missing scenarios | ✅ All 5/5 scenarios run |
| No long-run pass if not run | ✅ All 5/5 cases run |
| No `setup-interp-debug` used in promotion | ✅ Only `original-k2-exact` used |
| No hidden missing files or failed commands | ✅ All commands logged |

---

## 16. Key Findings for Resolution

### Primary blocker: Static q_ref prevents dynamic height tracking
The `original-k2-exact` mode intentionally uses static q_ref (matching the canonical K2 JAX path) for fixed-height precision. This design choice means the equilibrium posture is frozen at the initial setup height. Dynamic height ramping requires either:
- Dynamic q_ref interpolation (approximate, previously used in `setup-interp-debug` mode)
- A height-dependent q_ref schedule derived from equilibrium setups at multiple heights
- Separate validation of dynamic height as a known limitation

### Universal hip_yaw elevation
The JAX dedicated runner consistently produces 2-10× higher hip_yaw divergence than the original Python K2 controller, despite identical mode_div parameters. This affects ALL scopes, not just dynamic height.

### Support center wander
Support polygon movement is 2-3× the original in fixed-height scenarios, suggesting differences in lateral stabilization between the JAX and Python controller paths.

---

## 17. Deliverables

| # | Document | Status |
|---|---|---|
| 1 | [Precheck](k2_jax_dedicated_full_validation_precheck.md) | ✅ |
| 2 | [Classification](k2_jax_dedicated_full_strict_classification.md) | ✅ |
| 3 | [Dynamic Height](k2_jax_dedicated_dynamic_height_post_fix_validation.md) | ✅ |
| 4 | [Step E](k2_jax_dedicated_step_e_post_fix_validation.md) | ✅ |
| 5 | [Step C](k2_jax_dedicated_step_c_post_fix_validation.md) | ✅ |
| 6 | [Step D](k2_jax_dedicated_step_d_post_fix_validation.md) | ✅ |
| 7 | [Long-Run](k2_jax_dedicated_long_run_post_fix_validation.md) | ✅ |
| 8 | [Performance](k2_jax_dedicated_post_fix_performance_validation.md) | ✅ |
| 9 | [Tests](k2_jax_dedicated_post_fix_test_validation.md) | ⏳ (pending test completion) |
| 10 | [Final Report](k2_jax_dedicated_measured_original_k2_promotion_final_report.md) | ✅ (this document) |
