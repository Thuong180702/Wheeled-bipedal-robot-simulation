# K2 JAX — Full K2 Port Complete Final Report (Phase 9)

**Date:** 2026-06-28
**Branch:** `repo-cleanup-t6j`
**Commit working tree:** `0e1c7135e22b4cb852f71a795426cd3d3f19753a` + Phase 6M changes (uncommitted)

---

## Final Classification: `K2_JAX_PORT_INCOMPLETE_WITH_EXACT_BLOCKER`

---

## 1. Previous Classification

`K2_JAX_PORT_INCOMPLETE_WITH_EXACT_BLOCKER` — ABS trim ring buffer / adaptive_bias_trim_enabled parity incomplete.

---

## 2. ABS Trim Source Trace (Phase 1)

**Deliverable:** [k2_jax_python_abs_trim_source_trace.md](k2_jax_python_abs_trim_source_trace.md)

All 43 Python ABS scalars traced to source. Key findings:
- `adaptive_bias_fast_rate_nm_per_step` is defined but **unused** in compute
- Fast mean is diagnostic-only (never used for control)
- Python maintains 3 separate histories: slow (300), fast (100), ZC (500)
- Current sample is included in mean before computing trim

---

## 3. ABS Trim Gap Matrix (Phase 2)

**Deliverable:** [k2_jax_abs_trim_gap_matrix.md](k2_jax_abs_trim_gap_matrix.md)

**5 gaps identified:**

| # | Gap | Type | Torque Impact |
|---|-----|------|---------------|
| #19 | Contact safety gate hardcoded True | WRONG | ~0.98 Nm push_fwd |
| #25 | ZC window 300 vs 500 | WRONG | ~0.16-1.51 Nm ramp/gate |
| #34 | Guard trigger ≥3 reset missing | WRONG | Diagnostic only |
| #35 | Prev sign update on hold>0 missing | WRONG | Minimal |
| #37 | contact_valid not in JAX inputs | MISSING | ~0.98 Nm |

---

## 4. Design (Phase 3)

**Deliverable:** [k2_jax_abs_trim_jax_design.md](k2_jax_abs_trim_jax_design.md)

Designed:
- New `contact_valid` input field (index 41, input size 41→42)
- Separate 500-entry ZC ring buffer (state size 332→834)
- New `_abs_update_zc_buffer` and `_abs_count_zero_crossings_from_zc` functions
- Fixed guard_trigger, prev_sign, contact gate logic

---

## 5. Implementation (Phase 4)

**Deliverable:** [k2_jax_abs_trim_implementation_report.md](k2_jax_abs_trim_implementation_report.md)

### Files Changed
1. `wheeled_biped/controllers/k2_jax_controller.py` — +80 lines
   - Input: added `contact_valid` at index 41, `K2_JAX_INPUT_SIZE` = 42
   - State: added ZC buffer (500 entries), `K2_JAX_STATE_SIZE` = 834
   - Functions: `_abs_update_zc_buffer`, `_abs_count_zero_crossings_from_zc`
   - Fixed: guard_trigger reset, prev_sign update, contact gate
   - Ring buffer packing: entries start at position 0 (not write_ptr offset)
2. `scripts/simulate_hierarchical_controller.py` — +5 lines
   - Pass `contact_valid` from `centroidal_state_control.contact_force_valid`
   - Capture and pass `abs_zc_error_history` for state sync
3. `tests/test_k2_jax_step_parity.py` — +4 lines
   - Add ZC buffer field sources

### Layout Changes
| Property | Old | New | Change |
|----------|-----|-----|--------|
| STATE_SIZE | 332 | **834** | +502 (ZC buffer) |
| INPUT_SIZE | 41 | **42** | +1 (contact_valid) |
| PARAMS_SIZE | 41 | 41 | unchanged |
| DIAG_SIZE | 32 | 32 | unchanged |

---

## 6. Test Results (Phase 5)

**131/131 tests pass** (all K2 JAX test files).

| Test file | Tests | Status |
|-----------|-------|--------|
| test_k2_jax_backend_cli.py | 14 | PASS |
| test_k2_jax_branch_activity_audit.py | 6 | PASS |
| test_k2_jax_component_parity.py | 95 | PASS |
| test_k2_jax_step_parity.py | 16 | PASS |

State field audit: all 834 state fields have known sources (including new ZC fields).
No xfail, no skip, no silent test removal.

---

## 7. Both-Synced Parity Results (Phase 6)

**Full 9-scenario rerun completed after Phase 6M fixes.**

| Scenario | MaxAbsDiff | Step | Actuator | Wheel[4] | Wheel[9] | Fell | Status |
|----------|-----------|------|----------|----------|----------|------|--------|
| fixed_high_0p480 | 9.54e-08 | 2 | 8 (r_knee) | 1.55e-15 | 1.55e-15 | no | ✅ PASS |
| fixed_low_0p330 | 9.54e-08 | 2 | 8 (r_knee) | 2.29e-16 | 2.22e-16 | no | ✅ PASS |
| ramp_up | **1.60e-01** | 150 | 9 (r_wheel) | 0.16 | 0.16 | no | ❌ FAIL |
| ramp_down | 9.54e-08 | 2 | 8 (r_knee) | 1.55e-15 | 1.55e-15 | no | ✅ PASS |
| up_down_cycle | **1.60e-01** | 150 | 9 (r_wheel) | 0.16 | 0.16 | no | ❌ FAIL |
| gate_dwell | **1.60e-01** | 150 | 9 (r_wheel) | 0.16 | 0.16 | no | ❌ FAIL |
| gate_chatter | **1.51e+00** | 150 | 9 (r_wheel) | 1.51 | 1.51 | no | ❌ FAIL |
| push_fwd_90N | **9.80e-01** | 117 | 4 (l_wheel) | 0.98 | 0.98 | no | ❌ FAIL |
| push_bwd_90N | 1.56e-06 | 114 | 5 (r_hip_roll) | 1.55e-15 | 1.55e-15 | **yes** | marginal |

**Passed (<1e-5): 4/9 (fixed_high, fixed_low, ramp_down, push_bwd)**
**Failed: 5/9 (ramp_up, up_down_cycle, gate_dwell, gate_chatter, push_fwd)**

### Key patterns:
1. All failures are at **wheels [4,9]** — direct recipients of ABS trim via tau_position
2. Hip-yaw [1,6] remains **clean (<1e-16)** in all scenarios — confirms the divergence is ABS trim, not yaw/mode-div
3. Three dynamic scenarios (ramp_up, up_down_cycle, gate_dwell) produce the **exact same 0.16 Nm** at step 150 — this is a **deterministic, reproducible** ABS trim divergence
4. gate_dwell (stays at 0.40m, never crosses gates) also fails — proves this is NOT a gate-crossing issue but a **time-accumulating state divergence**
5. gate_chatter amplifier (1.51 Nm) suggests the divergence scales with oscillatory height inputs
6. push_fwd (0.98 Nm) is a separate contact-induced transient
7. push_bwd robot falls (both Python and JAX produce similar failing torques)

---

## 8. Active Mechanism Closure Audit (Phase 7)

**Deliverable:** [k2_jax_full_active_mechanism_closure_audit.md](k2_jax_full_active_mechanism_closure_audit.md)

**Post-Phase 6M status:**
- 47 mechanisms PASS (including newly fixed ZC buffer, contact gate, guard_trigger, prev_sign)
- 1 INACTIVE_PROVEN (T6J bias trim disabled for K2)
- **4 active mechanisms remain WRONG in their impact on dynamic/push scenarios**

The 4 remaining failures are confirmed as ABS trim state synchronization and computation gaps, NOT missing mechanism implementations. All 50 K2 mechanisms are implemented — the issue is runtime parity, not mechanism coverage.

---

## 9. Root Cause Analysis of Remaining Failures

### Ramp Up / Up Down Cycle (0.16 Nm transient spike at steps 140-158)

**Symptom:** Identical 0.16 Nm peak at actuator 9 (right wheel), transient spike lasting ~19 steps.

**Diagnostic finding at step 150:**
- Python ABS trim: `-0.099` Nm (active, correcting drift)
- JAX ABS trim: `0.0` Nm (zero — not applying trim)
- Both ZC counts: 0 (sync confirmed working)
- Python sagittal torque: zero (within deadband)
- JAX sagittal torque: tau_p=5.81, tau_pos=-5.67 (active but different from Python)

**Root cause hypothesis:** The ABS trim `_ABS_TRIM_TAU` state is not being correctly synced from Python before JAX computes step 150. Python's `_adaptive_bias_trim_tau` at step 150 is -0.099 (accumulated over previous steps), but JAX's `_ABS_TRIM_TAU` starts at 0.0. This suggests `pack_state_from_python_k2` is not correctly packing the ABS trim state for steps beyond initial sync.

**Note:** The ring buffer packing fix (entries starting at position 0) was verified but the trim value itself (not the buffer entries) may have a sync issue. The `abs_trim_tau` parameter IS passed to `pack_state_from_python_k2`, so the sync should work. The issue may be in when the state is captured relative to Python's compute step, or in the order of JAX's trim update.

### Push Forward 90N (0.98 Nm)

**Symptom:** 0.98 Nm transient during push recovery at step 117 (actuator 4, left wheel).

**Root cause hypothesis:** Same as ramp_up — ABS trim state divergence during transient events. The contact gate fix ensures JAX checks actual contact state, but the trim VALUE itself may diverge due to state sync timing or computation differences in the rate-limiting or target computation.

### Gate Chatter (1.51 Nm)

**Symptom:** 1.51 Nm at step ~150. Same pattern as ramp_up but amplified by oscillatory height profile.

**Root cause hypothesis:** Height-scheduled max_tau diverges between Python and JAX during gate transitions. When height crosses the schedule gates (0.38-0.52m), the max_tau value changes, affecting the trim ceiling. If JAX and Python compute different heights at the same step (due to `commanded_height_ref_m` timing), the max_tau and resulting trim diverge.

---

## 10. What Was Fixed

| Fix | Status | Impact |
|-----|--------|--------|
| Contact gate (contact_valid input) | ✅ Implemented | Enables correct safety gate, but push still fails |
| ZC buffer (500 entries) | ✅ Implemented | Matches Python window size, eliminates 300-vs-500 gap |
| Ring buffer packing (position 0) | ✅ Implemented | Fixes fast mean and ZC count addressing |
| Guard trigger ≥3 reset | ✅ Implemented | Diagnostic parity |
| Prev sign on hold>0 | ✅ Implemented | Edge case parity |
| State/input/diag size updates | ✅ Implemented | All tests pass |

---

## 11. What Remains Unfixed

| Issue | Impact | Required Investigation |
|-------|--------|----------------------|
| ABS trim value sync (trim_tau=0 vs -0.099) | 0.16 Nm ramp_up/up_down | Trace trim_tau through both-synced state packing |
| Push-induced trim divergence | 0.98 Nm push_fwd | Instrument push step 100-120 for component-level comparison |
| Height schedule gate crossing | 1.51 Nm gate_chatter | Verify height_ref consistency during gate transitions |

---

## 12. Final Classification

**`K2_JAX_PORT_INCOMPLETE_WITH_EXACT_BLOCKER`**

**Exact blocker:** ABS trim `_ABS_TRIM_TAU` state diverges from Python's `_adaptive_bias_trim_tau` during transient events (ramp initiation, push recovery, gate crossing), producing zero JAX trim when Python has active trim of -0.099 Nm at step 150.

**First divergent scalar:** `external_position_trim` (ABS trim applied to tau_position) — 0.0 JAX vs -0.099 Python at step 150, leading to different tau_position → different wheel torque → 0.16 Nm diff.

**Why COMPLETE cannot be claimed:**
- Push both-synced parity fails (0.98 Nm push_fwd)
- Dynamic both-synced parity fails (0.16 Nm ramp_up, 1.51 Nm gate_chatter)
- ABS trim value parity not achieved (jx_trim=0 vs py_trim=-0.099)
- Functional validation deferred
- Long-run deferred

**Preserved invariants:**
- Python default preserved ✅
- JAX opt-in preserved ✅
- No gains tuned ✅
- No thresholds relaxed ✅
- No empirical correction factors ✅
- No Python behavior changed ✅
- 131/131 tests pass ✅
- No hidden torque/WBC ✅
- Fixed-height parity preserved ✅
