# Phase 5: Height Range Extension Strategy

**Date:** 2026-06-06
**Phase:** HEIGHT_RANGE_EXTENSION_STRATEGY

---

## 1. Design Principles

### 1.1 Keep Old Baseline as Default

The old baseline must remain protected:

| Protected Element | Value | Justification |
|-----------------|-------|---------------|
| Default profile | `candidate_D2_wheel_velocity_damping_light` | Step E/C 5/5 PASS |
| Default HY2-DIV | **DISABLED** | Baseline behavior |
| Default WBC | **OFF** | Structural invariant |
| Official gates | Step E/C thresholds | Validation standard |

**Implication:** Any extension work must be **opt-in**, not a default behavior change.

### 1.2 Extreme-Height Extension Must Be Opt-In

New profiles must be explicitly enabled:

```bash
# NOT this:
python scripts/simulate_hierarchical_controller.py ...  # uses default, unchanged

# MUST be this:
python scripts/simulate_hierarchical_controller.py \
  --enable-extreme-height-profile EXTREME_D2_A0_LADDER \
  ...
```

### 1.3 Extend Gradually

Do NOT jump from 0.394m to 0.300m or 0.414m to 0.480m in one step.

Use incremental ladder with step sizes of ~20-30mm.

---

## 2. Height Ladder Design

### Low-Side Ladder (Starting from 0.394m)

| Step | Target Height (m) | Achieved Height (m) | Setup File | Evidence Needed |
|------|------------------|---------------------|------------|-----------------|
| 0 | **0.394** | 0.394 | baseline | **VALIDATED** |
| 1 | 0.380 | (interpolate) | generate | Static feasible |
| 2 | **0.360** | 0.363 | `low_0p360_setup.json` | **EXISTS** |
| 3 | 0.340 | (interpolate) | generate | Static feasible |
| 4 | **0.330** | 0.335 | `low_0p330_setup.json` | **EXISTS** |
| 5 | 0.320 | (interpolate) | generate | Static feasible |
| 6 | **0.300** | 0.295 | `low_0p300_setup.json` | **EXISTS** (posture FAIL) |

### High-Side Ladder (Starting from 0.414m)

| Step | Target Height (m) | Achieved Height (m) | Setup File | Evidence Needed |
|------|------------------|---------------------|------------|-----------------|
| 0 | **0.414** | 0.413 | baseline | **VALIDATED** |
| 1 | 0.430 | (interpolate) | generate | Static feasible |
| 2 | 0.450 | 0.451 | `high_0p450_setup.json` | **EXISTS** |
| 3 | 0.465 | (interpolate) | generate | Static feasible |
| 4 | **0.480** | 0.481 | `high_0p480_setup.json` | **EXISTS** (posture FAIL) |

---

## 3. Validation Order at Each Height

At each ladder step, validate in this order:

```
┌─────────────────────────────────────────────────────────────────┐
│ Stage 1: Static Feasibility (if not already done)               │
│   - Load setup file                                            │
│   - Verify wheel contact, no non-wheel contact                 │
│   - Verify joint limits                                        │
│   Gate: static_feasible = true                                 │
├─────────────────────────────────────────────────────────────────┤
│ Stage 2: 100-Step Posture Smoke                                │
│   - Run 100 steps with candidate profile                       │
│   - Check survival, contact, height                            │
│   Gate: survived_100 AND contact > 99% AND height_error < 0.05 │
├─────────────────────────────────────────────────────────────────┤
│ Stage 3: 500-Step Posture Screening                            │
│   - Run 500 steps                                             │
│   - Check divergence RMS, hip_yaw max                          │
│   Gate: div_RMS < 0.50 AND hip_yaw_max < 0.50                 │
├─────────────────────────────────────────────────────────────────┤
│ Stage 4: 2000-Step Posture Validation                          │
│   - Run 2000 steps                                            │
│   - Check divergence trend, support drift                       │
│   Gate: div_trend_bounded AND support < 0.20                 │
├─────────────────────────────────────────────────────────────────┤
│ Stage 5: 5000-Step Official Step E (if all above pass)         │
│   - Run 5000 steps                                            │
│   - Apply official Step E gates                                │
│   Gate: ALL official Step E gates                             │
├─────────────────────────────────────────────────────────────────┤
│ Stage 6: Step C Height Recovery (after Step E passes)          │
│   - Run Step C height recovery                                 │
│   Gate: ALL official Step C gates                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Candidate Profile Families

### Family A: Baseline-Only Ladder

**Purpose:** Find where old controller first fails on the ladder.

**Configuration:**
```python
profile_name = "baseline_only_ladder"
sagittal_authority_profile = "candidate_D2_wheel_velocity_damping_light"
hy2_div_enabled = False  # No HY2-DIV
```

**Expected behavior:**
- At 0.394m: PASS (baseline)
- As height decreases toward 0.300m: gradual degradation
- As height increases toward 0.480m: gradual degradation
- Fail point indicates where extension mechanism is needed

**Risk:** LOW — does not modify default behavior
**First validation height:** 0.360m (low side), 0.450m (high side)

**Stop condition:** Step E fails OR divergence exceeds acceptable threshold

**Rollback rule:** Disable extreme-height profile → revert to baseline

---

### Family B: HY2-DIV A0 Ladder

**Purpose:** Test whether A0 helps at intermediate heights.

**Configuration:**
```python
profile_name = "hy2_div_a0_ladder"
sagittal_authority_profile = "candidate_D2_wheel_velocity_damping_light"
hy2_div_enabled = True
hy2_div_profile = "hy2_div_A0"  # k=5.0, kd=1.0, tau_max=0.5, z_low=0.300, z_high=0.393
```

**Expected behavior:**
- Below z_low=0.300m: HY2-DIV fully active (gate=1.0)
- Above z_high=0.393m: HY2-DIV inactive (gate=0.0)
- For 0.330-0.393m: partial gate activation

**Risk:** MEDIUM — enables HY2-DIV, may affect nominal behavior
**First validation height:** 0.360m (low side), 0.450m (high side)

**Stop condition:**
- Clipping > 50% at any height → insufficient authority
- Nominal regression → HY2-DIV interference
- Step E fails

**Rollback rule:** Set hy2_div_enabled=False → revert to Family A

---

### Family C: Extended-Gate HY2-DIV Ladder

**Purpose:** Test whether extending gate upward helps high-side.

**Configuration:**
```python
profile_name = "hy2_div_extended_gate_ladder"
sagittal_authority_profile = "candidate_D2_wheel_velocity_damping_light"
hy2_div_enabled = True
hy2_div_profile = "hy2_div_B1"  # k=5.0, kd=1.0, tau_max=1.0, z_low=0.300, z_high=0.500
```

**Evidence for this family:**
- B1 showed slight nominal degradation (0.0242 vs 0.0230 RMS)
- But B1 might help at 0.450-0.480m where nominal A0 has gate=0

**Risk:** MEDIUM-HIGH — extends HY2-DIV to nominal range
**First validation height:** 0.450m (high side only)

**Stop condition:**
- Nominal regression > 10% → reject family
- Step E fails at any height

**Rollback rule:** Set hy2_div_z_high=0.393 → revert to Family B

---

### Family D: Stronger Low-Side Authority

**Purpose:** Test whether higher tau_max helps at very low heights.

**Configuration:**
```python
profile_name = "strong_low_authority"
sagittal_authority_profile = "candidate_D2_wheel_velocity_damping_light"
hy2_div_enabled = True
hy2_div_profile = "hy2_div_A3"  # k=7.5, kd=1.5, tau_max=1.0, z_low=0.300, z_high=0.393
```

**Evidence for this family:**
- A1/A2 (higher tau_max) didn't improve divergence at 5000 steps
- But A3 has BOTH higher gains AND higher tau_max
- May be needed for very low heights (0.300-0.320m)

**Risk:** MEDIUM — higher gains may cause instability
**First validation height:** 0.300m (only if Family B fails at low)

**Stop condition:**
- Instability at any height
- Step E fails

**Rollback rule:** Set hy2_div_profile=hy2_div_A0 → revert to Family B

---

### Family E: Support-Drift-Aware Controller

**Purpose:** Address high-side support drift coupling.

**Configuration:**
```python
profile_name = "support_drift_aware"
sagittal_authority_profile = "candidate_D2_wheel_velocity_damping_light"
# Plus: support-position feedback scheduling
```

**Evidence for this family:**
- high_0p480 shows support drift 3× baseline (0.378m vs 0.10m)
- Support drift correlates with divergence (r=-0.517)
- Current sagittal profile may not handle higher heights well

**Risk:** HIGH — requires controller modification
**First validation height:** 0.450m (only if Families B/C fail at high)

**Requires:**
- Controller modification (not currently available)
- Separate validation phase

**Rollback rule:** Revert to Family B/C

---

## 5. Gate Definitions for Extension

### Posture-First Gates (Before Official Step E)

| Gate | Threshold | Rationale |
|------|-----------|-----------|
| `survived_full_run` | true | Robot must survive |
| `contact_valid` | >= 99.5% | Contact must be maintained |
| `height_error` | < 0.05 m | Height tracking acceptable |
| `wbc_applied` | false | Structural invariant |
| `hidden_torque` | = 0.0 | Structural invariant |
| `ownership_violations` | = 0 | Structural invariant |
| `roll_max` | < 0.02 rad | Roll must be bounded |

### Posture Gates (Posture-Only Validation)

| Gate | Threshold | Rationale |
|------|-----------|-----------|
| `divergence_RMS` | < target | Height-dependent (see below) |
| `hip_yaw_abs_max` | < 0.30 rad | Hard limit |

| Height Region | Divergence RMS Target |
|--------------|---------------------|
| 0.360-0.394m | < 0.15 rad |
| 0.320-0.360m | < 0.25 rad |
| 0.300-0.320m | < 0.30 rad |
| 0.414-0.450m | < 0.20 rad |
| 0.450-0.480m | < 0.25 rad |

### Official Step E Gates (After Posture Pass)

Same as baseline:
- `support_max < 0.15 m`
- `wheel_vel_max < 5.0 rad/s`
- `contact_valid >= 99.9%`
- All structural invariants

### Pitch/Support Drift (Deferred)

| Metric | Status | Gate |
|--------|--------|------|
| `pitch_max` | RECORDED | DEFERRED to task-aware pitch |
| `support_drift_max` | RECORDED | DEFERRED to support controller phase |

---

## 6. Strategy Summary

### Low-Side Strategy (0.300m target)

```
0.394m [VALIDATED]
   ↓ (Family A: baseline_only_ladder)
0.380m [test]
   ↓
0.360m [test] ← EXISTS setup
   ↓
0.340m [test]
   ↓
0.330m [test] ← EXISTS setup
   ↓
0.320m [test]
   ↓
0.300m [test] ← EXISTS setup (posture FAIL currently)
   ↓ IF FAIL: Family B (A0)
   ↓ IF STILL FAIL: Family D (A3)
```

### High-Side Strategy (0.480m target)

```
0.414m [VALIDATED]
   ↓ (Family A: baseline_only_ladder)
0.430m [test]
   ↓
0.450m [test] ← EXISTS setup
   ↓ IF FAIL: Family B (A0 with gate=0 at this height)
   ↓ IF STILL FAIL: Family C (B1 extended gate)
   ↓ IF STILL FAIL: Family E (support-drift-aware)
0.465m [test]
   ↓
0.480m [test] ← EXISTS setup (posture FAIL currently)
```

---

## 7. What Must NOT Be Changed

| Element | Prohibition | Justification |
|---------|------------|---------------|
| `candidate_D2_wheel_velocity_damping_light` | DO NOT MODIFY | Baseline profile |
| Default HY2-DIV state | DO NOT ENABLE | Baseline behavior |
| Default WBC state | DO NOT ENABLE | Structural invariant |
| Official Step E/C gates | DO NOT RELAX | Validation standard |
| Height variants setup files | DO NOT MODIFY | Validated baselines |

---

## 8. Files Created

- `outputs/height_range_extension_strategy_audit/height_range_extension_strategy.json`
- (this document)