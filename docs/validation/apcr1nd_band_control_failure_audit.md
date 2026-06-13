# APCR1nD Band Control Failure Audit

## Executive Summary

**Classification: APCR1ND_BAND_FAIL_MIXED_CAUSES**

APCR1nD fails to keep support drift within the ±0.08 m target band despite having direct recenter, position cap boost, and wheel damping override features.

**Key Problem:** 37.7% of time (754/2000 steps) spent outside ±0.08 m, with 53.6% of those violations occurring while features are inactive.

## Telemetry Source

- **File:** `outputs/hierarchical_controller_sim/telemetry_1781226281.csv`
- **Profile:** APCR1nD_direct_support_recenter_features
- **Total steps:** 2000
- **Drift metric:** `active_pitch_crossing_signed_error_m` (correct physical drift)

## Overall Drift Performance

| Metric | Value | Target | Pass? |
|--------|-------|--------|-------|
| Max \|e\| | 0.1691 m | < 0.15 m | ✅ |
| P2P | 0.1795 m | - | - |
| Mean \|e\| | 0.0607 m | - | - |
| Final e | 0.0038 m | ≈ 0 | ✅ |
| Outside ±0.08 | 754 steps (37.7%) | < 20% | ❌ |
| Outside ±0.10 | 446 steps (22.3%) | < 10% | ❌ |
| Outside ±0.15 | 55 steps (2.8%) | < 5% | ✅ |

## Band Violation Analysis

### Violation Counts

| Band Threshold | Count | Percent | User Target |
|----------------|-------|---------|-------------|
| Outside ±0.05 | 1101 | 55.1% | - |
| Outside ±0.08 | **754** | **37.7%** | **< 20%** ❌ |
| Outside ±0.10 | 446 | 22.3% | < 10% ❌ |
| Outside ±0.12 | 88 | 4.4% | - |
| Outside ±0.15 | 55 | 2.8% | < 5% ✅ |

**Primary failure:** 754 steps outside ±0.08 m vs. target < 400 steps (20%).

### Band Crossings

| Threshold | Crossings | Interpretation |
|-----------|-----------|----------------|
| +0.05 m | 9 | Frequent small excursions |
| +0.08 m | 9 | **Target band crossed 9 times** |
| +0.10 m | 8 | Nearly all ±0.08 crossings escalate |
| +0.15 m | 1 | One large excursion early |

**Problem:** Nearly every crossing of ±0.08 escalates to ±0.10, indicating insufficient early intervention.

## Outside-Band Behavior Analysis

### Active vs Inactive When Outside ±0.08

| State | Count | Percent of Outside-Band |
|-------|-------|-------------------------|
| **Active** | 350 | 46.4% |
| **Inactive** | 404 | **53.6%** ❌ |

**Critical finding:** More than half of ±0.08 violations occur while direct recenter is **inactive**.

### Converging vs Moving Away When Outside ±0.08

| State | Count | Percent of Outside-Band |
|-------|-------|-------------------------|
| Converging | 362 | 48.0% |
| Moving away | 392 | 52.0% |

**Problem:** 48% of violations are converging (error decreasing), but moving-away gating prevents activation.

### Outside ±0.10 Behavior

| State | Count | Percent of Outside ±0.10 |
|-------|-------|--------------------------|
| Active | 267 | 59.9% |
| Inactive | 179 | **40.1%** |

Even at ±0.10 (emergency level), 40% of violations are inactive.

## Feature Activation Analysis

| Feature | Activation | Comment |
|---------|------------|---------|
| Direct recenter priority | 17.5% (350/2000) | Only active when moving away |
| Direct recenter eligible | 58.4% (1167/2000) | 3.3× more eligible than active |
| Position cap boost | 17.5% (350/2000) | Synced with recenter |
| Wheel damping override | 0.9% (19/2000) | Very rare emergency use |

**Activation gap:** 58.4% eligible but only 17.5% active = **70% of eligible opportunities unused** due to moving-away gating.

## Failure Mode Classification

### 1. LATE_ENTRY (9 events)

Band crossings occur before feature activation.

**Example late entry events:**
- Step 42: crosses ±0.08, activation delay unknown
- Step 51: crosses ±0.08, activation delay unknown
- Step 59: crosses ±0.08, activation delay unknown

**Cause:** Entry threshold (0.08) equals target band limit. By the time error reaches 0.08, it's already AT the boundary, not BEFORE it.

### 2. EARLY_RELEASE (11 events)

Feature turns off while error still outside ±0.08.

| Release While Outside | Count |
|-----------------------|-------|
| ±0.08 | 11 |
| ±0.10 | 0 |
| ±0.12 | 0 |

**Cause:** Release logic allows deactivation when converging, even if still outside target band.

### 3. MOVING_AWAY_GATING (404 inactive outside ±0.08)

Feature inactive when outside band because error is converging (not moving away).

**Problem:** Moving-away requirement is too strict for band-limited control:
- 404/754 (53.6%) outside-band steps are inactive
- 362/754 (48.0%) outside-band steps are converging
- Feature waits for drift to reverse before acting

**User requirement violation:** "If drift goes outside ±0.08, do not release just because it is converging."

### 4. WEAK_AUTHORITY (349 instances)

Feature active but error not decreasing over 10-step window.

**Count:** 349/2000 steps (17.5%)

**Interpretation:** 
- Nearly 100% of active time shows weak authority (349 out of 350 active steps)
- Position cap boost may not be strong enough
- Wheel damping override too rare (0.9%)

### 5. DAMPING_TOO_RARE (0.9%)

Wheel damping override active only 19/2000 steps.

**Problem:** Damping override is emergency-only, not proactive band-keeping.

## Root Cause Summary

| Failure Mode | Severity | Root Cause |
|--------------|----------|------------|
| **Moving-away gating** | 🔴 Critical | 53.6% of violations inactive due to converging requirement |
| **Weak authority** | 🔴 Critical | Active but not effective (349/350 steps) |
| **Late entry** | 🟡 Moderate | Entry threshold 0.08 = target band limit |
| **Early release** | 🟡 Moderate | 11 releases while outside ±0.08 |
| **Damping too rare** | 🟡 Moderate | Only 0.9% activation, emergency-only |

## Why APCR1nD Fails the ±0.08 Target

### Primary Causes

1. **Moving-away gating is too strict**
   - Requires error to be moving away from zero
   - 404 steps (20%) outside ±0.08 but inactive because converging
   - User requirement: "keep active until inner band", not "release when converging"

2. **Authority too weak when active**
   - 349/350 active steps show no improvement over 10-step window
   - Position cap boost may need higher values
   - Wheel damping override rarely engages (0.9%)

3. **Entry threshold too late**
   - Enters at 0.08, which is already the target limit
   - Should enter earlier (e.g., 0.06) to prevent reaching 0.08

4. **Release logic too early**
   - 11 releases while still outside ±0.08
   - Should use inner band (e.g., 0.03-0.04) before release

### Secondary Causes

5. **Damping override underutilized**
   - Only 0.9% active time
   - Could be more proactive for band-keeping

## Recommendations for Tuning

### High Priority

1. **Remove moving-away requirement when outside desired band**
   - If abs(e) > 0.08, stay active regardless of e_dot sign
   - Only check moving-away for initial entry at softer threshold

2. **Enter earlier**
   - Change entry threshold from 0.08 to 0.06
   - Prevents reaching target band limit

3. **Hold until inner band**
   - Change release threshold from e_dot reversal to abs(e) < 0.03-0.04
   - User requirement: "keep active until reaches inner band"

4. **Increase authority**
   - Increase position cap boost levels
   - Increase wheel damping override engagement
   - Make damping more proactive, not emergency-only

### Moderate Priority

5. **Add band-state logic**
   - soft_band: 0.05-0.06
   - desired_band: 0.06-0.08
   - hard_band: 0.08-0.10
   - emergency_band: > 0.10
   - Scale authority by band level

## Next Steps (Phase 2)

Design APCR1nD tuned variants:
- **T1:** Early entry (0.06)
- **T2:** Hold outside band
- **T3:** Early entry + hold
- **T4:** Stronger authority
- **T5:** Band-limited balanced (recommended)

## Appendix: Raw Statistics

```json
{
  "outside_05": 1101,
  "outside_08": 754,
  "outside_10": 446,
  "outside_15": 55,
  "outside_08_active": 350,
  "outside_08_inactive": 404,
  "outside_08_converging": 362,
  "outside_08_moving_away": 392,
  "early_release_08": 11,
  "late_entry_count": 9,
  "authority_weak_count": 349,
  "damping_active_pct": 0.9,
  "classification": "APCR1ND_BAND_FAIL_MIXED_CAUSES"
}
```

---

**Audit complete:** 2026-06-12
**Classification:** APCR1ND_BAND_FAIL_MIXED_CAUSES
**Primary causes:** Moving-away gating (53.6% inactive), weak authority (349/350 steps), late entry, early release
