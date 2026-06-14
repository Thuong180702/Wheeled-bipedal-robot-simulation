# APCR1nD Tuned Variants Design

## Design Goal

Achieve stricter band-limited support drift control: keep drift within ±0.08 m.

**Current APCR1nD performance:**
- Outside ±0.08: 754/2000 steps (37.7%) ❌
- Target: < 400/2000 steps (20%)

**Root causes identified:**
1. Moving-away gating too strict (53.6% inactive outside band)
2. Authority too weak (349/350 active steps ineffective)
3. Entry threshold too late (0.08 = target limit)
4. Release too early (11 events while outside ±0.08)
5. Damping override too rare (0.9%)

## Design Principles

### What to Keep from APCR1nD

✅ **Direct support recenter infrastructure**
✅ **Position cap boost mechanism**
✅ **Wheel damping override mechanism**
✅ **Safety gates (startup, contact, height, roll, pitch)**
✅ **Startup guard (first 100 steps)**

### What to Tune

🔧 **Entry thresholds** (soft/direct)
🔧 **Hold/release logic** (when to stay active)
🔧 **Moving-away requirement** (when required vs optional)
🔧 **Authority levels** (position cap, damping scale)
🔧 **Band-state awareness** (different responses by severity)

### What NOT to Change

🚫 D2 baseline
🚫 APCR1h profile
🚫 APCR1n profile
🚫 APCR1nD baseline (create new variants)
🚫 WBC path
🚫 HY2-DIV defaults
🚫 Safety gate thresholds

## Five Tuned Variants

### Variant T1: Early Entry

**Purpose:** Enter support recenter before reaching target band.

**Changes from APCR1nD:**
```yaml
direct_enter_m: 0.06  # was 0.08
soft_enter_m: 0.05    # was none
moving_away_required: true  # for initial entry
release_logic: same as APCR1nD
```

**Hypothesis:** Entering at 0.06 prevents escalation to 0.08.

**Risk:** May activate too frequently.

---

### Variant T2: Hold Outside Band

**Purpose:** Keep feature active while outside desired band, even if converging.

**Changes from APCR1nD:**
```yaml
direct_enter_m: 0.08  # unchanged
desired_band_m: 0.08  # new: band boundary
release_inner_m: 0.05  # new: must reach inner band before release

hold_logic:
  - if abs(e) > 0.08: stay active regardless of e_dot
  - if abs(e) <= 0.05: allow release
  - if 0.05 < abs(e) <= 0.08 and converging: hold

moving_away_required: true  # only for initial entry
```

**Hypothesis:** Holding while outside band prevents 404 inactive violations.

**Risk:** May stay active too long if convergence is slow.

---

### Variant T3: Early Entry + Hold

**Purpose:** Combine early entry and band hold.

**Changes from APCR1nD:**
```yaml
soft_enter_m: 0.05
direct_enter_m: 0.06
desired_band_m: 0.08
release_inner_m: 0.03  # stricter inner band

activation_logic:
  - if abs(e) >= 0.06 and moving_away: activate
  - if abs(e) >= 0.08: activate regardless of moving_away
  - if already active and abs(e) > 0.03: hold

release_logic:
  - release only when abs(e) <= 0.03
  - OR when abs(e) decreasing and < 0.05 for 20 consecutive steps
```

**Hypothesis:** Early entry + strict hold achieves best band control.

**Risk:** Most aggressive; may oscillate.

---

### Variant T4: Stronger Authority

**Purpose:** Increase torque authority when fighting drift.

**Changes from APCR1nD:**
```yaml
direct_enter_m: 0.06
release_inner_m: 0.03

position_cap:
  normal: 4.0       # was 3.5
  recenter: 6.0     # was 5.0
  emergency: 7.0    # was 6.0

wheel_damping_scale:
  normal: 1.0
  desired_band: 0.20  # more damping when fighting drift
  hard_band: 0.10     # emergency damping
  
damping_condition:
  - if abs(e) > 0.08 and fighting_drift: scale = 0.20
  - if abs(e) > 0.10 and fighting_drift: scale = 0.10
```

**Hypothesis:** Higher authority + more damping achieves faster return.

**Risk:** May increase wheel velocity; torque may become aggressive.

---

### Variant T5: Band-Limited Balanced (RECOMMENDED)

**Purpose:** Best practical tuned profile combining all improvements.

**Changes from APCR1nD:**
```yaml
# Thresholds
soft_enter_m: 0.05
direct_enter_m: 0.06
desired_band_m: 0.08
hard_band_m: 0.10
emergency_band_m: 0.12
release_inner_m: 0.03

# Activation logic
activate_if:
  - (abs(e) >= 0.06 and moving_away)
  - OR (abs(e) >= 0.08 regardless of moving_away)
  - OR (already_active and abs(e) > 0.03)

# Release logic
release_if:
  - abs(e) <= 0.03
  - AND (converging for 15 consecutive steps OR e_dot sign stable)

# Position cap by band
position_cap:
  normal: 4.0           # abs(e) < 0.05
  soft: 4.5             # 0.05 <= abs(e) < 0.06
  desired: 5.5          # 0.06 <= abs(e) < 0.08
  hard: 6.5             # 0.08 <= abs(e) < 0.10
  emergency: 7.0        # abs(e) >= 0.10

# Wheel damping scale by band
wheel_damping_scale:
  normal: 1.0           # abs(e) < 0.05
  soft: 0.50            # 0.05 <= abs(e) < 0.06
  desired: 0.30         # 0.06 <= abs(e) < 0.08
  hard: 0.15            # 0.08 <= abs(e) < 0.10
  emergency: 0.10       # abs(e) >= 0.10

# Damping preservation
preserve_damping_if_helps_recovery: true
  # Don't reduce damping if it's helping return to zero

# Safety gates (unchanged)
startup_guard: first 100 steps
contact_unsafe: blocks features
height_unsafe: blocks features
roll_unsafe: blocks features
pitch_hard_unsafe: blocks features
```

**Hypothesis:** Graduated response by band level + early entry + hold achieves ±0.08 target.

**Advantages:**
- Early entry prevents reaching 0.08
- Hold logic keeps active outside band
- Graduated authority scales with severity
- Proactive damping, not emergency-only
- Preserves successful damping

**Risk:** Most complex; requires careful testing.

---

## Implementation Strategy

### Reuse Existing APCR1nD Infrastructure

All variants reuse:
- `compute_direct_recenter_priority()` base logic
- `position_cap_boost` mechanism
- `wheel_damping_override` mechanism
- Safety gate validation
- Telemetry plumbing

### New Configuration Fields

Add to sagittal schedule config:
```python
@dataclass
class APCR1nDTunedConfig:
    variant_name: str  # "T1", "T2", "T3", "T4", "T5"
    
    # Thresholds
    soft_enter_m: float = 0.05
    direct_enter_m: float = 0.06
    desired_band_m: float = 0.08
    hard_band_m: float = 0.10
    emergency_band_m: float = 0.12
    release_inner_m: float = 0.03
    
    # Hold logic
    hold_outside_band: bool = True
    moving_away_required_when_outside_band: bool = False
    
    # Authority
    position_cap_by_band: Dict[str, float]
    wheel_damping_scale_by_band: Dict[str, float]
    
    # Damping preservation
    preserve_damping_if_helps: bool = True
```

### Telemetry Extensions

Add to telemetry:
```python
tuned_variant_name: str
tuned_recenter_active: bool
tuned_band_state: str  # "normal", "soft", "desired", "hard", "emergency"
tuned_abs_error: float
tuned_release_allowed: bool
tuned_active_reason: str
tuned_block_reason: str
tuned_position_cap_current: float
tuned_wheel_damping_scale: float
tuned_outside_band_active: bool
tuned_outside_band_inactive: bool
```

## Variant Comparison Matrix

| Feature | APCR1nD | T1 | T2 | T3 | T4 | T5 |
|---------|---------|----|----|----|----|-----|
| Entry threshold | 0.08 | 0.06 | 0.08 | 0.06 | 0.06 | 0.06 |
| Hold outside band | No | No | Yes | Yes | No | Yes |
| Release threshold | e_dot | e_dot | 0.05 | 0.03 | 0.03 | 0.03 |
| Moving-away when outside | Yes | Yes | No | No | Yes | No |
| Position cap max | 6.0 | 6.0 | 6.0 | 6.0 | 7.0 | 7.0 |
| Damping scale min | 1.0 | 1.0 | 1.0 | 1.0 | 0.10 | 0.10 |
| Band-aware authority | No | No | No | No | Yes | Yes |
| Complexity | Base | Low | Moderate | Moderate | Moderate | High |

## Expected Performance

| Metric | APCR1nD | T1 | T2 | T3 | T4 | T5 |
|--------|---------|----|----|----|----|-----|
| Outside ±0.08 % | 37.7 | 30-35 | 25-30 | 20-25 | 30-35 | **15-20** |
| Outside ±0.10 % | 22.3 | 18-22 | 15-20 | 12-18 | 18-22 | **10-15** |
| Max \|e\| | 0.169 | 0.15-0.17 | 0.14-0.16 | 0.13-0.15 | 0.14-0.16 | **0.12-0.14** |
| Active % | 17.5 | 25-30 | 30-35 | 35-40 | 25-30 | **30-35** |
| Wheel velocity | OK | OK | OK? | ? | Higher | Higher |

## Success Criteria

### Primary

1. **Outside ±0.08 < 20%** (< 400/2000 steps)
2. **Survives 2000 steps**
3. **Contact/height/roll stable**

### Secondary

4. Outside ±0.10 < 10% (< 200/2000 steps)
5. Max |e| < 0.15 m
6. P2P not worse than APCR1nD (< 0.18 m)
7. Wheel velocity acceptable

### Failure Modes to Avoid

❌ Too aggressive: oscillation, instability
❌ Too conservative: no improvement over APCR1nD
❌ Feature always active: no selectivity
❌ Wheel velocity unsafe

## Recommendation

**Start with T5 (Band-Limited Balanced)** as the most comprehensive solution, but test all five to understand which mechanisms are most effective.

**Testing order:**
1. T5 (expected best)
2. T3 (early + hold)
3. T2 (hold only)
4. T4 (stronger authority)
5. T1 (early entry only)

---

**Design complete:** 2026-06-12
**Next:** Phase 3 - Implementation
