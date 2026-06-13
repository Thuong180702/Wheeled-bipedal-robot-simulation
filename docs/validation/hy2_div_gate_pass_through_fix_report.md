# HY2-DIV Gate Pass-Through Fix Report

## Bug Found

**Bug Type:** Integration pass-through bug

**Location:** `build_balance_core_controllers()` in `scripts/simulate_hierarchical_controller.py`

**Issue:** The `ShapePostureController` accepts `divergence_gate_z_low` and `divergence_gate_z_high` parameters, but `build_balance_core_controllers()` did NOT pass these parameters. This meant ALL HY2-DIV candidates used the default gate values (z_low=0.300, z_high=0.393) regardless of their designed values, causing A/B candidates to behave identically.

## Root Cause Analysis

1. `HipYawDivergenceProfile` dataclasses (A0-A3, B1-B3, C1) correctly defined `z_low` and `z_high` values
2. `ShapePostureController.__init__()` correctly accepted and stored `divergence_gate_z_low` and `divergence_gate_z_high`
3. **BUG:** `build_balance_core_controllers()` did NOT have parameters for `hip_yaw_divergence_z_low`/`z_high`
4. **BUG:** `build_balance_core_controllers()` did NOT pass these to `ShapePostureController()`
5. **BUG:** CLI arguments for `--hip-yaw-divergence-z-low` and `--hip-yaw-divergence-z-high` did not exist

## Files Changed

### 1. `scripts/simulate_hierarchical_controller.py`

**Added CLI arguments (after line 1507):**
```python
parser.add_argument(
    "--hip-yaw-divergence-z-low",
    type=float,
    default=0.300,
    help="Lower height threshold for HY2-DIV gate (m). Default: 0.300.",
)
parser.add_argument(
    "--hip-yaw-divergence-z-high",
    type=float,
    default=0.393,
    help="Upper height threshold for HY2-DIV gate (m). Default: 0.393.",
)
```

**Added function parameters (line ~947):**
```python
hip_yaw_divergence_z_low: float = 0.300,
hip_yaw_divergence_z_high: float = 0.393,
```

**Added ShapePostureController kwargs (line ~1000):**
```python
divergence_gate_z_low=hip_yaw_divergence_z_low,
divergence_gate_z_high=hip_yaw_divergence_z_high,
```

**Added args passed to build_balance_core_controllers() (line ~2532):**
```python
hip_yaw_divergence_z_low=args.hip_yaw_divergence_z_low,
hip_yaw_divergence_z_high=args.hip_yaw_divergence_z_high,
```

### 2. `wheeled_biped/controllers/shape_posture_controller.py`

**Added telemetry fields:**
```python
"hip_yaw_div_z_low": float(self.divergence_gate_z_low),
"hip_yaw_div_z_high": float(self.divergence_gate_z_high),
```

### 3. `scripts/evaluate_hy2_div_authority_candidates.py`

**Added z_low/z_high to simulation command:**
```python
"--hip-yaw-divergence-z-low", str(config["z_low"]),
"--hip-yaw-divergence-z-high", str(config["z_high"]),
```

### 4. `tests/test_hip_yaw_divergence_control.py`

**Added test for gate z parameters:**
```python
def test_hy2_div_gate_z_params_passed(self):
    """HY2-DIV z_low/z_high gate parameters should be stored correctly."""
    # Verifies divergence_gate_z_low and divergence_gate_z_high are stored
```

**Updated telemetry test:**
```python
expected_fields = [
    # ... existing fields ...
    "hip_yaw_div_z_low",
    "hip_yaw_div_z_high",
]
```

### 5. `scripts/debug_gate_pass_through.py` (NEW)

Created dedicated debug script to verify gate pass-through behavior.

## Tests Run and Results

### Unit Tests
```
tests/test_hip_yaw_divergence_control.py - 32 passed
tests/test_shape_posture_hip_yaw_sign.py - 9 passed
tests/test_step_e_hip_yaw_authority_fix.py - 5 passed
```

### Gate Pass-Through Verification (100 steps)

| Candidate | Height | z_high | gate_min | gate_max | gate_mean | Expected |
|-----------|--------|--------|----------|----------|-----------|----------|
| A0 | nominal | 0.393 | 0.000 | 0.000 | 0.000 | ~0.0 (inactive) |
| A0 | low_0p300 | 0.393 | 1.000 | 1.000 | 1.000 | ~1.0 (active) |
| A0 | high_0p480 | 0.393 | 0.000 | 0.000 | 0.000 | ~0.0 (inactive) |
| B1 | nominal | 0.500 | 0.500 | 0.500 | 0.500 | >0.0 (active) |
| B1 | low_0p300 | 0.500 | 1.000 | 1.000 | 1.000 | ~1.0 (active) |
| B1 | high_0p480 | 0.500 | 0.028 | 0.028 | 0.028 | >0.0 (active) |

**Verdict: PASS** - Candidates produce DIFFERENT gate values. Parameters ARE being passed correctly.

## Candidate Evaluation Results

### 100-Step Smoke (All passed)

All 7 candidates (A0-A3, B1-B3) passed 100-step smoke tests at all heights.

### 500-Step Validation

| Candidate | Height | Survived | Div RMS | HY2 Active | Clipped | vs A0 | vs Baseline |
|-----------|--------|----------|---------|------------|---------|-------|-------------|
| A0 | nominal | YES | 0.0230 | 100.0% | 0.0% | - | PASS |
| A0 | low_0p300 | YES | 0.0534 | 100.0% | 10.6% | - | PASS |
| A0 | high_0p480 | YES | 0.0133 | 100.0% | 0.0% | - | PASS |
| A1 | nominal | YES | 0.0230 | 100.0% | 0.0% | same | PASS |
| A1 | low_0p300 | YES | 0.0540 | 100.0% | 0.0% | +1.1% | PASS |
| A1 | high_0p480 | YES | 0.0133 | 100.0% | 0.0% | same | PASS |
| A2 | nominal | YES | 0.0230 | 100.0% | 0.0% | same | PASS |
| A2 | low_0p300 | YES | 0.0540 | 100.0% | 0.0% | +1.1% | PASS |
| A2 | high_0p480 | YES | 0.0133 | 100.0% | 0.0% | same | PASS |
| B1 | nominal | YES | 0.0242 | 100.0% | 0.0% | +5.1% | PASS |
| B1 | low_0p300 | YES | 0.0540 | 100.0% | 0.0% | +1.1% | PASS |
| B1 | high_0p480 | YES | 0.0133 | 100.0% | 0.0% | same | PASS |
| B2 | nominal | YES | 0.0242 | 100.0% | 0.0% | +5.1% | PASS |
| B2 | low_0p300 | YES | 0.0540 | 100.0% | 0.0% | +1.1% | PASS |
| B2 | high_0p480 | YES | 0.0133 | 100.0% | 0.0% | same | PASS |

## Key Findings

1. **Gate pass-through is FIXED** - A0 and B1 now produce clearly different gate values at nominal height.

2. **Clipping does NOT improve divergence** - A0 has 10.6% clipping at low_0p300. A1/A2 eliminate clipping by doubling/quadrupling tau_max, but divergence stays at 0.054 (essentially same). This means HY2-DIV is not limited by tau_max clipping at low heights.

3. **HY2-DIV authority is fundamentally insufficient at low heights** - The gate-based damping (even at full gate=1.0) is insufficient to suppress divergence at low_0p300.

4. **Nominal/high heights are well-controlled** - Divergence is very low (0.013-0.024) at nominal and high heights.

5. **Extended gate (B1/B2) does NOT improve nominal performance** - Despite B1 having gate=0.5 at nominal (vs 0.0 for A0), divergence is slightly WORSE (0.0242 vs 0.0230). This suggests that the additional HY2-DIV authority at nominal may be interacting negatively with other controller actions.

## Final Action

**RESUME_HY2_DIV_AUTHORITY_EVALUATION**

The gate pass-through bug is FIXED. The evaluation pipeline is now working correctly. However, the results show that:

1. A0 (baseline) remains the best candidate at nominal/high heights
2. No candidate significantly improves low_0p300 performance
3. Extended gate (B1/B2) provides nominal coverage but doesn't improve overall performance
4. Further evaluation (5000-step) should proceed with ALL candidates that passed 100-step smoke to gather comprehensive data before making final recommendations

## Next Steps

1. Run 5000-step Step E evaluation for candidates passing 100-step smoke (A0-A3, B1-B3)
2. Evaluate C1 candidate with strong damping parameters
3. Analyze long-term divergence behavior and stability
4. Make final recommendation based on comprehensive data
