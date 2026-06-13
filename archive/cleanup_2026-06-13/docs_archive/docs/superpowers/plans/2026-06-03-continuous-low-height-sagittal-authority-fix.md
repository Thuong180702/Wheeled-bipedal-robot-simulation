# Continuous Low-Height Sagittal Authority Fix - Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a continuous formula-based k_position schedule for the SagittalVelocityDampedBalanceController that smoothly increases sagittal position authority at extreme low heights (0.300m CoM), without using variant-name-based gain patches.

**Architecture:** Add height-scheduled k_position to the existing SagittalVelocityDampedBalanceController using smoothstep interpolation. The schedule is formula-based (not variant-specific), uses commanded/reference height as input, and integrates with existing telemetry.

**Tech Stack:** Python, JAX, MuJoCo, pytest

---

## File Map

| File                                                                       | Responsibility                                                                  |
| -------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py` | Add smoothstep formula, height-scheduled k_position params, compute integration |
| `wheeled_biped/controllers/balance_core_types.py`                          | Add new telemetry fields to diagnostics dict                                    |
| `scripts/simulate_hierarchical_controller.py`                              | Add new E1/E2/E3 continuous profiles, pass commanded_height_ref_m               |
| `scripts/run_step_c_height_recovery.py`                                    | Pass commanded_height_ref_m to controller                                       |
| `scripts/evaluate_continuous_low_height_sagittal_authority_fix.py`         | New evaluation script for candidates                                            |
| `scripts/check_schedule_continuity.py`                                     | Continuity check using 181-point dense sweep + clamp verification               |
| `tests/test_sagittal_velocity_damped_balance_controller.py`                | Add tests for smoothstep and scheduled_k_position                               |

---

## Candidate Profiles

| Profile                      | k_nominal | k_low_max | z_low | z_high |
| ---------------------------- | --------- | --------- | ----- | ------ |
| candidate_E1_k60_continuous  | 40.0      | 60.0      | 0.300 | 0.393  |
| candidate_E2_k80_continuous  | 40.0      | 80.0      | 0.300 | 0.393  |
| candidate_E3_k100_continuous | 40.0      | 100.0     | 0.300 | 0.393  |

---

## Acceptance Criteria (required for all candidates)

A candidate only passes if **all** of the following are true:

| Metric                         | Threshold   |
| ------------------------------ | ----------- |
| support_position_error max_abs | <= 0.15 m   |
| hip_yaw_abs_max                | <= 0.07 rad |
| pitch_x max_abs                | <= 0.10 rad |
| roll_y max_abs                 | <= 0.05 rad |
| final height error max_abs     | <= 0.02 m   |
| non-wheel floor contacts       | = 0         |
| contact valid                  | >= 99.9%    |
| WBC applied                    | = false     |
| hidden torque                  | = 0         |
| ownership violations           | = 0         |

**Hip-yaw secondary root cause rule:** If support drift is fixed but `hip_yaw_abs_max > 0.07 rad`:

- Stop evaluation of this candidate
- Classify as secondary hip-yaw root cause
- Do not keep increasing k_position blindly
- Do not claim BOUNDARY_RANGE_PASS

---

## Final Decision Logic

Final decision can be one of:

- **BOUNDARY_RANGE_PASS**: Only if Step E 0.300/0.480 pass, Step C 0.300/0.480 pass, practical grid passes, five-variant regression passes, schedule continuity passes, WBC off, hidden torque 0, ownership 0.

- **LOW_HEIGHT_SAGITTAL_FIX_REQUIRED**: If no continuous k_position candidate fixes low_0p300.

- **FIX_CAUSED_REGRESSION**: If boundary improves but nominal or existing five variants regress.

- **NEW_ROOT_CAUSE_FOUND**: If support drift is fixed but hip-yaw/pitch/wheel/contact becomes primary.

---

## No-Patchwork Rule

The plan must state:

- Actual gain value must be computed only as k_position = f(z_ref)
- No variant-name-only gain patch
- No discrete bucket jump
- No `if low_0p300 then k_position` logic
- No global hip-yaw gain change
- No WBC
- No threshold relaxation

---

## Commanded/Reference Height Source Rule

- Step E / Step C boundary setup uses `target_com_z_m` from setup JSON if available
- Otherwise uses `achieved_com_z_m`
- `schedule_height_source` must be logged
- Raw instantaneous com_z must not be used as the main scheduling input
- Filtered current com_z is diagnostic/fallback only

---

## Task 1: Add smoothstep and scheduled_k_position functions + controller params

**Files:**

- Modify: `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py:1-55`

**Purpose:** Add the core formula functions at module level, then extend SagittalAuthoritySchedule with continuous k_position schedule parameters.

- [ ] **Step 1: Write the failing test**

```python
# Add to tests/test_sagittal_velocity_damped_balance_controller.py

def test_smoothstep01_boundary_values():
    """smoothstep01(0) = 0 and smoothstep01(1) = 1."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import smoothstep01
    assert abs(smoothstep01(0.0) - 0.0) < 1e-9
    assert abs(smoothstep01(1.0) - 1.0) < 1e-9

def test_smoothstep01_interpolation():
    """smoothstep is smooth (not linear) interpolation."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import smoothstep01
    # Midpoint should be 0.5 (by symmetry of smoothstep)
    val = smoothstep01(0.5)
    assert 0.4 < val < 0.6  # within smooth region

def test_scheduled_k_position_at_boundaries():
    """scheduled_k_position at z_low returns k_low_max; at z_high returns k_nominal."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import scheduled_k_position
    k_nominal = 40.0
    k_low_max = 80.0
    z_low = 0.300
    z_high = 0.393

    k_at_low = scheduled_k_position(z_low, k_nominal, k_low_max, z_low, z_high)
    k_at_high = scheduled_k_position(z_high, k_nominal, k_low_max, z_low, z_high)

    assert abs(k_at_low - k_low_max) < 1e-6, f"k_at_low={k_at_low}, expected {k_low_max}"
    assert abs(k_at_high - k_nominal) < 1e-6, f"k_at_high={k_at_high}, expected {k_nominal}"

def test_scheduled_k_position_outside_range():
    """scheduled_k_position clamps outside [z_low, z_high]."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import scheduled_k_position
    k_nominal = 40.0
    k_low_max = 80.0
    z_low = 0.300
    z_high = 0.393

    # Above z_high: returns k_nominal
    k_above = scheduled_k_position(0.480, k_nominal, k_low_max, z_low, z_high)
    assert abs(k_above - k_nominal) < 1e-6
    # Below z_low: returns k_low_max
    k_below = scheduled_k_position(0.280, k_nominal, k_low_max, z_low, z_high)
    assert abs(k_below - k_low_max) < 1e-6

def test_scheduled_k_position_monotonic_decrease():
    """scheduled_k_position decreases monotonically from z_low to z_high."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import scheduled_k_position
    k_nominal = 40.0
    k_low_max = 80.0
    z_low = 0.300
    z_high = 0.393

    prev_k = None
    for z in jnp.linspace(z_low, z_high, 50):
        z_val = float(z)
        k = scheduled_k_position(z_val, k_nominal, k_low_max, z_low, z_high)
        if prev_k is not None:
            assert k <= prev_k + 1e-9, f"Non-monotonic at z={z_val}: k={k} > prev_k={prev_k}"
        prev_k = k
```

Run: `pytest tests/test_sagittal_velocity_damped_balance_controller.py::test_smoothstep01_boundary_values -v`
Expected: FAIL with "import error: smoothstep01 not defined"

- [ ] **Step 2: Add smoothstep01 and scheduled_k_position at module level**

In `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py`, add after the imports (before the SagittalAuthoritySchedule dataclass):

```python
def smoothstep01(u: float) -> float:
    """Standard smoothstep interpolation: s(0)=0, s(1)=1, s'(0)=s'(1)=0."""
    u = max(0.0, min(1.0, u))
    return u * u * (3.0 - 2.0 * u)


def scheduled_k_position(
    z_ref: float,
    k_nominal: float,
    k_low_max: float,
    z_low: float,
    z_high: float,
) -> float:
    """Compute k_position as a smooth function of height.

    Uses smoothstep interpolation between k_nominal and k_low_max.

    Behavior:
        - z_ref = z_low  -> u = 1 -> smoothstep = 1 -> k_position = k_low_max
        - z_ref = z_high -> u = 0 -> smoothstep = 0 -> k_position = k_nominal
        - z_ref > z_high -> u clamped to 0 -> k_position = k_nominal
        - z_ref < z_low  -> u clamped to 1 -> k_position = k_low_max

    Args:
        z_ref: Commanded/reference height (m)
        k_nominal: k_position at nominal/high heights
        k_low_max: k_position at lowest heights
        z_low: Lower boundary where max authority applies
        z_high: Upper boundary where nominal authority applies

    Returns:
        Smoothly interpolated k_position value
    """
    u = (z_high - z_ref) / (z_high - z_low)
    s = smoothstep01(u)
    return k_nominal + (k_low_max - k_nominal) * s
```

Run: `pytest tests/test_sagittal_velocity_damped_balance_controller.py::test_smoothstep01_boundary_values tests/test_sagittal_velocity_damped_balance_controller.py::test_scheduled_k_position_at_boundaries tests/test_sagittal_velocity_damped_balance_controller.py::test_scheduled_k_position_outside_range tests/test_sagittal_velocity_damped_balance_controller.py::test_scheduled_k_position_monotonic_decrease -v`
Expected: PASS for all 4 tests

---

## Task 2: Extend SagittalAuthoritySchedule with continuous k_position fields

**Files:**

- Modify: `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py:30-51`

**Purpose:** Add boolean flag and parameters for continuous k_position scheduling to SagittalAuthoritySchedule.

- [ ] **Step 1: Write the failing test**

```python
def test_sagittal_authority_schedule_has_continuous_k_position_fields():
    """SagittalAuthoritySchedule has fields for continuous k_position scheduling."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import SagittalAuthoritySchedule
    sched = SagittalAuthoritySchedule(
        profile_name="test_continuous",
        continuous_k_position=True,
        k_position_nominal=40.0,
        k_position_low_max=80.0,
        k_position_z_low=0.300,
        k_position_z_high=0.393,
    )
    assert sched.continuous_k_position == True
    assert sched.k_position_nominal == 40.0
    assert sched.k_position_low_max == 80.0
    assert sched.k_position_z_low == 0.300
    assert sched.k_position_z_high == 0.393
```

Run: `pytest tests/test_sagittal_velocity_damped_balance_controller.py::test_sagittal_authority_schedule_has_continuous_k_position_fields -v`
Expected: FAIL - "SagittalAuthoritySchedule() got an unexpected keyword argument 'continuous_k_position'"

- [ ] **Step 2: Add continuous k_position fields to SagittalAuthoritySchedule dataclass**

Find the SagittalAuthoritySchedule dataclass in `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py` (around lines 30-51) and add these fields:

```python
@dataclass(frozen=True)
class SagittalAuthoritySchedule:
    profile_name: str = "baseline"
    applies_to_variants: tuple[str, ...] = ()
    position_tau_cap_scale: float = 1.0
    position_tau_cap_by_variant: tuple[tuple[str, float], ...] = ()
    pitch_tau_scale: float = 1.0
    pitch_tau_cap_nm: float | None = None
    velocity_damping_scale: float = 1.0
    support_velocity_scale: float = 1.0
    support_velocity_gain: float | None = None
    # Continuous k_position scheduling fields
    continuous_k_position: bool = False
    k_position_nominal: float = 40.0
    k_position_low_max: float = 80.0
    k_position_z_low: float = 0.300
    k_position_z_high: float = 0.393

    def is_active_for_variant(self, variant_name: str | None) -> bool:
        return variant_name is not None and variant_name in self.applies_to_variants
```

Run: `pytest tests/test_sagittal_velocity_damped_balance_controller.py::test_sagittal_authority_schedule_has_continuous_k_position_fields -v`
Expected: PASS

---

## Task 3: Integrate continuous k_position into SagittalVelocityDampedBalanceController.compute()

**Files:**

- Modify: `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py` (SagittalVelocityDampedBalanceController class)

**Purpose:** Add store for filtered current com_z, read commanded_height_ref_m in compute(), compute effective k_position from schedule, add telemetry.

- [ ] **Step 1: Write the failing test**

```python
def test_controller_has_continuous_k_position_scheduling_in_compute():
    """Controller compute() accepts commanded_height_ref_m and uses it for k_position scheduling."""
    sched = SagittalAuthoritySchedule(
        profile_name="test_e2",
        continuous_k_position=True,
        k_position_nominal=40.0,
        k_position_low_max=80.0,
        k_position_z_low=0.300,
        k_position_z_high=0.393,
    )
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=1.0,
        k_position=40.0,
        max_tau_wheel=5.0,
        authority_schedule=sched,
    )
    # At 0.300m, k_position should approach k_low_max=80
    tau, diag = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.01,
        com_z_m=0.300,
        commanded_height_ref_m=0.300,
    )
    assert "effective_k_position" in diag
    assert diag["effective_k_position"] > 75.0, f"Expected k_position > 75 at 0.300m, got {diag['effective_k_position']}"
    assert diag["schedule_height_source"] == "target_reference"
    assert "low_height_sagittal_schedule_active" in diag


def test_controller_active_flag_at_high_height():
    """Active flag is False when smoothstep is effectively zero (high height)."""
    sched = SagittalAuthoritySchedule(
        profile_name="test_high",
        continuous_k_position=True,
        k_position_nominal=40.0,
        k_position_low_max=80.0,
        k_position_z_low=0.300,
        k_position_z_high=0.393,
    )
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=1.0,
        k_position=40.0,
        max_tau_wheel=5.0,
        authority_schedule=sched,
    )
    tau, diag = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.0,
        com_z_m=0.480,
        commanded_height_ref_m=0.480,
    )
    # Active flag should be False at high height (smoothstep ~ 0)
    active_key = "low_height_sagittal_schedule_active"
    if active_key in diag:
        assert diag[active_key] == False, f"Active flag should be False at 0.480m, got {diag[active_key]}"
```

Run: `pytest tests/test_sagittal_velocity_damped_balance_controller.py::test_controller_has_continuous_k_position_scheduling_in_compute tests/test_sagittal_velocity_damped_balance_controller.py::test_controller_active_flag_at_high_height -v`
Expected: FAIL - "compute() got an unexpected keyword argument 'commanded_height_ref_m'" and "effective_k_position not in diag"

- [ ] **Step 2: Add filtered com_z state variable to **init****

Find `SagittalVelocityDampedBalanceController.__init__()` in `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py`. Add after the existing state variable initializations:

```python
# State for support position velocity computation
self.prev_support_position_error_m = 0.0
# State for continuous k_position scheduling: first-order filtered com_z
self._filtered_com_z = 0.4  # Initialize to default com_z
```

- [ ] **Step 3: Add commanded_height_ref_m parameter to compute()**

Find the `compute()` method signature (around line 148-163). Add `commanded_height_ref_m: float | None = None` to the parameters.

- [ ] **Step 4: Add scheduling logic after state updates (around line 185-199)**

After the existing schedule_active computation (after line 199 in the existing code), add:

```python
# ---- Continuous height-scheduled k_position ----
# Determine scheduling height source
if commanded_height_ref_m is not None:
    schedule_height_ref = commanded_height_ref_m
    schedule_height_source = "target_reference"
else:
    # Fallback: use first-order filtered current com_z
    alpha_filter = 0.9  # Slow filter to avoid gain oscillation
    self._filtered_com_z = alpha_filter * self._filtered_com_z + (1.0 - alpha_filter) * float(com_z_m)
    schedule_height_ref = self._filtered_com_z
    schedule_height_source = "filtered_current_fallback"

effective_k_position = self.k_position
if self.authority_schedule.continuous_k_position:
    effective_k_position = scheduled_k_position(
        z_ref=schedule_height_ref,
        k_nominal=self.authority_schedule.k_position_nominal,
        k_low_max=self.authority_schedule.k_position_low_max,
        z_low=self.authority_schedule.k_position_z_low,
        z_high=self.authority_schedule.k_position_z_high,
    )
    # Compute smoothstep variables for telemetry and active flag
    u_raw = (self.authority_schedule.k_position_z_high - schedule_height_ref) / (
        self.authority_schedule.k_position_z_high - self.authority_schedule.k_position_z_low
    )
    u_clamped = max(0.0, min(1.0, u_raw))
    smoothstep_value = u_clamped * u_clamped * (3.0 - 2.0 * u_clamped)

    u_for_telemetry = u_clamped  # normalized position [0,1]
    schedule_smoothstep = smoothstep_value
else:
    u_for_telemetry = 0.0
    smoothstep_value = 0.0
    schedule_smoothstep = 0.0

SMALL_EPSILON = 1e-6
low_height_sagittal_schedule_active = (
    self.authority_schedule.continuous_k_position and smoothstep_value > SMALL_EPSILON
)
```

- [ ] **Step 5: Replace k_position usage with effective_k_position in position term computation**

Find the line that computes `tau_position_p` (around line 233):

```python
tau_position_p = -self.k_position * sagittal_position_error_m
```

Replace with:

```python
tau_position_p = -effective_k_position * sagittal_position_error_m
```

- [ ] **Step 6: Add all required telemetry fields to diagnostics dict**

Add these fields to the diagnostics dictionary return (around line 395-465):

```python
"schedule_height_source": schedule_height_source,
"schedule_height_reference_m": float(schedule_height_ref),
"filtered_current_com_z_m": float(self._filtered_com_z),
"effective_k_position": float(effective_k_position),
"k_position_schedule_u": float(u_for_telemetry),
"k_position_schedule_smoothstep": float(schedule_smoothstep),
"low_height_sagittal_schedule_active": bool(low_height_sagittal_schedule_active),
"k_position_nominal": float(self.authority_schedule.k_position_nominal),
"k_position_low_max": float(self.authority_schedule.k_position_low_max),
"k_position_z_low": float(self.authority_schedule.k_position_z_low),
"k_position_z_high": float(self.authority_schedule.k_position_z_high),
```

Run: `pytest tests/test_sagittal_velocity_damped_balance_controller.py::test_controller_has_continuous_k_position_scheduling_in_compute tests/test_sagittal_velocity_damped_balance_controller.py::test_controller_active_flag_at_high_height -v`
Expected: PASS for both

---

## Task 4: Add continuous E1/E2/E3 profiles to simulate_hierarchical_controller.py

**Files:**

- Modify: `scripts/simulate_hierarchical_controller.py` (SAGITTAL_AUTHORITY_PROFILES dict)

**Purpose:** Add three new continuous k_position profiles for evaluation.

- [ ] **Step 1: Find the SAGITTAL_AUTHORITY_PROFILES dict location**

The profiles are in `scripts/simulate_hierarchical_controller.py` starting around line 98. Add three new entries after the existing profiles.

Add these entries after the SAGITTAL_AUTHORITY_PROFILES entries:

```python
"candidate_E1_k60_continuous": SagittalAuthoritySchedule(
    profile_name="candidate_E1_k60_continuous",
    applies_to_variants=(),  # Not variant-specific - formula-based
    continuous_k_position=True,
    k_position_nominal=40.0,
    k_position_low_max=60.0,
    k_position_z_low=0.300,
    k_position_z_high=0.393,
),
"candidate_E2_k80_continuous": SagittalAuthoritySchedule(
    profile_name="candidate_E2_k80_continuous",
    applies_to_variants=(),  # Not variant-specific - formula-based
    continuous_k_position=True,
    k_position_nominal=40.0,
    k_position_low_max=80.0,
    k_position_z_low=0.300,
    k_position_z_high=0.393,
),
"candidate_E3_k100_continuous": SagittalAuthoritySchedule(
    profile_name="candidate_E3_k100_continuous",
    applies_to_variants=(),  # Not variant-specific - formula-based
    continuous_k_position=True,
    k_position_nominal=40.0,
    k_position_low_max=100.0,
    k_position_z_low=0.300,
    k_position_z_high=0.393,
),
```

Add these profile names to the `--vd-sagittal-authority-profile` choices list.

- [ ] **Step 2: Pass commanded_height_ref_m to controller compute()**

In the main loop where `sagittal_wheel_balance.compute()` is called (find this by searching for "sagittal_wheel_balance.compute"), add `commanded_height_ref_m` parameter.

Commanded height source (per Commanded/Reference Height Source Rule):

- `height_variant_setup["target_com_z_m"]` if available, else
- `height_variant_setup["achieved_com_z_m"]`

Pass this as `commanded_height_ref_m=commanded_height_m` to the compute() call.

Run: `python -m py_compile scripts/simulate_hierarchical_controller.py`
Expected: PASS (no syntax errors)

---

## Task 5: Create schedule continuity check script

**Files:**

- Create: `scripts/check_schedule_continuity.py`

**Purpose:** Generate a CSV proving the schedule is continuous from 0.300 to 0.480m using a 181-point dense sweep, plus explicit clamp checks at 0.280 and 0.500.

- [ ] **Step 1: Create the continuity check script**

The script uses **181 dense points** from 0.300 to 0.480 AND **2 clamp check points** at 0.280 and 0.500. All rows are written to the same CSV with a `sample_type` column distinguishing them.

The CSV has these columns:

| Column                         | Description                                  |
| ------------------------------ | -------------------------------------------- |
| candidate                      | Profile name (E1_k60, E2_k80, E3_k100)       |
| sample_type                    | "dense" or "clamp_check"                     |
| z_ref_m                        | Reference height in meters                   |
| effective_k_position           | Computed k_position value                    |
| delta_k_position_per_step      | Change from previous dense point             |
| k_position_schedule_u          | Normalized position [0,1] in transition band |
| k_position_schedule_smoothstep | Smoothstep value                             |
| schedule_active                | Boolean (True if smoothstep > 1e-6)          |

The script must report:

- `max_abs_delta_k_position` across consecutive pairs in the **dense** sweep
- `no_discontinuity` boolean
- `monotonic_decrease_low_to_high` boolean (k_position decreases from z_low to z_high in dense rows)
- `constant_k_nominal_above_z_high` boolean (from dense rows with z_ref > z_high)
- `constant_k_low_max_below_z_low` boolean — from **clamp check row** at z=0.280 (NOT from dense rows, since dense sweep starts at 0.300)
- `clamp_check_verified` boolean — both `k(0.280) == k_low_max` AND `k(0.500) == k_nominal`
- `k_at_0.280` and `k_at_0.500` printed explicitly

**The report must not claim below-z_low clamp was checked unless the clamp check rows are actually evaluated.** Since the dense sweep starts at 0.300 (z_low), there are no rows with z_ref < z_low in the dense portion — the clamp check at 0.280 is the ONLY source of evidence for below-z_low behavior.

```python
"""Schedule continuity check for continuous low-height sagittal authority fix.

Generates schedule_continuity_check.csv proving the smoothstep k_position
schedule is continuous from 0.300 to 0.480m using a 181-point dense sweep,
plus explicit clamp check rows at 0.280 and 0.500.

Usage:
    python scripts/check_schedule_continuity.py
"""

import csv
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
    smoothstep01,
    scheduled_k_position,
)

OUTPUT_DIR = Path("outputs/continuous_low_height_sagittal_authority_fix")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATES = {
    "candidate_E1_k60": {"k_nominal": 40.0, "k_low_max": 60.0},
    "candidate_E2_k80": {"k_nominal": 40.0, "k_low_max": 80.0},
    "candidate_E3_k100": {"k_nominal": 40.0, "k_low_max": 100.0},
}

Z_LOW = 0.300
Z_HIGH = 0.393
# 181-point dense sweep from 0.300 to 0.480
HEIGHT_DENSE = list(np.linspace(0.300, 0.480, 181).tolist())
# Clamp check points — evaluate outside the dense range
# z=0.280 is BELOW z_low, z=0.500 is ABOVE z_high
CLAMP_CHECK_HEIGHTS = [0.280, 0.500]


def main():
    output_path = OUTPUT_DIR / "schedule_continuity_check.csv"

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "candidate", "sample_type", "z_ref_m", "effective_k_position",
            "delta_k_position_per_step", "k_position_schedule_u",
            "k_position_schedule_smoothstep", "schedule_active"
        ])

        for cand_name, params in CANDIDATES.items():
            prev_k = None

            # --- Dense sweep rows (181 points from 0.300 to 0.480) ---
            for z in HEIGHT_DENSE:
                k_pos = scheduled_k_position(
                    z, params["k_nominal"], params["k_low_max"], Z_LOW, Z_HIGH
                )
                u_raw = (Z_HIGH - z) / (Z_HIGH - Z_LOW)
                u = max(0.0, min(1.0, u_raw))
                s = smoothstep01(u)
                delta_k = k_pos - prev_k if prev_k is not None else 0.0
                schedule_active = s > 1e-6

                writer.writerow([
                    cand_name, "dense", f"{z:.6f}", f"{k_pos:.8f}",
                    f"{delta_k:.10f}", f"{u:.8f}",
                    f"{s:.8f}", schedule_active
                ])
                prev_k = k_pos

            # --- Clamp check rows (evaluate at z < z_low and z > z_high) ---
            for z_clamp in CLAMP_CHECK_HEIGHTS:
                k_pos = scheduled_k_position(
                    z_clamp, params["k_nominal"], params["k_low_max"], Z_LOW, Z_HIGH
                )
                u_raw = (Z_HIGH - z_clamp) / (Z_HIGH - Z_LOW)
                u = max(0.0, min(1.0, u_raw))
                s = smoothstep01(u)
                schedule_active = s > 1e-6
                writer.writerow([
                    cand_name, "clamp_check", f"{z_clamp:.6f}", f"{k_pos:.8f}",
                    "0.0000000000", f"{u:.8f}",
                    f"{s:.8f}", schedule_active
                ])

    print(f"Continuity check written to {output_path}")

    # Analyze continuity from CSV
    with open(output_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    results = {}
    for cand_name, params in CANDIDATES.items():
        cand_rows = [r for r in rows if r["candidate"] == cand_name]
        dense_rows = [r for r in cand_rows if r["sample_type"] == "dense"]
        clamp_rows = [r for r in cand_rows if r["sample_type"] == "clamp_check"]

        # Delta across consecutive DENSE rows only
        deltas = [abs(float(r["delta_k_position_per_step"])) for r in dense_rows[1:]]
        max_abs_delta = max(deltas) if deltas else 0.0

        # Check monotonic decrease in transition band [z_low, z_high] in dense rows
        in_band = [r for r in dense_rows if Z_LOW <= float(r["z_ref_m"]) <= Z_HIGH]
        k_in_band = [float(r["effective_k_position"]) for r in in_band]
        monotonic = all(k_in_band[i] >= k_in_band[i+1] for i in range(len(k_in_band)-1))

        # Check constant above z_high in dense rows
        above_high = [r for r in dense_rows if float(r["z_ref_m"]) > Z_HIGH]
        k_above_high = [float(r["effective_k_position"]) for r in above_high]
        constant_above = all(abs(k - params["k_nominal"]) < 1e-6 for k in k_above_high) if k_above_high else True

        # Check clamp at z=0.280 (z < z_low) from clamp_check row
        z_0280_row = next((r for r in clamp_rows if float(r["z_ref_m"]) == 0.280), None)
        k_at_0280 = float(z_0280_row["effective_k_position"]) if z_0280_row else None
        clamp_below_ok = (
            k_at_0280 is not None and abs(k_at_0280 - params["k_low_max"]) < 1e-6
        ) if k_at_0280 is not None else False

        # Check clamp at z=0.500 (z > z_high) from clamp_check row
        z_0500_row = next((r for r in clamp_rows if float(r["z_ref_m"]) == 0.500), None)
        k_at_0500 = float(z_0500_row["effective_k_position"]) if z_0500_row else None
        clamp_above_ok = (
            k_at_0500 is not None and abs(k_at_0500 - params["k_nominal"]) < 1e-6
        ) if k_at_0500 is not None else False

        clamp_check_verified = clamp_below_ok and clamp_above_ok

        results[cand_name] = {
            "max_abs_delta_k_position": max_abs_delta,
            "no_discontinuity": max_abs_delta < 0.5,
            "monotonic_decrease_low_to_high": monotonic,
            "constant_k_nominal_above_z_high": constant_above,
            "constant_k_low_max_below_z_low": clamp_below_ok,
            "clamp_check_verified": clamp_check_verified,
            "k_at_0.280": k_at_0280,
            "k_at_0.500": k_at_0500,
        }

        print(f"\n{cand_name}:")
        print(f"  max_abs_delta_k_position = {max_abs_delta:.8f}")
        print(f"  no_discontinuity = {results[cand_name]['no_discontinuity']}")
        print(f"  monotonic_decrease_low_to_high = {monotonic}")
        print(f"  constant_k_nominal_above_z_high = {constant_above}")
        print(f"  constant_k_low_max_below_z_low = {clamp_below_ok}")
        print(f"  k_at_0.280 = {k_at_0280} (expected {params['k_low_max']})")
        print(f"  k_at_0.500 = {k_at_0500} (expected {params['k_nominal']})")
        print(f"  clamp_check_verified = {clamp_check_verified}")

    all_pass = all(
        r["no_discontinuity"] and r["monotonic_decrease_low_to_high"]
        and r["constant_k_nominal_above_z_high"] and r["clamp_check_verified"]
        for r in results.values()
    )

    if all_pass:
        print("\nPASS: All candidates have continuous, monotonic k_position schedules with verified clamps.")
    else:
        print("\nFAIL: Some candidates have discontinuous or non-monotonic schedules, or clamp checks failed.")


if __name__ == "__main__":
    main()
```

Run: `python scripts/check_schedule_continuity.py`
Expected: CSV written with 181+2 rows per candidate (183 total per candidate, 549 total rows), all_pass = True, clamp_check_verified = True for all candidates

---

## Task 6: Create evaluation script + setup generation for intermediate grid heights

**Files:**

- Create: `scripts/evaluate_continuous_low_height_sagittal_authority_fix.py`
- Create: `outputs/physical_target_height_setups/low_0p330_setup.json` (if missing)
- Create: `outputs/physical_target_height_setups/low_0p360_setup.json` (if missing)
- Create: `outputs/physical_target_height_setups/high_0p450_setup.json` (if missing)

**Purpose:** Evaluate all candidates against the specified evaluation gates, including intermediate grid heights that require generated setup files.

### Setup Generation Rule

Before running dynamic Step E or Step C for any practical grid height, **ensure a statically valid setup JSON exists for that height**.

If the setup JSON does not exist:

1. Generate it using the existing physical target height setup generator (calibrate hip/knee/root_z)
2. Reject root-z-only setup (must have meaningful hip/knee values)
3. Run static validation
4. Only run dynamic validation if static validation passes

Required intermediate setup files (create if missing):

| Setup file                                                    | Height | Purpose                     |
| ------------------------------------------------------------- | ------ | --------------------------- |
| `outputs/physical_target_height_setups/low_0p330_setup.json`  | 0.330  | Practical grid intermediate |
| `outputs/physical_target_height_setups/low_0p360_setup.json`  | 0.360  | Practical grid intermediate |
| `outputs/physical_target_height_setups/high_0p450_setup.json` | 0.450  | Practical grid intermediate |

Existing setup files that should be reused without replacement:

| Setup file                                                    | Height |
| ------------------------------------------------------------- | ------ |
| `outputs/physical_target_height_setups/low_0p300_setup.json`  | 0.300  |
| `outputs/physical_target_height_setups/high_0p480_setup.json` | 0.480  |

Do not replace 0.300 or 0.480 setups with easier targets.

- [ ] **Step 1: Create the evaluation script**

The script must run ALL of the following:

### Candidates to evaluate

1. `baseline` (no continuous schedule)
2. `candidate_E1_k60_continuous`
3. `candidate_E2_k80_continuous`
4. `candidate_E3_k100_continuous`

### Required runs per candidate

| Test              | Steps | Condition           |
| ----------------- | ----- | ------------------- |
| low_0p300 Step E  | 1000  | Required for all    |
| low_0p300 Step E  | 5000  | Only if 1000 passes |
| high_0p480 Step E | 5000  | Required for all    |
| Step C low_0p300  | 5000  | Required for all    |
| Step C high_0p480 | 5000  | Required for all    |

### Height grid runs (Step E)

| Height                | Variant/Setup                    |
| --------------------- | -------------------------------- |
| 0.300                 | low_0p300                        |
| 0.330                 | low_0p330 (generate if missing)  |
| 0.360                 | low_0p360 (generate if missing)  |
| 0.393 or low_small    | low_small (or generated)         |
| nominal (0.400-0.420) | nominal                          |
| high_small            | high_small (if available)        |
| 0.450                 | high_0p450 (generate if missing) |
| 0.480                 | high_0p480                       |

### Step C height grid (at minimum)

| Height  |
| ------- |
| 0.300   |
| 0.360   |
| nominal |
| 0.480   |

### Five-variant regression

| Variant    |
| ---------- |
| nominal    |
| low_tiny   |
| high_tiny  |
| low_small  |
| high_small |

### Output artifacts

The script must produce:

1. `continuous_low_height_candidate_summary.csv` - CSV summary with all metrics per candidate
2. `continuous_low_height_candidate_summary.json` - JSON summary with structured results
3. `selected_candidate_telemetry/` - Telemetry directories from successful runs
4. `continuous_low_height_sagittal_fix_report.md` - Human-readable report
5. `continuous_low_height_sagittal_fix_summary.json` - Machine-readable summary for automation

### Acceptance gates per run

Each candidate/step must be checked against:

| Metric                         | Threshold   | Gate type |
| ------------------------------ | ----------- | --------- |
| support_position_error max_abs | <= 0.15 m   | HARD      |
| hip_yaw_abs_max                | <= 0.07 rad | HARD      |
| pitch_x max_abs                | <= 0.10 rad | HARD      |
| roll_y max_abs                 | <= 0.05 rad | HARD      |
| final height error max_abs     | <= 0.02 m   | HARD      |
| non-wheel floor contacts       | = 0         | HARD      |
| contact valid                  | >= 99.9%    | HARD      |
| WBC applied                    | = false     | HARD      |
| hidden torque                  | = 0         | HARD      |
| ownership violations           | = 0         | HARD      |

### Script skeleton

```python
"""Evaluation script for continuous low-height sagittal authority fix.

Evaluates candidate_E1_k60_continuous, candidate_E2_k80_continuous,
candidate_E3_k100_continuous against baseline across all required
evaluation gates.

Usage:
    python scripts/evaluate_continuous_low_height_sagittal_authority_fix.py
        [--output-dir OUTPUT_DIR]
        [--num-episodes NUM_EPISODES]
        [--max-steps MAX_STEPS]

Outputs:
    continuous_low_height_candidate_summary.csv
    continuous_low_height_candidate_summary.json
    selected_candidate_telemetry/
    continuous_low_height_sagittal_fix_report.md
    continuous_low_height_sagittal_fix_summary.json
"""

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ... implementation ...
```

Run: `python -m py_compile scripts/evaluate_continuous_low_height_sagittal_authority_fix.py`
Expected: PASS (no syntax errors)

---

## Task 7: Integration and smoke test

- [ ] Run all controller unit tests: `pytest tests/test_sagittal_velocity_damped_balance_controller.py -v`
- [ ] Run continuity check: `python scripts/check_schedule_continuity.py`
- [ ] Verify output CSV columns are correct (including `sample_type`)
- [ ] Verify CSV has clamp_check rows at z=0.280 and z=0.500 for each candidate
- [ ] Verify all_pass = True in continuity check
- [ ] Verify clamp_check_verified = True for all candidates
- [ ] Run baseline low_0p300 Step E 1000 as sanity check
- [ ] Run candidate_E2_k80_continuous low_0p300 Step E 1000 as smoke test
- [ ] Verify telemetry includes all required fields

---

## Verification Plan

After implementation, verify:

1. **Function tests**: All unit tests pass
2. **Continuity check**: 181-point dense sweep shows max_abs_delta_k_position < 0.5, no_discontinuity = True, monotonic_decrease_low_to_high = True
3. **Clamp checks**: k(0.280) == k_low_max and k(0.500) == k_nominal for all candidates, verified via clamp_check rows
4. **Baseline smoke**: Baseline low_0p300 fails acceptance criteria (expected)
5. **Candidate smoke**: At least one candidate passes low_0p300 acceptance criteria
6. **Regression check**: Nominal and five variants do not regress
7. **Telemetry check**: All required fields present in diagnostics output
8. **Final decision**: BOUNDARY_RANGE_PASS, LOW_HEIGHT_SAGITTAL_FIX_REQUIRED, FIX_CAUSED_REGRESSION, or NEW_ROOT_CAUSE_FOUND

---

**Sections changed:** File header (added File Map), Candidate Profiles table, Acceptance Criteria table, Final Decision Logic section, No-Patchwork Rule section, Commanded/Reference Height Source Rule section, Task 5 (fixed clamp check: now uses sample_type column, evaluates clamp_check rows at 0.280/0.500, reports k_at_0.280 and k_at_0.500, does NOT claim below-z_low verified from dense rows), Task 6 (added Setup Generation Rule, intermediate setup files to create if missing, explicit "do not replace 0.300/0.480" rule), Task 7 (added clamp_check_verified verification step), Verification Plan (added clamp check verification item).

**Whether code was implemented:** false

**Whether git commit was made:** false
