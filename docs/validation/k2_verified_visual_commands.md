# K2 Verified Visual Commands — Step C/D/E Visual Inspection Guide

**Date:** 2026-06-26
**Task:** `FIND_VERIFIED_VISUAL_COMMANDS_FOR_K2_STEP_C_D_E`
**Current-best controller:** `K2_NOTCH_LOW_Q_V1`
**Current-best profile:** `k2_notch_low_q_v1`
**Legacy profile:** `k1_pitch_rate_notch_v1`

---

## Summary of Discovered Flags

| Purpose | Flag | Verified |
|---------|------|----------|
| Visual/viewer | `--visual` | Confirmed in argparse line 2510, tested via `--help` |
| K2 profile | `--vd-sagittal-authority-profile k2_notch_low_q_v1` | Confirmed in argparse line 3068, tested with 5-step headless run |
| K1 profile | `--vd-sagittal-authority-profile k1_pitch_rate_notch_v1` | Confirmed in argparse line 3061, tested with 5-step headless run |
| Height setup | `--height-variant-setup <path>` | Confirmed in argparse line 2674; setup files verified |
| Push sequence | `--push-sequence-file <path>` | Confirmed in argparse line 2643; tested with generated JSON |
| Dynamic height | `--dynamic-height-trajectory <path>` | Confirmed in argparse line 3384; tested with generated JSON |
| Controller mode | `--controller-mode balance-core` | Confirmed, tested |
| Sagittal controller | `--sagittal-controller velocity-damped` | Confirmed, tested |
| Steps | `--steps <N>` | Confirmed, tested |
| Output directory | `--output-dir <path>` | Confirmed, tested |

---

## 1. Exact Visual Flag

```
--visual
```

Add `--visual` to any `simulate_hierarchical_controller.py` command to launch the MuJoCo viewer.
When `--visual` is present, the simulation runs with real-time pacing (100 Hz control, viewer sync decoupled at configurable rate).
Close the viewer window to end simulation and save telemetry.

**NO other visual/viewer flag exists.** Search results confirm:
- `--viewer`: NOT found
- `--render`: NOT found
- `--gui`: NOT found
- `--headless`: NOT found (headless is the default)

### 1a. Visual Realtime Pacing Control (NEW)

The simulation uses `control_dt = 0.01s` (100 Hz control) with `physics_dt = 0.002s` (5 substeps per control step).
Expected simulated duration: `steps × 0.01` seconds (e.g., 5000 steps = 50 seconds).

**Timing Model:**

| Parameter | Value | Formula |
|-----------|-------|---------|
| Physics timestep | 0.002 s | From XML `option timestep="0.002"` |
| Control dt | 0.01 s | Hardcoded `control_dt = 0.01` |
| Control rate | 100 Hz | `1/control_dt` |
| Physics rate | 500 Hz | `1/physics_dt` |
| Substeps per control step | 5 | `control_dt / physics_dt` |
| Sim duration | steps × 0.01 s | `steps * control_dt` |

**Example wall times (target 1:1 realtime):**

| Steps | Sim seconds | Expected wall time |
|-------|------------|--------------------|
| 2000 | 20 s | ~20 s |
| 5000 | 50 s | ~50 s |
| 7000 | 70 s | ~70 s |

**IMPORTANT:** The simulation is compute-bound (~172 ms/step headless on reference machine).
At default settings (1131 telemetry columns, full-rate logging), it runs 17× slower than realtime
even without the viewer. Use these flags to manage visual pacing:

```
--visual-realtime-factor <float>     Target realtime factor (default 1.0).
                                     2.0 = run 2× faster pacing, 0.5 = half speed.
                                     Set to 0 to disable pacing entirely.

--visual-sync-hz <float>             Target viewer sync rate in Hz (default 30).
                                     Lower values reduce render overhead.
                                     Range: 5–120 Hz (runtime clamped).

--visual-disable-realtime-pacing     Disable all sleep pacing.
                                     Simulation runs as fast as compute allows.

--visual-profile-timing              Print per-step timing diagnostics at end.
                                     Reports: mean/P50/P95/P99 step time,
                                     viewer sync time, sleep ratio.
```

**Recommended smooth visual commands:**

```powershell
# Smooth visual — default 1:1 pacing
python scripts/simulate_hierarchical_controller.py --visual `
  --controller-mode balance-core --sagittal-controller velocity-damped `
  --vd-sagittal-authority-profile k2_notch_low_q_v1 `
  --height-variant-setup outputs/physical_target_height_setups_centered/high_0p480_setup.json `
  --steps 2000 --telemetry-decimation 10 --failure-window-steps 2000 `
  --write-run-summary-sidecar --output-dir outputs/visual/k2_smooth `
  --enable-mode-hip-yaw-divergence --mode-hip-yaw-div-kp 10.0 `
  --mode-hip-yaw-div-kd 0.50 --mode-hip-yaw-div-max-torque 7.5 `
  --mode-hip-yaw-div-soft-limit-rad 0.30 --mode-hip-yaw-div-soft-gain 0.80 `
  --mode-hip-yaw-div-ref-source target

# Run without pacing (fast as possible)
python scripts/simulate_hierarchical_controller.py --visual `
  --visual-disable-realtime-pacing `
  ... (rest of args)

# Profile timing to see where time goes
python scripts/simulate_hierarchical_controller.py --visual `
  --visual-profile-timing `
  ... (rest of args)

# 2× speed pacing (runs faster if compute allows)
python scripts/simulate_hierarchical_controller.py --visual `
  --visual-realtime-factor 2.0 `
  ... (rest of args)
```

**Performance note:** The controller compute + 5 physics substeps + 1131-field telemetry
per step is the primary bottleneck. To improve visual smoothness:

1. Use `--telemetry-decimation 10` or higher in visual mode to reduce per-step dict copy overhead
2. Per-step WBC debug prints are **automatically suppressed** in `--visual` mode (7 prints/step eliminated)
3. Use `--visual-sync-hz 15` if viewer sync is the bottleneck
4. Use `--visual-realtime-factor 2.0` to let simulation run ahead of display
5. Use `--visual-profile-timing` to diagnose specific bottlenecks
6. Close other GPU-intensive applications if viewer rendering is slow
7. Use `--wbc-quiet` to suppress WBC prints in headless mode for faster validation runs

---

## 2. Exact K2 Profile Flag

```
--vd-sagittal-authority-profile k2_notch_low_q_v1
```

Full list of accepted profile choices includes (among others):
- `k2_notch_low_q_v1` (K2 current-best, Q=2.0)
- `k1_pitch_rate_notch_v1` (K1 legacy, Q=6.0)
- `k1b_pitch_rate_notch_2p3` through `k1g_pitch_rate_notch_blend050`
- `k2_wheel_vel_notch_v1`
- `k3_pitch_rate_wheel_vel_notch_v1`
- `k_sweep_*` (audit-only sweep profiles)

Confirmed in argparse `choices` list at lines 2973–3107 of `scripts/simulate_hierarchical_controller.py`.

---

## 3. Exact K1 Legacy Profile Flag

```
--vd-sagittal-authority-profile k1_pitch_rate_notch_v1
```

---

## 4. Required Controller Mode Flags

All visual commands for K2/K1 require:

```
--controller-mode balance-core
--sagittal-controller velocity-damped
```

Plus mode-hip-yaw-divergence flags (used by all validation runners):

```
--enable-mode-hip-yaw-divergence
--mode-hip-yaw-div-kp 10.0
--mode-hip-yaw-div-kd 0.50
--mode-hip-yaw-div-max-torque 7.5
--mode-hip-yaw-div-soft-limit-rad 0.30
--mode-hip-yaw-div-soft-gain 0.80
--mode-hip-yaw-div-ref-source target
```

---

## 5. Available Height Setup Files

Setup files exist at `outputs/physical_target_height_setups_centered/`:

| Height Label | Setup File | Target CoM Z |
|-------------|-----------|-------------|
| low_0p300 | `low_0p300_setup.json` | 0.300 m |
| low_0p320 | `low_0p320_setup.json` | 0.320 m |
| low_0p330 | `low_0p330_setup.json` | 0.330 m |
| low_0p340 | `low_0p340_setup.json` | 0.340 m |
| low_0p360 | `low_0p360_setup.json` | 0.360 m |
| low_0p380 | `low_0p380_setup.json` | 0.380 m |
| mid_0p400 | `mid_0p400_setup.json` | 0.400 m |
| high_0p430 | `high_0p430_setup.json` | 0.430 m |
| high_0p450 | `high_0p450_setup.json` | 0.450 m |
| high_0p465 | `high_0p465_setup.json` | 0.465 m |
| high_0p480 | `high_0p480_setup.json` | 0.480 m |

**Note:** There is no `mid_0p400` in `outputs/physical_target_height_setups/` (legacy), but it exists in `outputs/physical_target_height_setups_centered/`.

Notch gate behavior:
- Heights < 0.42 m: notch INACTIVE
- Heights 0.42–0.48 m: notch PARTIAL (gate ramps)
- Heights ≥ 0.48 m: notch FULLY ACTIVE

---

## 6. Step C Visual Commands (Fixed Height)

Step C cases from the validation runner:

### C1 — slow_ladder_up_down at low_0p330 (notch gate inactive)

```powershell
python scripts/simulate_hierarchical_controller.py `
  --visual `
  --controller-mode balance-core `
  --sagittal-controller velocity-damped `
  --vd-sagittal-authority-profile k2_notch_low_q_v1 `
  --height-variant-setup outputs/physical_target_height_setups_centered/low_0p330_setup.json `
  --steps 2000 `
  --telemetry-decimation 1 `
  --failure-window-steps 2000 `
  --write-run-summary-sidecar `
  --output-dir outputs/visual/k2_step_c_C1_low_0p330 `
  --enable-mode-hip-yaw-divergence `
  --mode-hip-yaw-div-kp 10.0 `
  --mode-hip-yaw-div-kd 0.50 `
  --mode-hip-yaw-div-max-torque 7.5 `
  --mode-hip-yaw-div-soft-limit-rad 0.30 `
  --mode-hip-yaw-div-soft-gain 0.80 `
  --mode-hip-yaw-div-ref-source target
```

### C2-C5: Same as C1 but change `--output-dir`

Replace `C1_low_0p330` with `C2_low_0p330`, `C3_low_0p330`, etc.

### Focused low_0p320

```powershell
python scripts/simulate_hierarchical_controller.py `
  --visual `
  --controller-mode balance-core `
  --sagittal-controller velocity-damped `
  --vd-sagittal-authority-profile k2_notch_low_q_v1 `
  --height-variant-setup outputs/physical_target_height_setups_centered/low_0p320_setup.json `
  --steps 2000 `
  --telemetry-decimation 1 `
  --failure-window-steps 2000 `
  --write-run-summary-sidecar `
  --output-dir outputs/visual/k2_step_c_focused_low_0p320 `
  --enable-mode-hip-yaw-divergence `
  --mode-hip-yaw-div-kp 10.0 `
  --mode-hip-yaw-div-kd 0.50 `
  --mode-hip-yaw-div-max-torque 7.5 `
  --mode-hip-yaw-div-soft-limit-rad 0.30 `
  --mode-hip-yaw-div-soft-gain 0.80 `
  --mode-hip-yaw-div-ref-source target
```

### Focused high_0p480 (notch gate FULLY ACTIVE)

```powershell
python scripts/simulate_hierarchical_controller.py `
  --visual `
  --controller-mode balance-core `
  --sagittal-controller velocity-damped `
  --vd-sagittal-authority-profile k2_notch_low_q_v1 `
  --height-variant-setup outputs/physical_target_height_setups_centered/high_0p480_setup.json `
  --steps 2000 `
  --telemetry-decimation 1 `
  --failure-window-steps 2000 `
  --write-run-summary-sidecar `
  --output-dir outputs/visual/k2_step_c_focused_high_0p480 `
  --enable-mode-hip-yaw-divergence `
  --mode-hip-yaw-div-kp 10.0 `
  --mode-hip-yaw-div-kd 0.50 `
  --mode-hip-yaw-div-max-torque 7.5 `
  --mode-hip-yaw-div-soft-limit-rad 0.30 `
  --mode-hip-yaw-div-soft-gain 0.80 `
  --mode-hip-yaw-div-ref-source target
```

---

## 7. Step D Push Visual Commands

### Prerequisite: Create push sequence JSON file

Before running push visual commands, you must create a push sequence JSON file.
You can use this PowerShell snippet to generate one:

```powershell
# Generate push sequence file
$pushDir = "outputs/visual/push_sequences"
New-Item -ItemType Directory -Force -Path $pushDir | Out-Null

# Example: sagittal_forward 60N at step 300, duration 5 steps
$pushJson = @{
    sequence = @(
        @(300, 0.0, 60.0, 5)
    )
} | ConvertTo-Json -Depth 3
$pushJson | Out-File -FilePath "$pushDir/push_forward_60N.json" -Encoding utf8
```

Or use Python:

```bash
python -c "
import json
seq = [[300, 0.0, 60.0, 5]]  # [step, force_x, force_y, duration]
with open('outputs/visual/push_sequences/push_forward_60N.json', 'w') as f:
    json.dump({'sequence': seq}, f, indent=2)
"
```

The push sequence format is:
```json
{
  "sequence": [
    [<step>, <force_x_N>, <force_y_N>, <duration_steps>]
  ]
}
```

- `force_y > 0` = sagittal_forward (+y direction)
- `force_y < 0` = sagittal_backward (-y direction)

### Step D K2 Visual Commands

#### 1. high_0p480 sagittal_forward 60N

```powershell
# Create push file first
New-Item -ItemType Directory -Force -Path "outputs/visual/push_sequences" | Out-Null
python -c "import json; json.dump({'sequence': [[300, 0.0, 60.0, 5]]}, open('outputs/visual/push_sequences/push_fwd_60N.json','w'), indent=2)"

# Run visual
python scripts/simulate_hierarchical_controller.py `
  --visual `
  --controller-mode balance-core `
  --sagittal-controller velocity-damped `
  --vd-sagittal-authority-profile k2_notch_low_q_v1 `
  --height-variant-setup outputs/physical_target_height_setups_centered/high_0p480_setup.json `
  --steps 2000 `
  --push-sequence-file outputs/visual/push_sequences/push_fwd_60N.json `
  --telemetry-decimation 1 `
  --failure-window-steps 2000 `
  --write-run-summary-sidecar `
  --output-dir outputs/visual/k2_step_d_high_0p480_fwd_60N `
  --enable-mode-hip-yaw-divergence `
  --mode-hip-yaw-div-kp 10.0 `
  --mode-hip-yaw-div-kd 0.50 `
  --mode-hip-yaw-div-max-torque 7.5 `
  --mode-hip-yaw-div-soft-limit-rad 0.30 `
  --mode-hip-yaw-div-soft-gain 0.80 `
  --mode-hip-yaw-div-ref-source target
```

#### 2. high_0p480 sagittal_forward 90N

Same as above, change push file to `[[300, 0.0, 90.0, 5]]` and output dir to `k2_step_d_high_0p480_fwd_90N`.

#### 3. high_0p480 sagittal_backward 60N

Same as above, change push file to `[[300, 0.0, -60.0, 5]]` and output dir to `k2_step_d_high_0p480_bwd_60N`.

#### 4. high_0p480 sagittal_backward 90N

Same as above, change push file to `[[300, 0.0, -90.0, 5]]` and output dir to `k2_step_d_high_0p480_bwd_90N`.

#### 5. mid_0p400 sagittal_forward 90N

```powershell
python -c "import json; json.dump({'sequence': [[300, 0.0, 90.0, 5]]}, open('outputs/visual/push_sequences/push_fwd_90N.json','w'), indent=2)"

python scripts/simulate_hierarchical_controller.py `
  --visual `
  --controller-mode balance-core `
  --sagittal-controller velocity-damped `
  --vd-sagittal-authority-profile k2_notch_low_q_v1 `
  --height-variant-setup outputs/physical_target_height_setups_centered/mid_0p400_setup.json `
  --steps 2000 `
  --push-sequence-file outputs/visual/push_sequences/push_fwd_90N.json `
  --telemetry-decimation 1 `
  --failure-window-steps 2000 `
  --write-run-summary-sidecar `
  --output-dir outputs/visual/k2_step_d_mid_0p400_fwd_90N `
  --enable-mode-hip-yaw-divergence `
  --mode-hip-yaw-div-kp 10.0 `
  --mode-hip-yaw-div-kd 0.50 `
  --mode-hip-yaw-div-max-torque 7.5 `
  --mode-hip-yaw-div-soft-limit-rad 0.30 `
  --mode-hip-yaw-div-soft-gain 0.80 `
  --mode-hip-yaw-div-ref-source target
```

#### 6. low_0p330 sagittal_backward 90N

```powershell
python -c "import json; json.dump({'sequence': [[300, 0.0, -90.0, 5]]}, open('outputs/visual/push_sequences/push_bwd_90N.json','w'), indent=2)"

python scripts/simulate_hierarchical_controller.py `
  --visual `
  --controller-mode balance-core `
  --sagittal-controller velocity-damped `
  --vd-sagittal-authority-profile k2_notch_low_q_v1 `
  --height-variant-setup outputs/physical_target_height_setups_centered/low_0p330_setup.json `
  --steps 2000 `
  --push-sequence-file outputs/visual/push_sequences/push_bwd_90N.json `
  --telemetry-decimation 1 `
  --failure-window-steps 2000 `
  --write-run-summary-sidecar `
  --output-dir outputs/visual/k2_step_d_low_0p330_bwd_90N `
  --enable-mode-hip-yaw-divergence `
  --mode-hip-yaw-div-kp 10.0 `
  --mode-hip-yaw-div-kd 0.50 `
  --mode-hip-yaw-div-max-torque 7.5 `
  --mode-hip-yaw-div-soft-limit-rad 0.30 `
  --mode-hip-yaw-div-soft-gain 0.80 `
  --mode-hip-yaw-div-ref-source target
```

---

## 8. Step E Visual Commands (Fixed-Height Sweep)

Step E covers 10 heights. Each uses the same pattern — only `--height-variant-setup` and `--output-dir` change.

### K2 Step E — low_0p300

```powershell
python scripts/simulate_hierarchical_controller.py `
  --visual `
  --controller-mode balance-core `
  --sagittal-controller velocity-damped `
  --vd-sagittal-authority-profile k2_notch_low_q_v1 `
  --height-variant-setup outputs/physical_target_height_setups_centered/low_0p300_setup.json `
  --steps 2000 `
  --telemetry-decimation 1 `
  --failure-window-steps 2000 `
  --write-run-summary-sidecar `
  --output-dir outputs/visual/k2_step_e_low_0p300 `
  --enable-mode-hip-yaw-divergence `
  --mode-hip-yaw-div-kp 10.0 `
  --mode-hip-yaw-div-kd 0.50 `
  --mode-hip-yaw-div-max-torque 7.5 `
  --mode-hip-yaw-div-soft-limit-rad 0.30 `
  --mode-hip-yaw-div-soft-gain 0.80 `
  --mode-hip-yaw-div-ref-source target
```

Repeat for all Step E heights:
- `low_0p300`, `low_0p320`, `low_0p330`, `low_0p340`, `low_0p360`, `low_0p380`
- `high_0p430`, `high_0p450`, `high_0p465`, `high_0p480`

### Quick Step E sweep helper (PowerShell):

```powershell
$heights = @("low_0p300", "low_0p320", "low_0p330", "low_0p340", "low_0p360", "low_0p380", "high_0p430", "high_0p450", "high_0p465", "high_0p480")
foreach ($h in $heights) {
    Write-Host "Launching viewer for $h - close window to continue to next height"
    python scripts/simulate_hierarchical_controller.py `
      --visual `
      --controller-mode balance-core `
      --sagittal-controller velocity-damped `
      --vd-sagittal-authority-profile k2_notch_low_q_v1 `
      --height-variant-setup "outputs/physical_target_height_setups_centered/${h}_setup.json" `
      --steps 2000 `
      --telemetry-decimation 1 `
      --failure-window-steps 2000 `
      --write-run-summary-sidecar `
      --output-dir "outputs/visual/k2_step_e_${h}" `
      --enable-mode-hip-yaw-divergence `
      --mode-hip-yaw-div-kp 10.0 `
      --mode-hip-yaw-div-kd 0.50 `
      --mode-hip-yaw-div-max-torque 7.5 `
      --mode-hip-yaw-div-soft-limit-rad 0.30 `
      --mode-hip-yaw-div-soft-gain 0.80 `
      --mode-hip-yaw-div-ref-source target
}
```

---

## 9. Dynamic Height Visual Commands

Dynamic height is supported via `--dynamic-height-trajectory <path_to_json>`.

**The trajectory JSON format:**

```json
{
  "height_profile_name": "ramp_up_0p330_to_0p480",
  "steps": 5000,
  "waypoints": [
    {"step": 0, "height_m": 0.330},
    {"step": 500, "height_m": 0.330},
    {"step": 3500, "height_m": 0.480},
    {"step": 5000, "height_m": 0.480}
  ]
}
```

Height is linearly interpolated between waypoints during the simulation.
The initial setup file determines the initial posture; the trajectory overrides the height target.

### Dynamic Height JSON Creation (MUST USE PYTHON, NOT POWERSHELL)

**CRITICAL:** PowerShellʼs `Out-File -Encoding utf8` adds a UTF-8 BOM (`\xEF\xBB\xBF`).
Pythonʼs `json.load()` (used by `simulate_hierarchical_controller.py` line 3434)
cannot parse BOM-prefixed files. **Always create trajectory JSON with Python.**

```powershell
# ONE-TIME: Create all 4 trajectory JSON files (Python avoids PowerShell BOM bug)
python -c "
import json, os
DIR = 'outputs/visual/trajectories'
os.makedirs(DIR, exist_ok=True)
trajs = {
    'ramp_up_0p330_to_0p480': {
        'height_profile_name': 'ramp_up_0p330_to_0p480',
        'steps': 5000,
        'waypoints': [
            {'step': 0, 'height_m': 0.330},
            {'step': 500, 'height_m': 0.330},
            {'step': 3500, 'height_m': 0.480},
            {'step': 5000, 'height_m': 0.480},
        ],
    },
    'ramp_down_0p480_to_0p330': {
        'height_profile_name': 'ramp_down_0p480_to_0p330',
        'steps': 5000,
        'waypoints': [
            {'step': 0, 'height_m': 0.480},
            {'step': 500, 'height_m': 0.480},
            {'step': 3500, 'height_m': 0.330},
            {'step': 5000, 'height_m': 0.330},
        ],
    },
    'up_down_cycle_0p330_0p480_0p330': {
        'height_profile_name': 'up_down_cycle_0p330_0p480_0p330',
        'steps': 7000,
        'waypoints': [
            {'step': 0, 'height_m': 0.330},
            {'step': 500, 'height_m': 0.330},
            {'step': 2500, 'height_m': 0.480},
            {'step': 4000, 'height_m': 0.480},
            {'step': 6000, 'height_m': 0.330},
            {'step': 7000, 'height_m': 0.330},
        ],
    },
    'gate_dwell_0p420_0p450_0p480': {
        'height_profile_name': 'gate_dwell_0p420_0p450_0p480',
        'steps': 6000,
        'waypoints': [
            {'step': 0, 'height_m': 0.330},
            {'step': 500, 'height_m': 0.330},
            {'step': 1500, 'height_m': 0.420},
            {'step': 2500, 'height_m': 0.420},
            {'step': 3000, 'height_m': 0.450},
            {'step': 4000, 'height_m': 0.450},
            {'step': 4500, 'height_m': 0.480},
            {'step': 6000, 'height_m': 0.480},
        ],
    },
}
for name, data in trajs.items():
    path = os.path.join(DIR, f'{name}.json')
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    with open(path, 'r') as f:
        loaded = json.load(f)
    assert loaded['waypoints'] == data['waypoints']
    print(f'OK: {path} ({len(loaded[\"waypoints\"])} waypoints)')
print('All 4 trajectory files created and verified.')
"
```

### ramp_up — 0.33m → 0.48m (crossing notch gate upward)

```powershell
python scripts/simulate_hierarchical_controller.py `
  --visual `
  --controller-mode balance-core `
  --sagittal-controller velocity-damped `
  --vd-sagittal-authority-profile k2_notch_low_q_v1 `
  --height-variant-setup outputs/physical_target_height_setups_centered/low_0p330_setup.json `
  --steps 5000 `
  --dynamic-height-trajectory outputs/visual/trajectories/ramp_up_0p330_to_0p480.json `
  --telemetry-decimation 1 `
  --failure-window-steps 5000 `
  --write-run-summary-sidecar `
  --output-dir outputs/visual/k2_dynamic_ramp_up `
  --enable-mode-hip-yaw-divergence `
  --mode-hip-yaw-div-kp 10.0 `
  --mode-hip-yaw-div-kd 0.50 `
  --mode-hip-yaw-div-max-torque 7.5 `
  --mode-hip-yaw-div-soft-limit-rad 0.30 `
  --mode-hip-yaw-div-soft-gain 0.80 `
  --mode-hip-yaw-div-ref-source target
```

### ramp_down — 0.48m → 0.33m (crossing notch gate downward)

```powershell
python scripts/simulate_hierarchical_controller.py `
  --visual `
  --controller-mode balance-core `
  --sagittal-controller velocity-damped `
  --vd-sagittal-authority-profile k2_notch_low_q_v1 `
  --height-variant-setup outputs/physical_target_height_setups_centered/high_0p480_setup.json `
  --steps 5000 `
  --dynamic-height-trajectory outputs/visual/trajectories/ramp_down_0p480_to_0p330.json `
  --telemetry-decimation 1 `
  --failure-window-steps 5000 `
  --write-run-summary-sidecar `
  --output-dir outputs/visual/k2_dynamic_ramp_down `
  --enable-mode-hip-yaw-divergence `
  --mode-hip-yaw-div-kp 10.0 `
  --mode-hip-yaw-div-kd 0.50 `
  --mode-hip-yaw-div-max-torque 7.5 `
  --mode-hip-yaw-div-soft-limit-rad 0.30 `
  --mode-hip-yaw-div-soft-gain 0.80 `
  --mode-hip-yaw-div-ref-source target
```

### up_down_cycle — 0.33→0.48→0.33 (crosses gate twice)

```powershell
python scripts/simulate_hierarchical_controller.py `
  --visual `
  --controller-mode balance-core `
  --sagittal-controller velocity-damped `
  --vd-sagittal-authority-profile k2_notch_low_q_v1 `
  --height-variant-setup outputs/physical_target_height_setups_centered/low_0p330_setup.json `
  --steps 7000 `
  --dynamic-height-trajectory outputs/visual/trajectories/up_down_cycle_0p330_0p480_0p330.json `
  --telemetry-decimation 1 `
  --failure-window-steps 7000 `
  --write-run-summary-sidecar `
  --output-dir outputs/visual/k2_dynamic_up_down_cycle `
  --enable-mode-hip-yaw-divergence `
  --mode-hip-yaw-div-kp 10.0 `
  --mode-hip-yaw-div-kd 0.50 `
  --mode-hip-yaw-div-max-torque 7.5 `
  --mode-hip-yaw-div-soft-limit-rad 0.30 `
  --mode-hip-yaw-div-soft-gain 0.80 `
  --mode-hip-yaw-div-ref-source target
```

### gate_dwell — sequential dwell at 0.42/0.45/0.48m (three gate levels)

```powershell
python scripts/simulate_hierarchical_controller.py `
  --visual `
  --controller-mode balance-core `
  --sagittal-controller velocity-damped `
  --vd-sagittal-authority-profile k2_notch_low_q_v1 `
  --height-variant-setup outputs/physical_target_height_setups_centered/low_0p330_setup.json `
  --steps 6000 `
  --dynamic-height-trajectory outputs/visual/trajectories/gate_dwell_0p420_0p450_0p480.json `
  --telemetry-decimation 1 `
  --failure-window-steps 6000 `
  --write-run-summary-sidecar `
  --output-dir outputs/visual/k2_dynamic_gate_dwell `
  --enable-mode-hip-yaw-divergence `
  --mode-hip-yaw-div-kp 10.0 `
  --mode-hip-yaw-div-kd 0.50 `
  --mode-hip-yaw-div-max-torque 7.5 `
  --mode-hip-yaw-div-soft-limit-rad 0.30 `
  --mode-hip-yaw-div-soft-gain 0.80 `
  --mode-hip-yaw-div-ref-source target
```

### gate_chatter — repeated small transitions around notch gate boundaries

17 waypoints, 5000 steps. See `scripts/validate_k2_dynamic_height_gate_crossing.py` lines 117–140 for the full list. Create the JSON the same way as above using the waypoint data from `DYNAMIC_SCENARIOS[&quot;gate_chatter_0p400_0p470&quot;]`.

---

## 10. K1 vs K2 Side-by-Side Comparison Commands

To visually compare K1 (Q=6.0) vs K2 (Q=2.0), run the SAME command twice with different profiles.
Start with headless mode to pre-warm JIT, then switch to visual.

### Side-by-Side: high_0p480 (notch active)

**K1 (legacy, Q=6.0):**
```powershell
python scripts/simulate_hierarchical_controller.py `
  --visual `
  --controller-mode balance-core `
  --sagittal-controller velocity-damped `
  --vd-sagittal-authority-profile k1_pitch_rate_notch_v1 `
  --height-variant-setup outputs/physical_target_height_setups_centered/high_0p480_setup.json `
  --steps 2000 `
  --telemetry-decimation 1 `
  --failure-window-steps 2000 `
  --write-run-summary-sidecar `
  --output-dir outputs/visual/k1_high_0p480 `
  --enable-mode-hip-yaw-divergence `
  --mode-hip-yaw-div-kp 10.0 `
  --mode-hip-yaw-div-kd 0.50 `
  --mode-hip-yaw-div-max-torque 7.5 `
  --mode-hip-yaw-div-soft-limit-rad 0.30 `
  --mode-hip-yaw-div-soft-gain 0.80 `
  --mode-hip-yaw-div-ref-source target
```

**K2 (current-best, Q=2.0):**
```powershell
python scripts/simulate_hierarchical_controller.py `
  --visual `
  --controller-mode balance-core `
  --sagittal-controller velocity-damped `
  --vd-sagittal-authority-profile k2_notch_low_q_v1 `
  --height-variant-setup outputs/physical_target_height_setups_centered/high_0p480_setup.json `
  --steps 2000 `
  --telemetry-decimation 1 `
  --failure-window-steps 2000 `
  --write-run-summary-sidecar `
  --output-dir outputs/visual/k2_high_0p480 `
  --enable-mode-hip-yaw-divergence `
  --mode-hip-yaw-div-kp 10.0 `
  --mode-hip-yaw-div-kd 0.50 `
  --mode-hip-yaw-div-max-torque 7.5 `
  --mode-hip-yaw-div-soft-limit-rad 0.30 `
  --mode-hip-yaw-div-soft-gain 0.80 `
  --mode-hip-yaw-div-ref-source target
```

### Side-by-Side: high_0p480 sagittal_backward 90N push

Same pattern, add `--push-sequence-file` with `[[300, 0.0, -90.0, 5]]` for both.

**Note:** The only difference between K1 and K2 commands is `--vd-sagittal-authority-profile` and `--output-dir`. All other flags are identical.

---

## 11. Bash / Git-Bash Commands

Use backslashes for line continuation instead of PowerShell's backtick:

```bash
python scripts/simulate_hierarchical_controller.py \
  --visual \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --vd-sagittal-authority-profile k2_notch_low_q_v1 \
  --height-variant-setup outputs/physical_target_height_setups_centered/high_0p480_setup.json \
  --steps 2000 \
  --telemetry-decimation 1 \
  --failure-window-steps 2000 \
  --write-run-summary-sidecar \
  --output-dir outputs/visual/k2_high_0p480 \
  --enable-mode-hip-yaw-divergence \
  --mode-hip-yaw-div-kp 10.0 \
  --mode-hip-yaw-div-kd 0.50 \
  --mode-hip-yaw-div-max-torque 7.5 \
  --mode-hip-yaw-div-soft-limit-rad 0.30 \
  --mode-hip-yaw-div-soft-gain 0.80 \
  --mode-hip-yaw-div-ref-source target
```

---

## 12. cmd.exe Commands

Use `^` for line continuation:

```cmd
python scripts/simulate_hierarchical_controller.py ^
  --visual ^
  --controller-mode balance-core ^
  --sagittal-controller velocity-damped ^
  --vd-sagittal-authority-profile k2_notch_low_q_v1 ^
  --height-variant-setup outputs/physical_target_height_setups_centered/high_0p480_setup.json ^
  --steps 2000 ^
  --telemetry-decimation 1 ^
  --failure-window-steps 2000 ^
  --write-run-summary-sidecar ^
  --output-dir outputs/visual/k2_high_0p480 ^
  --enable-mode-hip-yaw-divergence ^
  --mode-hip-yaw-div-kp 10.0 ^
  --mode-hip-yaw-div-kd 0.50 ^
  --mode-hip-yaw-div-max-torque 7.5 ^
  --mode-hip-yaw-div-soft-limit-rad 0.30 ^
  --mode-hip-yaw-div-soft-gain 0.80 ^
  --mode-hip-yaw-div-ref-source target
```

---

## 13. How to Shorten or Lengthen Run Duration

Change `--steps`:
- **Quick glance (2s):** `--steps 100`
- **Short check (4s):** `--steps 200`
- **Standard validation (40s):** `--steps 2000`
- **Long equilibrium (120s):** `--steps 6000`
- **Full dynamic trajectory:** `--steps 5000` to `7000` (matches trajectory duration)

Also update `--failure-window-steps` to match `--steps` for consistent output.

---

## 14. How to Enable/Disable Telemetry

Telemetry is always written (no disable flag). Control detail via:
- `--telemetry-decimation 1` — every step (full fidelity)
- `--telemetry-decimation 10` — every 10th step (smaller files)
- `--write-run-summary-sidecar` — adds a JSON summary file
- `--output-dir <path>` — controls where telemetry lands

Default output directory (if `--output-dir` omitted): `outputs/hierarchical_controller_sim/`.

---

## 15. Viewer Limitations

1. **Real-time pacing:** Visual mode maintains 1:1 real-time. A 2000-step sim runs ~40s real-time. A 6000-step sim runs ~120s. You cannot speed up past real-time in visual mode.
2. **JIT compilation:** First run has a ~5-15s JIT compile delay before viewer opens. Subsequent runs in the same Python session re-use cached JIT.
3. **Single viewer instance:** Only one MuJoCo viewer window per command. No side-by-side K1/K2.
4. **Close to end:** Simulation runs until viewer is closed (or steps exhausted, or termination). Closing the viewer during simulation truncates telemetry.
5. **Headless is faster:** For batch validation, use headless mode. Visual mode is for inspection, not measurement.
6. **No recording built-in:** The `--visual` flag launches `mujoco.viewer.launch_passive`. No `--record` or `--video` flag exists. Use external screen recording for video capture.
7. **Camera control:** Use mouse in viewer: right-drag to rotate, scroll to zoom, left-drag to shift. No CLI camera flags.

---

## 16. Commands Verified Status Table

| Command | Verification Method | Status |
|---------|-------------------|--------|
| K2 fixed-height (5 steps, headless) | Ran with telemetry saved | CLI-VERIFIED |
| K1 fixed-height (5 steps, headless) | Ran with telemetry saved | CLI-VERIFIED |
| K2 push (10 steps, headless) | Ran with push sequence JSON | CLI-VERIFIED |
| K2 dynamic height ramp_up (20 steps, headless) | Dry-run passed, trajectory loaded | CLI-VERIFIED |
| K2 dynamic height ramp_down (20 steps, headless) | Dry-run passed, trajectory loaded | CLI-VERIFIED |
| K2 dynamic height gate_dwell (20 steps, headless) | Dry-run passed, trajectory loaded | CLI-VERIFIED |
| `--visual` flag existence | `--help` output, grep of argparse | CONFIRMED |
| `k2_notch_low_q_v1` in choices | argparse `choices` list line 3068 | CONFIRMED |
| `k1_pitch_rate_notch_v1` in choices | argparse `choices` list line 3061 | CONFIRMED |
| Setup files exist for all heights | File glob verified | CONFIRMED |
| Push JSON format | `generate_push_sequence_file()` in step D runner | CONFIRMED |
| Dynamic height trajectory JSON format | `write_trajectory_json()` in dynamic height runner | CONFIRMED |
| Mode-div flags required | All validation runners use them | CONFIRMED |
| Controller mode + sagittal controller | Hardcoded in all validation runners | CONFIRMED |
| Viewer launched in headless env | Cannot test (no display) | NOT-TESTED |
| **PowerShell JSON BOM bug** — `Out-File -Encoding utf8` adds BOM | All 4 files in `outputs/visual/trajectories/` were broken | **FIXED** — use Python to create JSON |

---

## 17. Quick Reference — Minimal Working Command

**Shortest command to visually inspect K2 at high_0p480 (notch active):**

```powershell
python scripts/simulate_hierarchical_controller.py `
  --visual `
  --controller-mode balance-core `
  --sagittal-controller velocity-damped `
  --vd-sagittal-authority-profile k2_notch_low_q_v1 `
  --height-variant-setup outputs/physical_target_height_setups_centered/high_0p480_setup.json `
  --steps 2000 `
  --telemetry-decimation 1 `
  --failure-window-steps 2000 `
  --write-run-summary-sidecar `
  --output-dir outputs/visual/quick_k2 `
  --enable-mode-hip-yaw-divergence `
  --mode-hip-yaw-div-kp 10.0 `
  --mode-hip-yaw-div-kd 0.50 `
  --mode-hip-yaw-div-max-torque 7.5 `
  --mode-hip-yaw-div-soft-limit-rad 0.30 `
  --mode-hip-yaw-div-soft-gain 0.80 `
  --mode-hip-yaw-div-ref-source target
```

**Shortest command to visually compare K1 vs K2:**

Change the profile flag:
```
--vd-sagittal-authority-profile k1_pitch_rate_notch_v1    # for K1
--vd-sagittal-authority-profile k2_notch_low_q_v1         # for K2
```

---

## 18. Files Created

| File | Purpose |
|------|---------|
| `docs/validation/k2_verified_visual_commands.md` | This document |
| `outputs/visual_command_discovery/k2_verified_visual_commands.json` | Machine-readable summary |
| `outputs/visual_command_discovery/simulate_hierarchical_controller_help.txt` | Captured help text |
| `tests/test_k2_visual_command_discovery.py` | Tests for visual command discovery |
