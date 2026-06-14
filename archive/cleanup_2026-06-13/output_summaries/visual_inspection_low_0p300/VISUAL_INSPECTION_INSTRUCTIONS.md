# Visual Inspection Instructions - low_0p300

**Date:** 2026-06-05  
**Status:** ✅ VISUAL ARTIFACTS GENERATED - AWAITING USER INSPECTION  
**Decision Required:** Pitch threshold acceptance for z=0.300m boundary height

---

## ✅ Successfully Generated Visual Artifacts

### Telemetry-Based Animations (GIF)
Located in `outputs/visual_inspection_low_0p300/videos/`:

- **baseline_low_0p300_telemetry.gif** (7.4 MB)
- **J2_low_0p300_telemetry.gif** (8.3 MB)
- **J3_low_0p300_telemetry.gif** (8.3 MB)

Each GIF shows animated plots of:
- Pitch excursion over time (with 5.7° gate threshold)
- Hip yaw excursion over time (with 4.0° gate threshold)
- Support position error over time (with 0.15m gate threshold)
- CoM height tracking over time (with 0.300m target)

### Static Summary Plots (PNG)
Located in `outputs/visual_inspection_low_0p300/replay_fallback/`:

- **baseline_low_0p300_summary.png** (200 KB)
- **J2_low_0p300_summary.png** (262 KB)
- **J3_low_0p300_summary.png** (255 KB)

Each PNG shows:
- Pitch over time with peak marked
- Hip yaw over time with peak marked
- Support error over time with peak marked
- Summary metrics table with pass/fail status

### Telemetry Data (CSV)
Located in `outputs/visual_inspection_low_0p300/`:

- **baseline_telemetry.csv** (8.9 MB, 1000 steps)
- **J2_telemetry.csv** (8.9 MB, 1000 steps)
- **J3_telemetry.csv** (8.9 MB, 1000 steps)

**Note:** MP4 video generation unavailable (requires ffmpeg). GIF animations provide equivalent visualization.

---

## 📊 Numerical Summary

| Profile | Support | Hip Yaw | Pitch | Overall |
|---------|---------|---------|-------|---------|
| **Baseline** | 0.243 m ❌ | 12.2° ❌ | 5.5° ✅ | 1/3 gates |
| **J2** | 0.114 m ✅ | 7.8° ❌ | 9.0° ❌ | 1/3 gates |
| **J3** | 0.125 m ✅ | 5.1° ❌ | 8.7° ❌ | 1/3 gates |

**Gate Thresholds:**
- Support position error ≤ 0.15 m
- Hip yaw max ≤ 0.07 rad (4.0°)
- Pitch max ≤ 0.10 rad (5.7°)

**Key Finding:**
- **Baseline:** Pitch safe but support/hip-yaw fail badly
- **J2/J3:** Support passes, hip-yaw close, but pitch exceeds threshold by ~50-60%
- **J2/J3 improvements:** 50%+ better support, 36-59% better hip-yaw vs baseline
- **J2/J3 trade-off:** Pitch increases from 5.5° → 8.7-9.0°

---

## 🎯 What You Need to Decide

After reviewing the GIF animations and summary plots, decide:

### Option A: Accept J2 or J3 with relaxed pitch threshold
- Relax pitch threshold to 0.14-0.15 rad (8.0-8.6°) for low_0p300 only
- Accept that extreme boundary height (z=0.300m) has reduced pitch margin
- Proceed with J2 or J3 as the low-height solution
- **Rationale:** Support and hip-yaw improvements are substantial, pitch looks stable in visual inspection

### Option B: Implement pitch-aware position control (Option C)
- Keep pitch threshold at 0.10 rad (5.7°)
- Implement pitch-aware scaling to reduce position authority when pitch approaches threshold
- Create and evaluate P0-P5 candidate profiles
- **Rationale:** Preserve strict pitch gate, attempt to break position-pitch coupling

### Option C: Mark z=0.300m as outside operational envelope
- Define operational z_min = 0.330m
- Mark z=0.300m as physical boundary (achievable but degraded)
- **Rationale:** 0.300m is 25% below nominal height, at kinematic limit

---

## 🖥️ Manual Visual Inspection Commands

If you want to see live 3D simulation with OpenGL viewer (requires machine with GPU/OpenGL support):

### Baseline (reference - pitch safe, support/yaw fail)
```bash
cd f:/ROBOTCUATAO/Wheeled-bipedal-robot-simulation

python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --height-variant-setup outputs/physical_target_height_setups/low_0p300_setup.json \
  --vd-sagittal-authority-profile baseline \
  --steps 1000 \
  --visual
```

### J2 (support/yaw better, pitch 9.0°)
```bash
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --height-variant-setup outputs/physical_target_height_setups/low_0p300_setup.json \
  --vd-sagittal-authority-profile J2 \
  --steps 1000 \
  --visual
```

### J3 (best support, yaw better, pitch 8.7°)
```bash
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --height-variant-setup outputs/physical_target_height_setups/low_0p300_setup.json \
  --vd-sagittal-authority-profile J3 \
  --steps 1000 \
  --visual
```

---

## 🔧 OpenGL Troubleshooting

If you see: `GLFWError: WGL: The driver does not appear to support OpenGL`

### 1. Update GPU Driver
- **NVIDIA:** [nvidia.com/drivers](https://www.nvidia.com/drivers)
- **AMD:** [amd.com/support](https://www.amd.com/support)
- **Intel:** [intel.com/download-center](https://www.intel.com/content/www/us/en/download-center)

### 2. Check Display Adapter
- Open Device Manager -> Display adapters
- If you see "Microsoft Basic Display Adapter", GPU driver is not installed
- Need proper GPU driver for OpenGL acceleration

### 3. Avoid Remote Desktop
- Remote Desktop disables OpenGL acceleration
- Use VNC, TeamViewer, or physical console instead
- WSL/headless environments may not support OpenGL

### 4. Verify OpenGL Version
- MuJoCo requires OpenGL 3.3+
- Windows: Use GPU-Z to check OpenGL version
- If OpenGL < 3.3, update GPU driver or use different machine

### 5. Alternative: Use Generated Artifacts
- If OpenGL unavailable, review the GIF animations and PNG summary plots
- Telemetry data is identical to what live viewer would show
- Only missing: 3D robot pose visualization

---

## 🚫 What's Currently Blocked

Until you complete visual inspection and make a decision:

❌ **Do NOT modify controller code**  
❌ **Do NOT change pitch threshold in code**  
❌ **Do NOT implement pitch-aware candidates yet**  
❌ **Do NOT proceed to full Phase 6 evaluation**  
❌ **Do NOT tune any parameters**  

**Next step depends on your decision:**
- **If Option A:** Update pitch threshold, document acceptance rationale, proceed to Phase 6
- **If Option B:** Implement pitch-aware scaling (Task 4-7 from prior protocol)
- **If Option C:** Document operational envelope boundary, mark low_0p300 as physical limit

---

## 📁 Generated Files Summary

```
outputs/visual_inspection_low_0p300/
├── videos/
│   ├── baseline_low_0p300_telemetry.gif    (7.4 MB) ✅
│   ├── J2_low_0p300_telemetry.gif          (8.3 MB) ✅
│   └── J3_low_0p300_telemetry.gif          (8.3 MB) ✅
├── replay_fallback/
│   ├── baseline_low_0p300_summary.png      (200 KB) ✅
│   ├── J2_low_0p300_summary.png            (262 KB) ✅
│   └── J3_low_0p300_summary.png            (255 KB) ✅
├── baseline_telemetry.csv                   (8.9 MB) ✅
├── J2_telemetry.csv                         (8.9 MB) ✅
├── J3_telemetry.csv                         (8.9 MB) ✅
├── visual_inspection_low_0p300_metrics.json ✅
├── visual_inspection_low_0p300_summary.md   ✅
└── VISUAL_INSPECTION_INSTRUCTIONS.md        ✅ (this file)
```

**Scripts:**
- `scripts/render_low_0p300_visual_replay.py` ✅ (reusable for future visualizations)

---

## 🎬 How to Review

1. **Open GIF animations** in `outputs/visual_inspection_low_0p300/videos/`
   - Watch baseline first (reference case)
   - Watch J2 and J3
   - Compare pitch excursions visually

2. **Open PNG summaries** in `outputs/visual_inspection_low_0p300/replay_fallback/`
   - Review peak frames and metrics table
   - Compare which profile looks better overall

3. **Decide:** Does pitch 8-9° at z=0.300m look acceptable?
   - If YES → Option A (relax threshold) or Option B (pitch-aware fix)
   - If NO → Option C (mark 0.300m outside operational envelope)

4. **Communicate decision** so work can proceed accordingly

---

**Status:** ✅ All visual artifacts generated successfully. Awaiting user decision on pitch threshold acceptance at low_0p300 boundary height.
