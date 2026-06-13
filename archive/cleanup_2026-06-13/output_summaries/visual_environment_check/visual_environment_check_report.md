# Visual Environment Diagnostic Report

**Date:** 2026-06-05T12:12:39.877327
**Platform:** Windows-10-10.0.26200-SP0

## Summary

**Status:** [FAIL] VISUAL MODE NOT SUPPORTED

**Error:** glfw.create_window() returned None
**Error Type:** WINDOW_CREATION_FAILED

Live MuJoCo viewer will not work in this environment.

## Detailed Checks

### Python Environment

- Python version: 3.10.2
- Platform: Windows-10-10.0.26200-SP0
- Architecture: 64bit

### MuJoCo

[OK] MuJoCo installed (version 3.6.0)

### GLFW

[OK] GLFW installed (version 2.10.0)

[OK] GLFW initialization successful

[FAIL] GLFW window creation failed: glfw.create_window() returned None

### Platform-Specific Checks

[OK] Not running under Remote Desktop

**GPU Adapters:**
```
Name          : NVIDIA GeForce MX250
DriverVersion : 30.0.15.1169

Name          : Intel(R) UHD Graphics 620
DriverVersion : 27.20.100.8681
```

## Troubleshooting

## Alternative: Offline Visual Replay

If live viewer cannot be fixed, use offline rendering:

```bash
python scripts/render_low_0p300_visual_replay.py
```

This generates telemetry-based animations and plots without requiring OpenGL.

## Test Commands

**Test GLFW manually:**
```bash
python -c "import glfw; print('GLFW init:', glfw.init()); w=glfw.create_window(640,480,'test',None,None); print('Window:', w); glfw.terminate()"
```

**Run MuJoCo viewer:**
```bash
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --height-variant-setup outputs/physical_target_height_setups/low_0p300_setup.json \
  --vd-sagittal-authority-profile J2 \
  --steps 1000 \
  --visual
```
