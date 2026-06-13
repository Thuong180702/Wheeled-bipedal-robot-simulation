# Visual Viewer Troubleshooting Guide

**Last Updated:** 2026-06-05  
**Audience:** Developers encountering MuJoCo live viewer failures

---

## The Problem

MuJoCo live viewer (launched with `--visual` flag) fails to start with:

```
GLFWError: (65542) b'WGL: The driver does not appear to support OpenGL'
ERROR: could not create window
```

Or similar variants:
- `glfw.create_window() returned None`
- `OpenGL context creation failed`
- `WGL: Failed to make context current`

---

## What This Means

MuJoCo's viewer requires GLFW (a graphics library) to create an OpenGL window for 3D rendering. This error means:

1. **GLFW can initialize** (the library loads)
2. **GLFW cannot create an OpenGL context** (GPU driver issue)

This is an **environment/driver problem**, NOT a controller bug or simulation issue. The simulation code itself works fine - only the visual rendering component is affected.

---

## Root Causes

### 1. Missing or Outdated GPU Driver

**Most Common Cause**

- System is using "Microsoft Basic Display Adapter" (software fallback)
- GPU driver is outdated and doesn't support OpenGL 3.3+
- GPU driver got corrupted during Windows update

**How to Check:**

Windows Device Manager:
```
1. Win + X → Device Manager
2. Expand "Display adapters"
3. Look for your GPU name
```

If you see:
- ✅ "NVIDIA GeForce [model]" or "AMD Radeon [model]" - GPU detected
- ❌ "Microsoft Basic Display Adapter" - no driver installed
- ⚠️  Yellow exclamation mark - driver problem

Command line check:
```bash
wmic path win32_VideoController get name,driverversion
```

Expected output shows your actual GPU (NVIDIA/AMD/Intel), not "Microsoft Basic Display Adapter".

**Solution:**

Update GPU driver from manufacturer:

- **NVIDIA:** https://www.nvidia.com/drivers
  - Use "Game Ready Driver" or "Studio Driver"
  - Select your GPU model
  - Install and restart

- **AMD:** https://www.amd.com/support
  - Use "Adrenalin" driver package
  - Select your GPU series
  - Install and restart

- **Intel:** https://www.intel.com/content/www/us/en/download-center
  - Search for "Intel Graphics Driver"
  - Select your processor generation
  - Install and restart

After installation:
1. Restart computer
2. Verify Device Manager shows proper GPU
3. Re-run diagnostic: `python scripts/check_visual_environment.py`

---

### 2. Remote Desktop / RDP Session

**Second Most Common Cause**

Windows Remote Desktop (RDP) disables hardware-accelerated OpenGL by default. GLFW cannot create OpenGL contexts over RDP even if drivers are correct.

**How to Check:**

```bash
echo $SESSIONNAME
```

If output starts with `RDP-`, you're in Remote Desktop session.

**Solutions:**

**Option A: Use physical console**
- Log in directly at the machine (not remotely)
- Or use console redirection if available

**Option B: Use alternative remote desktop**
- VNC (TightVNC, RealVNC)
- TeamViewer
- Parsec (gaming-focused, good OpenGL support)
- Chrome Remote Desktop

**Option C: Use offline rendering fallback**
- Generate telemetry-based animations instead of live viewer
- See "Offline Alternatives" section below

---

### 3. Laptop Hybrid Graphics

**Common on Laptops**

Laptop has both integrated GPU (Intel) and dedicated GPU (NVIDIA/AMD), but Python uses the low-power integrated GPU which may have limited OpenGL support.

**How to Check:**

Look for two GPUs in Device Manager:
- Intel UHD Graphics (integrated)
- NVIDIA GeForce MX / GTX / RTX (dedicated)

**Solution - Windows 10/11:**

```
1. Settings → System → Display → Graphics settings
2. Click "Browse" and add Python executable:
   C:\Users\[YourName]\AppData\Local\Programs\Python\Python310\python.exe
3. Click "Options" on the Python entry
4. Select "High performance" (forces dedicated GPU)
5. Restart terminal/IDE
6. Re-run simulation
```

Alternative via NVIDIA Control Panel:
```
1. Right-click desktop → NVIDIA Control Panel
2. Manage 3D Settings → Program Settings
3. Add python.exe
4. Set "OpenGL rendering GPU" to your NVIDIA GPU
```

---

### 4. WSL / Headless Environment

**For WSL Users**

WSL1/WSL2 do not have native OpenGL window support. GLFW can run but cannot create windows without X server or Wayland.

**Solutions:**

- Install and configure X server (VcXsrv, Xming)
- Use WSLg (WSL2 with GUI support, Windows 11 only)
- Run simulations on Windows native Python instead of WSL
- Use offline rendering fallback

---

### 5. Virtualization / VM

**For Virtual Machines**

VirtualBox, VMware, Hyper-V may not expose GPU to guest OS or may have limited 3D acceleration.

**Solutions:**

- Enable 3D acceleration in VM settings
- Install guest additions/tools
- Run on host OS instead of VM
- Use offline rendering fallback

---

## Diagnostic Workflow

Run the automated diagnostic:

```bash
python scripts/check_visual_environment.py
```

This checks:
- Python/MuJoCo/GLFW versions
- GLFW initialization
- OpenGL window creation
- GPU adapters (Windows)
- Remote Desktop detection

**Output files:**
- `outputs/visual_environment_check/visual_environment_check.json` (machine-readable)
- `outputs/visual_environment_check/visual_environment_check_report.md` (human-readable)

**Interpret results:**

```
[OK] VISUAL MODE SUPPORTED
→ Live viewer should work. If it still fails, file a bug report.

[FAIL] VISUAL MODE NOT SUPPORTED
→ Follow troubleshooting steps for your detected error type.
```

---

## Manual Testing

**Test GLFW directly:**

```bash
python -c "import glfw; print('GLFW init:', glfw.init()); w=glfw.create_window(640,480,'test',None,None); print('Window:', w); glfw.terminate()"
```

Expected output (working):
```
GLFW init: True
Window: <glfw.LP__GLFWwindow object at 0x...>
```

Failure output:
```
GLFWError: (65542) b'WGL: The driver does not appear to support OpenGL'
GLFW init: True
Window: None
```

If window is `None`, OpenGL context creation failed.

**Check OpenGL version:**

If window creation works, check OpenGL version:

```python
import glfw
from OpenGL.GL import *

glfw.init()
window = glfw.create_window(640, 480, "test", None, None)
glfw.make_context_current(window)
print("OpenGL version:", glGetString(GL_VERSION).decode())
print("Renderer:", glGetString(GL_RENDERER).decode())
glfw.terminate()
```

MuJoCo requires OpenGL 3.3+. If version is lower, update GPU driver.

---

## Offline Alternatives

If live viewer cannot be fixed, use offline rendering:

### Option 1: Telemetry-Based Replay

Already implemented:

```bash
python scripts/render_low_0p300_visual_replay.py
```

**Generates:**
- GIF animations of pitch/hip-yaw/support over time
- PNG summary plots with peak frames
- Does NOT require OpenGL

**Output:**
- `outputs/visual_inspection_low_0p300/videos/*.gif`
- `outputs/visual_inspection_low_0p300/replay_fallback/*.png`

### Option 2: Save and Inspect Telemetry

Run simulation without `--visual`:

```bash
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --height-variant-setup outputs/physical_target_height_setups/low_0p300_setup.json \
  --vd-sagittal-authority-profile J2 \
  --steps 1000
```

Telemetry saved to: `outputs/hierarchical_controller_sim/telemetry_[timestamp].csv`

Analyze with pandas/matplotlib or Excel.

### Option 3: Transfer to Machine with GPU

1. Run simulation headless (no --visual)
2. Copy telemetry CSV to machine with working OpenGL
3. Generate visualizations there

---

## Example Commands

**Run diagnostic:**
```bash
python scripts/check_visual_environment.py
```

**Test live viewer (balance-core mode, J2 profile):**
```bash
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --height-variant-setup outputs/physical_target_height_setups/low_0p300_setup.json \
  --vd-sagittal-authority-profile J2 \
  --steps 1000 \
  --visual
```

**Generate offline replay:**
```bash
python scripts/render_low_0p300_visual_replay.py
```

---

## Still Not Working?

If you've tried all troubleshooting steps and live viewer still fails:

1. **Verify diagnostic result:**
   ```bash
   python scripts/check_visual_environment.py
   ```
   
2. **Check GPU in Device Manager:**
   - Not "Microsoft Basic Display Adapter"
   - No yellow exclamation marks
   
3. **Verify you restarted after driver install**

4. **Check MuJoCo version compatibility:**
   ```bash
   python -c "import mujoco; print(mujoco.__version__)"
   ```
   Should be >= 3.0.0

5. **File a bug report:**
   - Include `visual_environment_check_report.md`
   - Include full error message
   - Include GPU model and driver version
   - Specify: Windows version, Python version, how you're accessing machine (local/RDP/VNC)

---

## Summary

| Problem | Solution |
|---------|----------|
| Microsoft Basic Display Adapter | Install GPU driver from manufacturer |
| Outdated driver | Update GPU driver, restart |
| Remote Desktop | Use VNC/TeamViewer or physical console |
| Laptop using integrated GPU | Force dedicated GPU in Graphics Settings |
| WSL/VM | Use native Windows Python or offline rendering |
| Still broken | Use offline rendering fallback |

**Key Point:** This is an environment issue, NOT a simulation bug. The controller/physics work fine - only 3D rendering is affected.
