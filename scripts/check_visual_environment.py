#!/usr/bin/env python3
"""
Visual Environment Diagnostic Check

Checks whether the current environment can support MuJoCo live visual viewer.
Tests GLFW/OpenGL window creation and reports environment details.
"""

import sys
import json
import platform
import subprocess
from pathlib import Path
from typing import Dict, Any, List


OUTPUT_DIR = Path("outputs/visual_environment_check")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def check_python_version() -> Dict[str, Any]:
    """Check Python version."""
    return {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "architecture": platform.architecture(),
    }


def check_mujoco_import() -> Dict[str, Any]:
    """Check if MuJoCo can be imported."""
    result = {"available": False, "version": None, "error": None}
    try:
        import mujoco
        result["available"] = True
        result["version"] = mujoco.__version__
    except ImportError as e:
        result["error"] = str(e)
    return result


def check_glfw_import() -> Dict[str, Any]:
    """Check if GLFW can be imported."""
    result = {"available": False, "version": None, "error": None}
    try:
        import glfw
        result["available"] = True
        result["version"] = getattr(glfw, "__version__", "unknown")
    except ImportError as e:
        result["error"] = str(e)
    return result


def check_glfw_init() -> Dict[str, Any]:
    """Test GLFW initialization."""
    result = {"success": False, "error": None}
    try:
        import glfw
        if glfw.init():
            result["success"] = True
            glfw.terminate()
        else:
            result["error"] = "glfw.init() returned False"
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {str(e)}"
    return result


def check_glfw_window_creation() -> Dict[str, Any]:
    """Test GLFW window creation (the critical test)."""
    result = {
        "success": False,
        "error": None,
        "error_type": None,
        "window_created": False,
    }

    try:
        import glfw

        if not glfw.init():
            result["error"] = "glfw.init() failed"
            result["error_type"] = "GLFW_INIT_FAILED"
            return result

        # Try to create a window
        window = glfw.create_window(640, 480, "GLFW Test Window", None, None)

        if window:
            result["success"] = True
            result["window_created"] = True
            glfw.destroy_window(window)
        else:
            result["error"] = "glfw.create_window() returned None"
            result["error_type"] = "WINDOW_CREATION_FAILED"

        glfw.terminate()

    except Exception as e:
        error_str = str(e)
        result["error"] = f"{type(e).__name__}: {error_str}"

        # Classify error type
        if "WGL" in error_str or "OpenGL" in error_str:
            result["error_type"] = "OPENGL_DRIVER_ERROR"
        elif "GLFW" in error_str:
            result["error_type"] = "GLFW_ERROR"
        else:
            result["error_type"] = "UNKNOWN_ERROR"

    return result


def check_remote_desktop_windows() -> Dict[str, Any]:
    """Check if running under Remote Desktop (Windows only)."""
    result = {"remote_desktop_detected": None, "method": None}

    if platform.system() != "Windows":
        return result

    try:
        # Check SESSIONNAME environment variable
        session_name = subprocess.check_output(
            ["powershell", "-Command", "$env:SESSIONNAME"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5
        ).strip()

        result["method"] = "SESSIONNAME"
        result["remote_desktop_detected"] = session_name.startswith("RDP-")

    except Exception:
        pass

    return result


def check_gpu_adapters_windows() -> Dict[str, Any]:
    """Get GPU adapter info (Windows only)."""
    result = {"adapters": [], "method": None, "error": None}

    if platform.system() != "Windows":
        return result

    # Try wmic
    try:
        output = subprocess.check_output(
            ["wmic", "path", "win32_VideoController", "get", "name,driverversion"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10
        )

        lines = [line.strip() for line in output.split("\n") if line.strip()]
        if len(lines) > 1:  # Skip header
            for line in lines[1:]:
                if line:
                    result["adapters"].append(line)
            result["method"] = "wmic"

    except Exception as e:
        result["error"] = f"wmic failed: {str(e)}"

    # If wmic failed, try PowerShell
    if not result["adapters"]:
        try:
            output = subprocess.check_output(
                ["powershell", "-Command",
                 "Get-WmiObject Win32_VideoController | Select-Object Name,DriverVersion | Format-List"],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=10
            )

            result["adapters"].append(output.strip())
            result["method"] = "powershell"
            result["error"] = None

        except Exception as e:
            if not result["error"]:
                result["error"] = f"powershell failed: {str(e)}"

    return result


def generate_report_text(checks: Dict[str, Any]) -> str:
    """Generate human-readable report."""
    lines = [
        "# Visual Environment Diagnostic Report",
        "",
        "**Date:** " + checks["timestamp"],
        "**Platform:** " + checks["python"]["platform"],
        "",
        "## Summary",
        "",
    ]

    # Overall status
    window_check = checks["glfw_window_creation"]
    if window_check["success"]:
        lines.extend([
            "**Status:** [OK] VISUAL MODE SUPPORTED",
            "",
            "GLFW can create OpenGL windows. Live MuJoCo viewer should work.",
            "",
        ])
    else:
        lines.extend([
            "**Status:** [FAIL] VISUAL MODE NOT SUPPORTED",
            "",
            f"**Error:** {window_check['error']}",
            f"**Error Type:** {window_check['error_type']}",
            "",
            "Live MuJoCo viewer will not work in this environment.",
            "",
        ])

    # Detailed checks
    lines.extend([
        "## Detailed Checks",
        "",
        "### Python Environment",
        "",
        f"- Python version: {checks['python']['python_version'].split()[0]}",
        f"- Platform: {checks['python']['platform']}",
        f"- Architecture: {checks['python']['architecture'][0]}",
        "",
        "### MuJoCo",
        "",
    ])

    mujoco = checks["mujoco"]
    if mujoco["available"]:
        lines.append(f"[OK] MuJoCo installed (version {mujoco['version']})")
    else:
        lines.append(f"[FAIL] MuJoCo not available: {mujoco['error']}")

    lines.extend(["", "### GLFW", ""])

    glfw = checks["glfw"]
    if glfw["available"]:
        lines.append(f"[OK] GLFW installed (version {glfw['version']})")
    else:
        lines.append(f"[FAIL] GLFW not available: {glfw['error']}")
        lines.extend(["", "**Cannot proceed with visual mode checks without GLFW.**", ""])
        return "\n".join(lines)

    lines.append("")

    glfw_init = checks["glfw_init"]
    if glfw_init["success"]:
        lines.append("[OK] GLFW initialization successful")
    else:
        lines.append(f"[FAIL] GLFW initialization failed: {glfw_init['error']}")

    lines.append("")

    if window_check["success"]:
        lines.append("[OK] GLFW window creation successful")
    else:
        lines.append(f"[FAIL] GLFW window creation failed: {window_check['error']}")

    lines.extend(["", "### Platform-Specific Checks", ""])

    if platform.system() == "Windows":
        # Remote Desktop check
        rd = checks.get("remote_desktop", {})
        if rd.get("remote_desktop_detected") is True:
            lines.append("[WARN] Remote Desktop detected - may disable OpenGL acceleration")
        elif rd.get("remote_desktop_detected") is False:
            lines.append("[OK] Not running under Remote Desktop")

        lines.append("")

        # GPU adapters
        gpu = checks.get("gpu_adapters", {})
        if gpu.get("adapters"):
            lines.append("**GPU Adapters:**")
            lines.append("```")
            for adapter in gpu["adapters"]:
                lines.append(adapter)
            lines.append("```")

            # Check for Microsoft Basic Display Adapter
            adapters_str = " ".join(gpu["adapters"]).lower()
            if "microsoft basic display adapter" in adapters_str:
                lines.extend([
                    "",
                    "[WARN] **WARNING:** Microsoft Basic Display Adapter detected.",
                    "This is a fallback driver with no OpenGL support.",
                    "Install proper GPU driver from manufacturer.",
                    "",
                ])
        else:
            lines.append("[WARN] Could not detect GPU adapters")
            if gpu.get("error"):
                lines.append(f"   Error: {gpu['error']}")

    # Troubleshooting
    if not window_check["success"]:
        lines.extend([
            "",
            "## Troubleshooting",
            "",
        ])

        if window_check["error_type"] == "OPENGL_DRIVER_ERROR":
            lines.extend([
                "**OpenGL/WGL Error Detected**",
                "",
                "This usually means:",
                "1. GPU driver is missing, outdated, or corrupted",
                "2. Using Microsoft Basic Display Adapter (fallback driver)",
                "3. Running over Remote Desktop with OpenGL disabled",
                "",
                "**Solutions:**",
                "",
                "1. **Update GPU driver** from manufacturer:",
                "   - NVIDIA: https://www.nvidia.com/drivers",
                "   - AMD: https://www.amd.com/support",
                "   - Intel: https://www.intel.com/content/www/us/en/download-center",
                "",
                "2. **Check Device Manager:**",
                "   - Open Device Manager -> Display adapters",
                "   - If you see 'Microsoft Basic Display Adapter', proper driver not installed",
                "   - Uninstall and reinstall GPU driver",
                "",
                "3. **Avoid Remote Desktop:**",
                "   - Remote Desktop often disables OpenGL acceleration",
                "   - Use VNC, TeamViewer, or physical console instead",
                "",
                "4. **Restart after driver install:**",
                "   - Some driver updates require system restart",
                "",
                "5. **Force dedicated GPU (laptops):**",
                "   - Open Graphics Settings",
                "   - Add Python executable",
                "   - Set to 'High performance' (dedicated GPU)",
                "",
            ])

        lines.extend([
            "## Alternative: Offline Visual Replay",
            "",
            "If live viewer cannot be fixed, use offline rendering:",
            "",
            "```bash",
            "python scripts/render_low_0p300_visual_replay.py",
            "```",
            "",
            "This generates telemetry-based animations and plots without requiring OpenGL.",
            "",
        ])

    lines.extend([
        "## Test Commands",
        "",
        "**Test GLFW manually:**",
        "```bash",
        "python -c \"import glfw; print('GLFW init:', glfw.init()); "
        "w=glfw.create_window(640,480,'test',None,None); print('Window:', w); glfw.terminate()\"",
        "```",
        "",
        "**Run MuJoCo viewer:**",
        "```bash",
        "python scripts/simulate_hierarchical_controller.py \\",
        "  --controller-mode balance-core \\",
        "  --sagittal-controller velocity-damped \\",
        "  --height-variant-setup outputs/physical_target_height_setups/low_0p300_setup.json \\",
        "  --vd-sagittal-authority-profile J2 \\",
        "  --steps 1000 \\",
        "  --visual",
        "```",
        "",
    ])

    return "\n".join(lines)


def main():
    """Run all diagnostic checks and generate report."""
    print("="*80)
    print("Visual Environment Diagnostic Check")
    print("="*80)
    print()

    import datetime

    checks = {
        "timestamp": datetime.datetime.now().isoformat(),
        "python": check_python_version(),
        "mujoco": check_mujoco_import(),
        "glfw": check_glfw_import(),
    }

    print("[1/6] Checking Python environment...")
    print(f"      Python {checks['python']['python_version'].split()[0]}")

    print("[2/6] Checking MuJoCo...")
    if checks["mujoco"]["available"]:
        print(f"      [OK] MuJoCo {checks['mujoco']['version']}")
    else:
        print(f"      [FAIL] MuJoCo not available")

    print("[3/6] Checking GLFW...")
    if checks["glfw"]["available"]:
        print(f"      [OK] GLFW {checks['glfw']['version']}")
    else:
        print(f"      [FAIL] GLFW not available")
        print()
        print("Cannot proceed with visual checks without GLFW.")
        return 1

    print("[4/6] Testing GLFW initialization...")
    checks["glfw_init"] = check_glfw_init()
    if checks["glfw_init"]["success"]:
        print("      [OK] GLFW init successful")
    else:
        print(f"      [FAIL] GLFW init failed: {checks['glfw_init']['error']}")

    print("[5/6] Testing GLFW window creation...")
    checks["glfw_window_creation"] = check_glfw_window_creation()
    if checks["glfw_window_creation"]["success"]:
        print("      [OK] Window creation successful")
    else:
        print(f"      [FAIL] Window creation failed")
        print(f"      Error: {checks['glfw_window_creation']['error']}")

    print("[6/6] Platform-specific checks...")
    if platform.system() == "Windows":
        checks["remote_desktop"] = check_remote_desktop_windows()
        checks["gpu_adapters"] = check_gpu_adapters_windows()
        print("      Detected Windows - checking GPU adapters and Remote Desktop...")

    # Generate reports
    print()
    print("="*80)
    print("Generating reports...")
    print("="*80)

    # JSON report
    json_path = OUTPUT_DIR / "visual_environment_check.json"
    with open(json_path, 'w') as f:
        json.dump(checks, f, indent=2)
    print(f"[OK] JSON report: {json_path}")

    # Markdown report
    report_text = generate_report_text(checks)
    md_path = OUTPUT_DIR / "visual_environment_check_report.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"[OK] Markdown report: {md_path}")

    # Final verdict
    print()
    print("="*80)
    print("RESULT")
    print("="*80)
    print()

    if checks["glfw_window_creation"]["success"]:
        print("[OK] VISUAL MODE SUPPORTED")
        print()
        print("OpenGL window creation successful. Live MuJoCo viewer should work.")
        return 0
    else:
        print("[FAIL] VISUAL MODE NOT SUPPORTED")
        print()
        print("OpenGL window creation failed. This is an environment/driver issue,")
        print("not a controller issue.")
        print()
        print("Error:", checks["glfw_window_creation"]["error"])
        print()
        print("Solutions:")
        print("- Update GPU driver from manufacturer website")
        print("- Avoid Remote Desktop (disables OpenGL)")
        print("- Ensure not using 'Microsoft Basic Display Adapter'")
        print("- Run on machine with GPU/OpenGL support")
        print("- Use offline rendering fallback:")
        print("  python scripts/render_low_0p300_visual_replay.py")
        print()
        print(f"See detailed report: {md_path}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
