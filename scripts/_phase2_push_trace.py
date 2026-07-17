"""Phase 2: Push first-divergence instrumented trace."""
import subprocess, sys, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "k2_jax_phase2_push_trace"
OUT.mkdir(parents=True, exist_ok=True)

# Create push sequence file
push_seq = OUT / "push_fwd_90N_trace.json"
import json
with open(push_seq, "w") as f:
    json.dump([[50, 0.0, 90.0, 5]], f)

# Run with full output, capturing SYNCED lines
cmd = [
    sys.executable, str(ROOT / "scripts" / "simulate_hierarchical_controller.py"),
    "--controller-mode", "balance-core",
    "--sagittal-controller", "velocity-damped",
    "--vd-sagittal-authority-profile", "k2_notch_low_q_v1",
    "--controller-backend", "both-synced",
    "--wbc-quiet",
    "--height-variant-setup", str(ROOT / "outputs" / "physical_target_height_setups_centered" / "high_0p480_setup.json"),
    "--steps", "120",
    "--push-sequence-file", str(push_seq),
]
print(f"Running: {' '.join(cmd)}")

r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=300)
stdout = r.stdout

# Save full output
with open(OUT / "full_output.txt", "w", encoding="utf-8") as f:
    f.write(stdout)

# Find the worst diff step
worst_step = 0
worst_val = 0.0
for line in stdout.splitlines():
    m = re.search(r'Worst max_abs_diff:\s*([\d.e+\-]+)', line)
    if m:
        worst_val = float(m.group(1))
    m = re.search(r'at step\s+(\d+)', line)
    if m:
        worst_step = int(m.group(1))
print(f"\nWorst diff: {worst_val:.6e} at step {worst_step}")

# Extract the step-by-step DIFF for wheels [4,9]
print("\n=== STEP-BY-STEP WHEEL DIFFS ===")
for step_num in range(0, 120):
    # Find the SYNCED line for this step
    pattern = f"[SYNCED@{step_num}] max_abs_diff="
    in_synced = False
    diffs = None
    for i, line in enumerate(stdout.splitlines()):
        if pattern in line:
            in_synced = True
            continue
        if in_synced and "DIFF=" in line:
            # Parse diff values
            m = re.search(r'DIFF=\s+\[(.+)\]', line)
            if m:
                vals = [float(v.strip()) for v in m.group(1).split(",")]
                if len(vals) >= 10:
                    diffs = vals
            in_synced = False
    if diffs and (abs(diffs[4]) > 1e-6 or abs(diffs[9]) > 1e-6):
        print(f"  Step {step_num}: wheel[4]={diffs[4]:.6e} wheel[9]={diffs[9]:.6e}")
        # Also print the DIVERGENT steps (step -20 to step + 5)
        # Find surrounding lines for context
        for j, l in enumerate(stdout.splitlines()):
            if f"[SYNCED@{step_num}]" in l:
                # Print sag terms + DIFF + APCR1ND for this step
                for k in range(j, min(j + 15, len(stdout.splitlines()))):
                    sl = stdout.splitlines()[k]
                    if any(tag in sl for tag in ["DIFF=", "SAG_TERMS:", "APCR1ND:"]):
                        print(f"    {sl.strip()}")
                break

# Find the first-divergence step
print("\n=== FIRST DIVERGENT STEP ===")
for step_num in range(50, 75):
    pattern = f"[SYNCED@{step_num}] max_abs_diff="
    in_synced = False
    max_diff = None
    for i, line in enumerate(stdout.splitlines()):
        if pattern in line:
            m = re.search(r'max_abs_diff=([\d.e+\-]+)', line)
            if m:
                max_diff = float(m.group(1))
            # Also print sag terms
            for k in range(i, min(i + 15, len(stdout.splitlines()))):
                sl = stdout.splitlines()[k]
                if "SAG_TERMS:" in sl and "tau_wheel" in sl:
                    print(f"  Step {step_num} (max_diff={max_diff}): {sl.strip()}")
                if "DIFF=" in sl:
                    print(f"  Step {step_num} DIFF: {sl.strip()}")
                if "APCR1ND:" in sl:
                    print(f"  Step {step_num}: {sl.strip()}")
            break

print(f"\nFull output saved to: {OUT / 'full_output.txt'}")
