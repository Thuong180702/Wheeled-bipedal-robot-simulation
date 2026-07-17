"""Run all validation suites in sequence and produce final reports."""
import subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "mode_hip_yaw_div_full_real_validation"
LOG = OUT / "_runner_log.txt"

def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")

def run(cmd, desc):
    log(f"Starting: {desc}")
    log(f"  {cmd}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=7200)
    elapsed = time.time() - t0
    if result.returncode == 0:
        log(f"  DONE ({elapsed:.0f}s)")
    else:
        log(f"  FAILED rc={result.returncode} ({elapsed:.0f}s)")
        log(f"  stderr: {result.stderr[-500:]}")
    return result.returncode

# Step 1: Step E A/B/C at 2000 steps
rc = run([
    sys.executable, "scripts/run_mode_hip_yaw_div_full_real_validation.py",
    "--suite", "step_e", "--profiles", "A,B,C", "--target-steps", "2000"
], "Step E A/B/C (2000 steps)")
if rc != 0:
    log("Step E A/B/C failed, continuing anyway")

# Step 2: Step C standard for A/B/C/D at 2000 steps
rc = run([
    sys.executable, "scripts/run_mode_hip_yaw_div_full_real_validation.py",
    "--suite", "step_c", "--profiles", "A,B,C,D", "--target-steps", "2000"
], "Step C A/B/C/D (2000 steps)")
if rc != 0:
    log("Step C failed, continuing anyway")

# Step 3: Step D standard for A/B/C/D at 1000 steps
rc = run([
    sys.executable, "scripts/run_mode_hip_yaw_div_full_real_validation.py",
    "--suite", "step_d", "--profiles", "A,B,C,D"
], "Step D A/B/C/D")
if rc != 0:
    log("Step D failed, continuing anyway")

# Step 4: D4/D5 focused 1000 steps for A/B/C/D
rc = run([
    sys.executable, "scripts/run_mode_hip_yaw_div_full_real_validation.py",
    "--suite", "d4_d5_1000", "--profiles", "A,B,C,D"
], "D4/D5 focused 1000 steps")
if rc != 0:
    log("D4/D5 failed, continuing anyway")

# Step 5: Generate summary
rc = run([
    sys.executable, "scripts/run_mode_hip_yaw_div_full_real_validation.py",
    "--suite", "summary"
], "Summary generation")
if rc != 0:
    log("Summary failed")

log("ALL SUITES COMPLETE")
print(f"\nCheck {LOG} for details")
