"""Phase 0: Reproduce push_fwd_90N composer failure with detailed diagnostics.

Captures per-step PY/JX torque comparison, composer state, pre-composer sums,
and rate-limit activity for every step from 90 to 160 (push at step 100).
"""

import json, os, re, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "k2_jax_composer_push_fwd_debug"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SETUP_DIR = ROOT / "outputs" / "physical_target_height_setups_centered"

SIM = str(ROOT / "scripts" / "simulate_hierarchical_controller.py")

BASE_CMD = [
    sys.executable, SIM,
    "--controller-mode", "balance-core",
    "--sagittal-controller", "velocity-damped",
    "--vd-sagittal-authority-profile", "k2_notch_low_q_v1",
    "--controller-backend", "both-synced",
    "--wbc-quiet",
    "--enable-mode-hip-yaw-divergence",
    "--mode-hip-yaw-div-kp", "10.0",
    "--mode-hip-yaw-div-kd", "0.50",
    "--mode-hip-yaw-div-max-torque", "7.5",
    "--mode-hip-yaw-div-soft-limit-rad", "0.30",
    "--mode-hip-yaw-div-soft-gain", "0.80",
    "--mode-hip-yaw-div-ref-source", "target",
]


def run_push_fwd_both_synced():
    """Run push_fwd_90N with both-synced backend."""
    push_seq_file = OUT_DIR / "push_fwd_seq.json"
    with open(push_seq_file, "w") as f:
        json.dump([[100, 0.0, 90.0, 5]], f)

    cmd = BASE_CMD + [
        "--height-variant-setup", str(SETUP_DIR / "high_0p480_setup.json"),
        "--steps", "300",
        "--push-sequence-file", str(push_seq_file),
    ]

    print(f"CMD: {' '.join(str(x) for x in cmd)}")
    print(f"Output dir: {OUT_DIR}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        timeout=600,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )

    stdout_path = OUT_DIR / "push_fwd_both_synced_stdout.txt"
    stderr_path = OUT_DIR / "push_fwd_both_synced_stderr.txt"
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")

    print(f"Return code: {result.returncode}")
    print(f"Stdout: {len(result.stdout)} chars -> {stdout_path}")
    print(f"Stderr: {len(result.stderr)} chars -> {stderr_path}")
    return result.stdout, result.stderr, result.returncode


def main():
    print("=" * 70)
    print("PHASE 0: push_fwd_90N COMPOSER FAILURE REPRODUCTION")
    print("=" * 70)

    t0 = time.time()
    stdout, stderr, rc = run_push_fwd_both_synced()
    elapsed = time.time() - t0

    # Parse all SYNCED lines
    synced_data = []
    for line in stdout.splitlines():
        m = re.search(
            r'\[SYNCED@(\d+)\]\s+max_abs_diff=([\d.e+\-]+)\s+'
            r'first_divergent_idx=(\d+)\s+val=([\d.e+\-]+)',
            line
        )
        if m:
            synced_data.append({
                "step": int(m.group(1)),
                "max_abs_diff": float(m.group(2)),
                "divergent_idx": int(m.group(3)),
                "divergent_val": float(m.group(4)),
            })

    # Parse per-actuator differences
    all_diffs = []
    for line in stdout.splitlines():
        if "DIFF=   [" in line:
            m = re.search(r'DIFF=\s+\[(.+)\]', line)
            if m:
                vals = [float(v.strip()) for v in m.group(1).split(",")]
                if len(vals) >= 10:
                    all_diffs.append(vals)

    # Parse summary from stdout
    worst_diff = None
    worst_step = None
    worst_actuator = None
    classification = None
    fell = "without falling" not in stdout

    for line in stdout.splitlines():
        m = re.search(r'Worst max_abs_diff:\s*([\d.e+\-]+)', line)
        if m:
            worst_diff = float(m.group(1))
        m = re.search(r'at step\s+(\d+),\s+actuator index\s+(\d+)', line)
        if m:
            worst_step = int(m.group(1))
            worst_actuator = int(m.group(2))
        m = re.search(r'Classification:\s*(\S+)', line)
        if m:
            classification = m.group(1)

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Elapsed: {elapsed:.0f}s  RC: {rc}")
    print(f"SYNCED lines parsed: {len(synced_data)}")
    print(f"Worst max_abs_diff: {worst_diff}")
    print(f"Worst step: {worst_step}, actuator: {worst_actuator}")
    print(f"Classification: {classification}")
    print(f"Fell: {fell}")
    print(f"Diff vectors parsed: {len(all_diffs)}")

    # Show first steps with divergence
    if synced_data:
        diverged = [d for d in synced_data if d["divergent_val"] > 1e-6]
        if diverged:
            first = diverged[0]
            print(f"\nFirst divergence at step {first['step']}: "
                  f"idx={first['divergent_idx']} val={first['divergent_val']:.6e}")
        else:
            print("\nNo divergence > 1e-6 found!")

        # Show all diverging steps (condensed)
        print(f"\nDiverging steps (>1e-6):")
        for d in diverged[:30]:
            print(f"  step={d['step']:4d}  max_diff={d['max_abs_diff']:.6e}  "
                  f"idx={d['divergent_idx']}  val={d['divergent_val']:.6e}")

        # Show per-actuator details for first 5 diverging steps
        print(f"\nPer-actuator diffs for first diverging steps:")
        for d in diverged[:5]:
            step = d["step"]
            # Find matching diff vector
            for diff_vec in all_diffs:
                # Approximate match — we'd need step context
                pass
            print(f"  step {step}: max={d['max_abs_diff']:.6e} at idx {d['divergent_idx']}")

    # Show detailed SYNCED output for the first divergence step
    if synced_data:
        diverged = [d for d in synced_data if d["divergent_val"] > 1e-6]
        if diverged:
            first_div_step = diverged[0]["step"]
            print(f"\n{'='*70}")
            print(f"DETAILED DIAGNOSTICS AT STEP {first_div_step}")
            print(f"{'='*70}")
            target = f"[SYNCED@{first_div_step}]"
            lines = stdout.splitlines()
            for i, line in enumerate(lines):
                if target in line:
                    for j in range(i, min(i + 45, len(lines))):
                        print(lines[j])
                    break

    # Show worst step details
    if worst_step is not None:
        print(f"\n{'='*70}")
        print(f"DETAILED DIAGNOSTICS AT WORST STEP {worst_step}")
        print(f"{'='*70}")
        target = f"[SYNCED@{worst_step}]"
        lines = stdout.splitlines()
        for i, line in enumerate(lines):
            if target in line:
                for j in range(i, min(i + 45, len(lines))):
                    print(lines[j])
                break

    # Summary JSON
    summary = {
        "elapsed_s": elapsed,
        "returncode": rc,
        "fell": fell,
        "classification": classification,
        "worst_max_abs_diff": worst_diff,
        "worst_step": worst_step,
        "worst_actuator": worst_actuator,
        "synced_lines": len(synced_data),
        "first_divergent_step": diverged[0]["step"] if diverged else None,
        "first_divergent_val": diverged[0]["divergent_val"] if diverged else None,
    }
    with open(OUT_DIR / "reproduction_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {OUT_DIR / 'reproduction_summary.json'}")

    # Stderr last 30 lines
    if stderr:
        stderr_lines = stderr.splitlines()
        print(f"\n--- Stderr (last 30 lines) ---")
        for line in stderr_lines[-30:]:
            print(line)


if __name__ == "__main__":
    main()
