"""Phase 0: Run all K2 JAX tests and capture results."""
import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "k2_jax_phase0_semantic_closure"
OUT.mkdir(parents=True, exist_ok=True)

# Run pytest
print("=== Running K2 JAX tests ===")
result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/test_k2_jax_step_parity.py",
     "tests/test_k2_jax_component_parity.py", "tests/test_k2_jax_backend_cli.py",
     "tests/test_k2_jax_branch_activity_audit.py", "-v", "--tb=long"],
    cwd=str(ROOT), capture_output=True, text=True, timeout=600,
)
with open(OUT / "pytest_output.txt", "w") as f:
    f.write(result.stdout)
    f.write("\n\n=== STDERR ===\n")
    f.write(result.stderr)

print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
if result.stderr:
    print("STDERR:", result.stderr[-1000:])

# Parse results
lines = result.stdout.splitlines()
passed = sum(1 for l in lines if "PASSED" in l and "::" in l)
failed = sum(1 for l in lines if "FAILED" in l and "::" in l)
total = passed + failed
summary_line = [l for l in lines if "passed" in l and "failed" in l and "=" in l]
print(f"\nTest summary: {passed} passed, {failed} failed, {total} total")
if summary_line:
    print(summary_line[-1])

# Exit code based on test result
sys.exit(0 if result.returncode == 0 else 1)
