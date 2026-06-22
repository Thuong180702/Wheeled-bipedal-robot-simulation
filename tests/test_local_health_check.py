import subprocess
import os

def test_local_health_check():
    """Run the local health check script and ensure it exits successfully."""
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "run_local_health_check.sh"))
    assert os.path.isfile(script_path), f"Health check script not found at {script_path}"
    # Use sh interpreter which is available in this environment
    result = subprocess.run(["sh", script_path], capture_output=True, text=True)
    assert result.returncode == 0, (
        f"Health check script failed with exit code {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )