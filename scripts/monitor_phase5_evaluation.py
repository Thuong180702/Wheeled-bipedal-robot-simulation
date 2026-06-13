"""Monitor Phase 5 evaluation progress."""

import time
from pathlib import Path
from datetime import datetime

sim_dir = Path("outputs/hierarchical_controller_sim")
last_count = 0
start_time = time.time()

print("Monitoring Phase 5 HY-FF Evaluation Progress")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%H:%M:%S')}")
print(f"Expected: 18 simulations (6 candidates × 3 variants)")
print(f"Target: ~3-5 minutes total\n")

while True:
    telemetry_files = list(sim_dir.glob("telemetry_*.csv"))
    current_count = len(telemetry_files)

    if current_count > last_count:
        elapsed = time.time() - start_time
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Simulations completed: {current_count - last_count} new ({current_count} total) - Elapsed: {elapsed:.0f}s")
        last_count = current_count

        # Estimate completion
        if current_count > 0:
            avg_time_per_sim = elapsed / current_count
            remaining = 18 - current_count
            est_remaining = avg_time_per_sim * remaining
            print(f"  → Estimated completion: {est_remaining:.0f}s remaining")

    time.sleep(5)

    # Check for evaluation results
    results_file = Path("outputs/hip_yaw_hy_ff_evaluation/evaluation_results.json")
    if results_file.exists():
        print(f"\n{'=' * 80}")
        print("EVALUATION COMPLETE!")
        print(f"Results file: {results_file}")
        print(f"Total time: {time.time() - start_time:.0f}s")
        break
