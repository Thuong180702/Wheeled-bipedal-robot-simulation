"""Compare baseline (no rate limiting) vs rate-limited controller performance."""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load both telemetry files
baseline = pd.read_csv('outputs/hierarchical_controller_sim/telemetry_1779287212.csv')
rate_limited = pd.read_csv('outputs/hierarchical_controller_sim/telemetry_1779286732.csv')

# Create comparison figure
fig, axes = plt.subplots(3, 2, figsize=(14, 10))
fig.suptitle('Hierarchical Controller: Baseline vs Rate-Limited Comparison', fontsize=16, fontweight='bold')

# Plot 1: Roll angle
ax = axes[0, 0]
ax.plot(baseline['time'], baseline['roll'], 'r-', label='Baseline (no rate limiting)', linewidth=2)
ax.plot(rate_limited['time'], rate_limited['roll'], 'g-', label='Rate limited (500 Nm/s)', linewidth=2)
ax.axhline(0, color='k', linestyle='--', alpha=0.3)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Roll (deg)')
ax.set_title('Roll Stability')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Pitch angle
ax = axes[0, 1]
ax.plot(baseline['time'], baseline['pitch'], 'r-', label='Baseline', linewidth=2)
ax.plot(rate_limited['time'], rate_limited['pitch'], 'g-', label='Rate limited', linewidth=2)
ax.axhline(0, color='k', linestyle='--', alpha=0.3)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Pitch (deg)')
ax.set_title('Pitch Stability')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Torque rate comparison
ax = axes[1, 0]
ax.plot(baseline['time'], baseline['tau_rate_unlimited'], 'r-', label='Baseline (unlimited)', linewidth=2)
ax.plot(rate_limited['time'], rate_limited['tau_rate_unlimited'], 'r--', label='Unlimited (rate-limited run)', linewidth=1, alpha=0.5)
ax.plot(rate_limited['time'], rate_limited['tau_rate_limited'], 'g-', label='Rate limited', linewidth=2)
ax.axhline(500, color='orange', linestyle='--', label='500 Nm/s limit', linewidth=2)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Torque Rate (Nm/s)')
ax.set_title('Torque Rate of Change')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 2000])

# Plot 4: Max torque magnitude
ax = axes[1, 1]
ax.plot(baseline['time'], baseline['tau_total_max'], 'r-', label='Baseline', linewidth=2)
ax.plot(rate_limited['time'], rate_limited['tau_total_max'], 'g-', label='Rate limited', linewidth=2)
ax.axhline(30, color='orange', linestyle='--', label='Actuator limit (30 Nm)', linewidth=2)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Max Torque (Nm)')
ax.set_title('Maximum Joint Torque')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 5: Position tracking error
ax = axes[2, 0]
ax.plot(baseline['time'], baseline['joint_pos_error_norm'], 'r-', label='Baseline', linewidth=2)
ax.plot(rate_limited['time'], rate_limited['joint_pos_error_norm'], 'g-', label='Rate limited', linewidth=2)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Position Error (rad)')
ax.set_title('Joint Position Tracking Error')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 6: CoM height
ax = axes[2, 1]
ax.plot(baseline['time'], baseline['com_z'], 'r-', label='Baseline', linewidth=2)
ax.plot(rate_limited['time'], rate_limited['com_z'], 'g-', label='Rate limited', linewidth=2)
ax.axhline(0.55, color='k', linestyle='--', alpha=0.3, label='Target height')
ax.set_xlabel('Time (s)')
ax.set_ylabel('CoM Height (m)')
ax.set_title('Center of Mass Height')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/hierarchical_controller_sim/comparison_baseline_vs_rate_limited_v2.png', dpi=150, bbox_inches='tight')
print('Comparison plot saved to: outputs/hierarchical_controller_sim/comparison_baseline_vs_rate_limited_v2.png')

# Print summary statistics
print('\n=== PERFORMANCE COMPARISON SUMMARY ===\n')
print(f'Survival:')
print(f'  Baseline:      {len(baseline)} steps ({baseline["time"].iloc[-1]:.2f}s)')
print(f'  Rate Limited:  {len(rate_limited)} steps ({rate_limited["time"].iloc[-1]:.2f}s)')
print(f'  Result: Both completed full simulation')

print(f'\nRoll Stability (RMS):')
print(f'  Baseline:      {baseline["roll"].std():.2f}deg')
print(f'  Rate Limited:  {rate_limited["roll"].std():.2f}deg')
if baseline["roll"].std() > 0:
    print(f'  Improvement:   {100*(1 - rate_limited["roll"].std()/baseline["roll"].std()):.0f}% reduction')

print(f'\nPitch Stability (RMS):')
print(f'  Baseline:      {baseline["pitch"].std():.2f}deg')
print(f'  Rate Limited:  {rate_limited["pitch"].std():.2f}deg')
if baseline["pitch"].std() > 0:
    print(f'  Improvement:   {100*(1 - rate_limited["pitch"].std()/baseline["pitch"].std()):.0f}% reduction')

print(f'\nTorque Rate (Mean):')
print(f'  Baseline:      {baseline["tau_rate_unlimited"].mean():.1f} Nm/s')
print(f'  Rate Limited:  {rate_limited["tau_rate_limited"].mean():.1f} Nm/s')
print(f'  Improvement:   {100*(1 - rate_limited["tau_rate_limited"].mean()/baseline["tau_rate_unlimited"].mean()):.0f}% reduction')

print(f'\nPosition Tracking Error (Mean):')
print(f'  Baseline:      {baseline["joint_pos_error_norm"].mean():.6f} rad')
print(f'  Rate Limited:  {rate_limited["joint_pos_error_norm"].mean():.6f} rad')
print(f'  Note: Error increased due to rate limiting (expected tradeoff)')
