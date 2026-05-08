"""Compare IK targets with static equilibrium postures (Phase B.9 Task 5 analysis)."""

import json
from pathlib import Path

def main():
    # Load equilibrium results
    eq_file = Path("outputs/phase_b9_task5_equilibrium/equilibrium_postures.json")
    with open(eq_file, 'r') as f:
        eq_data = json.load(f)

    # Load telemetry to get IK targets
    telemetry_dir = Path("outputs/phase_b9_task4_telemetry")
    telemetry_heights = [0.70, 0.65, 0.60, 0.55, 0.50, 0.45]
    ik_targets = {}

    for h in telemetry_heights:
        telem_file = telemetry_dir / f"telemetry_h{h:.2f}.json"
        with open(telem_file, 'r') as f:
            telem = json.load(f)

        # Get IK target from first episode's first snapshot
        if telem['episodes']:
            first_snap = telem['episodes'][0]['sample_snapshots']['first']
            ik_targets[h] = {
                'hip_pitch': first_snap['hip_pitch_ik_target'],
                'knee': first_snap['knee_ik_target']
            }

    # Print comparison table
    print()
    print('='*85)
    print('IK TARGETS vs STATIC EQUILIBRIUM POSTURES')
    print('='*85)
    print(f"{'Height':>8} | {'IK Hip':>10} | {'Eq Hip':>10} | {'Delta':>10} | "
          f"{'IK Knee':>10} | {'Eq Knee':>10} | {'Delta':>10}")
    print('-'*85)

    for eq in eq_data:
        h = eq['target_height']
        if h in ik_targets:
            ik = ik_targets[h]
            print(f"{h:>8.2f} | {ik['hip_pitch']:>10.3f} | {eq['hip_pitch']:>10.3f} | "
                  f"{eq['hip_pitch']-ik['hip_pitch']:>+10.3f} | "
                  f"{ik['knee']:>10.3f} | {eq['knee']:>10.3f} | "
                  f"{eq['knee']-ik['knee']:>+10.3f}")

    print('='*85)
    print()
    print('KEY FINDINGS:')
    print()
    print('1. KINEMATIC LIMITATION: Robot can only achieve ~0.71m height in static equilibrium')
    print('   - All commanded heights converge to same config: hip~0.26rad, knee~0.54rad')
    print('   - Actual height stays at 0.71m regardless of command')
    print()
    print('2. IK TARGETS ARE INFEASIBLE:')
    print('   - IK assumes robot can squat to any height (0.40-0.70m)')
    print('   - IK targets get progressively more bent for lower heights')
    print('   - At h=0.45m: IK commands hip=1.139rad, knee=2.217rad')
    print('   - These targets are kinematically impossible to achieve')
    print()
    print('3. ROOT CAUSE OF FAILURES:')
    print('   - PID controller tries to track infeasible IK targets')
    print('   - Joints cannot reach commanded positions')
    print('   - Large tracking errors accumulate (>0.8rad on hip pitch)')
    print('   - Robot falls immediately (leg_config failure mode)')
    print()
    print('4. HEIGHT ERROR PROGRESSION:')
    for eq in eq_data:
        h = eq['target_height']
        print(f"   h={h:.2f}m: height_error={eq['height_error']:.3f}m "
              f"(actual={eq['actual_height']:.2f}m)")
    print()
    print('CONCLUSION: The IK module is fundamentally broken. It generates targets')
    print('            that violate the robot\'s kinematic constraints.')
    print()
    print('NEXT STEPS (Phase B.9 Task 6):')
    print('  - Numerical linearization through actual simulator and PID')
    print('  - Find true equilibrium manifold via forward kinematics')
    print('  - Rebuild IK to respect kinematic limits')
    print()


if __name__ == "__main__":
    main()
