"""Compare old IK module targets with corrected empirical IK manifold.

This validates whether the old IK targets are achievable according to the
contact-aware forward kinematics sweep.
"""

import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()


def main():
    # Load old IK targets from telemetry
    telemetry_dir = Path("outputs/phase_b9_task4_telemetry")
    telemetry_heights = [0.70, 0.65, 0.60, 0.55, 0.50, 0.45]
    old_ik_targets = {}

    for h in telemetry_heights:
        telem_file = telemetry_dir / f"telemetry_h{h:.2f}.json"
        if not telem_file.exists():
            continue

        with open(telem_file, 'r') as f:
            telem = json.load(f)

        if telem['episodes']:
            first_snap = telem['episodes'][0]['sample_snapshots']['first']
            old_ik_targets[h] = {
                'hip_pitch': first_snap['hip_pitch_ik_target'],
                'knee': first_snap['knee_ik_target']
            }

    # Load corrected empirical IK targets
    empirical_file = Path("outputs/phase_b9_task6_empirical_ik_corrected/empirical_ik_targets.json")
    with open(empirical_file, 'r') as f:
        empirical_data = json.load(f)

    empirical_targets = {t['target_height']: t for t in empirical_data['targets']}
    min_height = empirical_data['min_height']
    max_height = empirical_data['max_height']

    # Compare
    console.print()
    console.print('='*90)
    console.print('OLD IK MODULE vs CORRECTED EMPIRICAL IK MANIFOLD')
    console.print('='*90)

    table = Table(title="IK Target Comparison")
    table.add_column("Height [m]", justify="right")
    table.add_column("Old IK Hip", justify="right")
    table.add_column("Empirical Hip", justify="right")
    table.add_column("D Hip [rad]", justify="right")
    table.add_column("Old IK Knee", justify="right")
    table.add_column("Empirical Knee", justify="right")
    table.add_column("D Knee [rad]", justify="right")
    table.add_column("Status", justify="center")

    for h in sorted(old_ik_targets.keys()):
        old = old_ik_targets[h]
        emp = empirical_targets.get(h, {})

        if not emp:
            continue

        delta_hip = emp['hip_pitch'] - old['hip_pitch']
        delta_knee = emp['knee'] - old['knee']

        # Determine status
        if not emp['achievable']:
            status = "[red]OUT OF RANGE[/red]"
        elif abs(delta_hip) < 0.1 and abs(delta_knee) < 0.1:
            status = "[green]MATCH[/green]"
        elif abs(delta_hip) < 0.3 and abs(delta_knee) < 0.3:
            status = "[yellow]CLOSE[/yellow]"
        else:
            status = "[red]MISMATCH[/red]"

        table.add_row(
            f"{h:.2f}",
            f"{old['hip_pitch']:.3f}",
            f"{emp['hip_pitch']:.3f}",
            f"{delta_hip:+.3f}",
            f"{old['knee']:.3f}",
            f"{emp['knee']:.3f}",
            f"{delta_knee:+.3f}",
            status,
        )

    console.print(table)

    # Analysis
    console.print()
    console.print('[bold cyan]ANALYSIS:[/bold cyan]')
    console.print()
    console.print(f'Empirical manifold range: [{min_height:.3f}, {max_height:.3f}] m')
    console.print()

    # Check each height
    match_count = 0
    close_count = 0
    mismatch_count = 0
    out_of_range_count = 0

    for h in sorted(old_ik_targets.keys()):
        old = old_ik_targets[h]
        emp = empirical_targets.get(h, {})

        if not emp['achievable']:
            out_of_range_count += 1
            console.print(f'[red]h={h:.2f}m: OUT OF RANGE[/red]')
            console.print(f'  Target height {h:.2f}m exceeds max achievable {max_height:.3f}m')
        else:
            delta_hip = abs(emp['hip_pitch'] - old['hip_pitch'])
            delta_knee = abs(emp['knee'] - old['knee'])

            if delta_hip < 0.1 and delta_knee < 0.1:
                match_count += 1
                console.print(f'[green]h={h:.2f}m: MATCH[/green]')
                console.print(f'  Old IK targets are achievable (D_hip={delta_hip:.3f}, D_knee={delta_knee:.3f})')
            elif delta_hip < 0.3 and delta_knee < 0.3:
                close_count += 1
                console.print(f'[yellow]h={h:.2f}m: CLOSE[/yellow]')
                console.print(f'  Old IK targets are approximately achievable (D_hip={delta_hip:.3f}, D_knee={delta_knee:.3f})')
            else:
                mismatch_count += 1
                console.print(f'[red]h={h:.2f}m: MISMATCH[/red]')
                console.print(f'  Old IK: hip={old["hip_pitch"]:.3f}, knee={old["knee"]:.3f}')
                console.print(f'  Empirical: hip={emp["hip_pitch"]:.3f}, knee={emp["knee"]:.3f}')
                console.print(f'  Large deviation (D_hip={delta_hip:.3f}, D_knee={delta_knee:.3f})')

    console.print()
    console.print('='*90)
    console.print('[bold]SUMMARY:[/bold]')
    console.print(f'  MATCH: {match_count}/{len(old_ik_targets)} heights')
    console.print(f'  CLOSE: {close_count}/{len(old_ik_targets)} heights')
    console.print(f'  MISMATCH: {mismatch_count}/{len(old_ik_targets)} heights')
    console.print(f'  OUT OF RANGE: {out_of_range_count}/{len(old_ik_targets)} heights')
    console.print()

    if match_count + close_count == len(old_ik_targets):
        console.print('[bold green]CONCLUSION: Old IK targets are VALIDATED by empirical manifold[/bold green]')
        console.print('The old IK module generates achievable targets within the kinematic range.')
    elif out_of_range_count > 0:
        console.print('[bold yellow]CONCLUSION: Old IK targets are PARTIALLY VALID[/bold yellow]')
        console.print(f'{out_of_range_count} height(s) exceed the achievable range [{min_height:.3f}, {max_height:.3f}]m')
        console.print('The old IK module needs height clamping to stay within kinematic limits.')
    else:
        console.print('[bold red]CONCLUSION: Old IK targets are INVALID[/bold red]')
        console.print('The old IK module generates targets that deviate significantly from the achievable manifold.')

    console.print()


if __name__ == "__main__":
    main()
