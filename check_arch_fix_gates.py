import pandas as pd
import numpy as np

df = pd.read_csv('outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/signfix_500_T6F_sign_corrected/telemetry_1781269776.csv')

band_state = df['tuned_band_state_id'].values
arch_fix_active = df['arch_fix_active'].values
arch_fix_height_gate = df['arch_fix_height_gate_pass'].values
arch_fix_band_gate = df['arch_fix_band_gate_pass'].values
arch_fix_safety_gate = df['arch_fix_safety_gate_pass'].values
arch_fix_recenter_gate = df['arch_fix_recenter_gate_pass'].values

# Steps in hard or emergency
hard_or_emergency = (band_state == 3) | (band_state == 4)

print(f'Steps in hard/emergency band: {int(np.sum(hard_or_emergency))}')
print(f'Arch fix active: {int(np.sum(arch_fix_active))}')
print(f'Difference: {int(np.sum(hard_or_emergency)) - int(np.sum(arch_fix_active))} steps')

# Check gates during hard/emergency
print(f'\nDuring hard/emergency steps:')
print(f'  Height gate pass: {int(np.sum(arch_fix_height_gate[hard_or_emergency]))} / {int(np.sum(hard_or_emergency))}')
print(f'  Band gate pass: {int(np.sum(arch_fix_band_gate[hard_or_emergency]))} / {int(np.sum(hard_or_emergency))}')
print(f'  Safety gate pass: {int(np.sum(arch_fix_safety_gate[hard_or_emergency]))} / {int(np.sum(hard_or_emergency))}')
print(f'  Recenter gate pass: {int(np.sum(arch_fix_recenter_gate[hard_or_emergency]))} / {int(np.sum(hard_or_emergency))}')

# Which gate failed most?
hard_emergency_steps = np.where(hard_or_emergency & ~arch_fix_active)[0]
print(f'\n{len(hard_emergency_steps)} hard/emergency steps where arch_fix NOT active:')
if len(hard_emergency_steps) > 0:
    for idx in hard_emergency_steps[:5]:
        print(f'  Step {idx}: height={arch_fix_height_gate[idx]}, band={arch_fix_band_gate[idx]}, safety={arch_fix_safety_gate[idx]}, recenter={arch_fix_recenter_gate[idx]}')
