import pandas as pd
import numpy as np

df = pd.read_csv('outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/signfix_500_T6F_sign_corrected/telemetry_1781269776.csv')

band_state = df['tuned_band_state_id'].values
arch_fix_active = df['arch_fix_active'].values

print('Band state distribution:')
for state in [0, 1, 2, 3, 4]:
    count = int(np.sum(band_state == state))
    pct = 100.0 * count / len(df)
    state_names = {0: "normal", 1: "soft", 2: "desired", 3: "hard", 4: "emergency"}
    print(f'  {state} ({state_names.get(state, "unknown")}): {count} ({pct:.1f}%)')

print(f'\nMax error: {df["tuned_abs_error"].max():.4f}m')
print(f'Arch fix active: {int(np.sum(arch_fix_active))} steps ({100.0*np.mean(arch_fix_active):.1f}%)')
print(f'\nSteps in hard or emergency band: {int(np.sum((band_state == 3) | (band_state == 4)))}')
