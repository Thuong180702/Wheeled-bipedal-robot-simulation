import pandas as pd

df = pd.read_csv('outputs/hierarchical_controller_sim/telemetry_1779287212.csv')
print(f'Steps: {len(df)}')
print(f'Terminated: {df["terminated"].iloc[-1]}')
print(f'Reason: {df["termination_reason"].iloc[-1]}')
print(f'Final roll: {df["roll"].iloc[-1]:.2f} deg')
print(f'Final pitch: {df["pitch"].iloc[-1]:.2f} deg')
print(f'Final height: {df["com_z"].iloc[-1]:.3f} m')
print(f'Roll range: [{df["roll"].min():.1f}, {df["roll"].max():.1f}] deg')
print(f'Pitch range: [{df["pitch"].min():.1f}, {df["pitch"].max():.1f}] deg')
