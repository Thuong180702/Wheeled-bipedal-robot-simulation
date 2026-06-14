"""Check APCR1n safety gate conditions"""
import pandas as pd

df = pd.read_csv("outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/phase2_ablation_2000_APCR1n/telemetry_apcr1n.csv")

# Find pitch column
pitch_col = [c for c in df.columns if 'pitch' in c.lower() and 'x' in c.lower() and 'deg' in c.lower()]
roll_col = [c for c in df.columns if 'roll' in c.lower() and 'y' in c.lower() and 'deg' in c.lower()]

print("Pitch columns:", pitch_col)
print("Roll columns:", roll_col)

if pitch_col and roll_col:
    print("\nrobot_pitch_x_deg:", df[pitch_col[0]].min(), "-", df[pitch_col[0]].max())
    print("robot_roll_y_deg:", df[roll_col[0]].min(), "-", df[roll_col[0]].max())

    post = df[df['step'] >= 100]
    pitch_ok = (abs(post[pitch_col[0]]) <= 20).sum()
    roll_ok = (abs(post[roll_col[0]]) <= 10).sum()

    print("\nSafety gate checks (steps 100+):")
    print("  |pitch| <= 20:", pitch_ok, "/", len(post))
    print("  |roll| <= 10:", roll_ok, "/", len(post))