"""
Check available APCR telemetry columns in the 1000-step run.
"""
import pandas as pd

CSV_PATH = "outputs/hierarchical_controller_sim/telemetry_1781058071.csv"
df = pd.read_csv(CSV_PATH)

# Find all columns related to APCR
apcr_cols = [c for c in df.columns if 'apcr' in c.lower()]
print(f"APCR columns ({len(apcr_cols)}):")
for col in sorted(apcr_cols):
    print(f"  {col}")

# Find all columns related to support drift
support_cols = [c for c in df.columns if 'support' in c.lower()]
print(f"\nSupport columns ({len(support_cols)}):")
for col in sorted(support_cols)[:30]:
    print(f"  {col}")

# Find all columns related to error/drift
error_cols = [c for c in df.columns if 'error' in c.lower() or 'drift' in c.lower()]
print(f"\nError/drift columns ({len(error_cols)}):")
for col in sorted(error_cols)[:30]:
    print(f"  {col}")

# Find all columns related to tau/torque
tau_cols = [c for c in df.columns if 'tau' in c.lower() or 'torque' in c.lower()]
print(f"\nTorque/tau columns ({len(tau_cols)}):")
for col in sorted(tau_cols)[:40]:
    print(f"  {col}")

# Check for hysteresis state columns
hysteresis_cols = [c for c in df.columns if 'hysteresis' in c.lower() or 'state' in c.lower()]
print(f"\nHysteresis/state columns ({len(hysteresis_cols)}):")
for col in sorted(hysteresis_cols)[:30]:
    print(f"  {col}")

# Check all columns for anything that might be APCR1i state
all_cols = sorted(df.columns.tolist())
print(f"\nTotal columns: {len(all_cols)}")

# Look for columns with specific keywords
keywords = ['recenter', 'cross', 'hyst', 'center_from', 'positive', 'negative', 'hold', 'neutral', 'emergency']
for kw in keywords:
    matches = [c for c in all_cols if kw in c.lower()]
    if matches:
        print(f"\nKeyword '{kw}' matches:")
        for m in matches[:10]:
            print(f"  {m}")