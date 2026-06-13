# Telemetry CSV Writing Bug Audit

## Classification: `CSV_BUG_INCONCLUSIVE`

**Note**: The exact bug mechanism is inconclusive from static analysis. The fix must include defensive logging to determine the root cause during the next run.

## Observed Behavior

| Metric | Value |
|--------|-------|
| F1b survived steps | 500 |
| Summary JSON `written_telemetry_rows` | 500 |
| CSV data rows | 0 |
| CSV header columns | 508 |
| CSV file size | ~11184 bytes |

## Key Evidence

1. **Summary JSON says 500 rows written**, but CSV has no data rows
2. **File size is consistent**: All broken CSVs are ~11184 bytes (header only)
3. **Working CSVs from June 6**: ~900KB with thousands of rows

## Telemetry Writing Code

```python
# Lines 4714-4720 in simulate_hierarchical_controller.py
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(telemetry.keys())
    n_rows = min(len(values) for values in telemetry.values()) if telemetry else 0
    for i in range(n_rows):
        writer.writerow([telemetry[k][i] for k in telemetry.keys()])
```

## Possible Root Causes

### 1. TELEMETRY_BUFFER_EMPTY
All telemetry lists are empty when CSV is written.
- `n_rows = 0` would write header only
- But summary says 500 rows were appended...

### 2. ROW_SCHEMA_MISMATCH (Most Likely)
Balance-core telemetry columns are initialized empty and populated later:

```python
# Line 2654-2655
for key, values in make_balance_core_telemetry_columns().items():
    telemetry.setdefault(key, values)  # Sets to empty list []
```

If `append_balance_core_telemetry()` doesn't populate these columns (e.g., result.telemetry is empty or missing keys), those columns remain empty.

Then `min(len(values) for values in telemetry.values())` returns 0.

### 3. CSV_FILE_COPIED_BEFORE_FLUSH
File is copied or truncated before flush/close.

### 4. EXCEPTION_DURING_WRITE
An exception in the CSV writing code is caught somewhere.

## Suspicious Pattern

All broken CSV files (June 8) have exactly 508 columns and ~11184 bytes.
Working CSV files (June 6) have the same columns but proper data.

This suggests the bug was introduced between June 6 and June 8.

## Proposed Fix

Add defensive logging to CSV writing code:

```python
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(telemetry.keys())
    n_rows = min(len(values) for values in telemetry.values()) if telemetry else 0
    
    # DEFENSIVE: Log warning if n_rows=0 despite populated columns
    if n_rows == 0:
        populated = [(k, len(v)) for k, v in telemetry.items() if len(v) > 0]
        print(f"[WARNING] CSV n_rows=0 despite {len(populated)} populated columns")
        if populated:
            print(f"[WARNING] First 10 populated: {populated[:10]}")
    
    for i in range(n_rows):
        writer.writerow([telemetry[k][i] for k in telemetry.keys()])
    
    print(f"[INFO] CSV written: {n_rows} rows, {len(telemetry)} columns")
```

Also add to sidecar summary:
```python
"csv_n_rows": n_rows,
"csv_has_data_rows": n_rows > 0,
```

## Next Steps

1. Add defensive logging to CSV writing code
2. Run D2 500-step simulation to reproduce
3. Verify if `n_rows=0` despite populated columns
4. Fix root cause based on evidence