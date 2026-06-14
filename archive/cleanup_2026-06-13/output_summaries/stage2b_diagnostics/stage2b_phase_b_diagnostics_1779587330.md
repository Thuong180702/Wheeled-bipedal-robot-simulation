# Stage 2B Phase B: Feedforward Diagnostics Report

**Date:** 2026-05-24 08:48:50

## Phase A: Candidate Torque Analysis

| Candidate | Hip Pitch L | Hip Pitch R | Knee L | Knee R | Max Abs | Asymmetry | Feasible | Margin |
|-----------|-------------|-------------|--------|--------|---------|-----------|----------|--------|
| qfrc_bias | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | YES | 56.0 |
| qfrc_inverse | 0.0 | 0.0 | -0.0 | 0.0 | 0.0 | 0.0 | YES | 57.0 |
| empirical | 4.1 | 3.2 | -15.5 | -15.8 | 15.8 | 1.3 | YES | 41.2 |
| +empirical | 4.1 | 3.2 | -15.5 | -15.8 | 15.8 | 1.3 | YES | 41.2 |
| -empirical | -4.1 | -3.2 | 15.5 | 15.8 | 15.8 | 1.3 | YES | 41.2 |

## Phase B: One-Step Validation Results

| Candidate | CoM Drop | Roll | Left Wheel | Right Wheel | Non-Wheel | Sim Steps | Passes |
|-----------|----------|------|------------|-------------|-----------|-----------|--------|
| qfrc_bias | 5.3mm | 0.1° | YES | YES | NO | YES | PASS |
| qfrc_inverse | 4.6mm | 0.6° | YES | YES | NO | YES | PASS |
| empirical | 0.0mm | 1.5° | YES | YES | NO | YES | PASS |
| +empirical | 0.0mm | 1.5° | YES | YES | NO | YES | PASS |
| -empirical | 7.2mm | 3.0° | NO | NO | NO | YES | FAIL |

## Contact Classification Analysis

### Question: Are the 4 contacts all wheel-floor contacts?

**qfrc_bias:**
- Wheel-floor contact records (first 10 steps): [4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
- Left wheel maintains floor contact: True
- Right wheel maintains floor contact: True
- Non-wheel floor contacts detected: False

**qfrc_inverse:**
- Wheel-floor contact records (first 10 steps): [4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
- Left wheel maintains floor contact: True
- Right wheel maintains floor contact: True
- Non-wheel floor contacts detected: False

**empirical:**
- Wheel-floor contact records (first 10 steps): [4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
- Left wheel maintains floor contact: True
- Right wheel maintains floor contact: True
- Non-wheel floor contacts detected: False

**+empirical:**
- Wheel-floor contact records (first 10 steps): [4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
- Left wheel maintains floor contact: True
- Right wheel maintains floor contact: True
- Non-wheel floor contacts detected: False

**-empirical:**
- Wheel-floor contact records (first 10 steps): [4, 4, 0, 0, 0, 0, 0, 0, 0, 0]
- Left wheel maintains floor contact: False
- Right wheel maintains floor contact: False
- Non-wheel floor contacts detected: False

### Question: Does the robot actually step and move?

**qfrc_bias:**
- Simulation steps properly (time advances): True
- CoM actually moves: True
- Initial CoM: 0.404m
- Final CoM: 0.399m
- CoM drop: 5.3mm

**qfrc_inverse:**
- Simulation steps properly (time advances): True
- CoM actually moves: True
- Initial CoM: 0.404m
- Final CoM: 0.400m
- CoM drop: 4.6mm

**empirical:**
- Simulation steps properly (time advances): True
- CoM actually moves: True
- Initial CoM: 0.404m
- Final CoM: 0.410m
- CoM drop: 0.0mm

**+empirical:**
- Simulation steps properly (time advances): True
- CoM actually moves: True
- Initial CoM: 0.404m
- Final CoM: 0.410m
- CoM drop: 0.0mm

**-empirical:**
- Simulation steps properly (time advances): True
- CoM actually moves: True
- Initial CoM: 0.404m
- Final CoM: 0.397m
- CoM drop: 7.2mm

### Question: Does empirical feedforward improve behavior?

| Candidate | CoM Drop | Roll | Contact Stable | Passes |
|-----------|----------|------|----------------|--------|
| empirical | 0.0mm | 1.5° | YES | PASS |
| +empirical | 0.0mm | 1.5° | YES | PASS |
| -empirical | 7.2mm | 3.0° | NO | FAIL |

## Summary

[SUCCESS] **4 candidate(s) passed Phase B validation:**

- qfrc_bias
- qfrc_inverse
- empirical
- +empirical

**Recommendation:** Proceed to Phase C configuration sweep with validated candidates.

