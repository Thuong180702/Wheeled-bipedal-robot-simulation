# Stage 2B Phase A-C: Feedforward Diagnostics Report

**Date:** 2026-05-24 08:21:48

## Phase A: Candidate Torque Analysis

| Candidate | Hip Pitch L | Hip Pitch R | Knee L | Knee R | Max Abs | Asymmetry | Feasible | Margin |
|-----------|-------------|-------------|--------|--------|---------|-----------|----------|--------|
| qfrc_bias | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | YES | 56.0 |
| qfrc_inverse | 0.0 | 0.0 | -0.0 | 0.0 | 0.0 | 0.0 | YES | 57.0 |
| empirical | 4.1 | 3.2 | -15.5 | -15.8 | 15.8 | 1.3 | YES | 41.2 |

## Phase C: Best Configuration

**No successful configuration found.**

All tested configurations failed to meet acceptance criteria:
- Min CoM height > 0.38 m
- Contact stable (double contact for first 10 steps)
- Max roll < 20°

## Summary

