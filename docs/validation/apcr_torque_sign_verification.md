# APCR Torque Sign Verification

## Classification: APCR_TORQUE_SIGN_CONFIRMED

## Executive Summary

From the existing sign verification analysis (`outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/sign_verification.json`), the torque sign convention is confirmed.

## Phase 2 Verification Results

### Sign Conventions Verified

| Convention | Verified |
|------------|----------|
| positive_pitch_means_forward_lean | ✓ |
| positive_tau_pitch_opposes_forward_lean | ✓ |
| positive_wheel_torque_creates_negative_pitch_rate | ✓ |
| positive_signed_error_means_forward_drift | ✓ |

### Torque Sign for APCR States

**CROSS_FROM_POSITIVE (positive pitch + positive drift):**
- Apply NEGATIVE wheel torque
- This creates negative pitch_rate (robot leans backward)
- Goal: make support drift move back toward zero

**CROSS_FROM_NEGATIVE (negative pitch + negative drift):**
- Apply POSITIVE wheel torque
- This creates positive pitch_rate (robot leans forward)
- Goal: make support drift move back toward zero

### Rationale

When the robot has:
- Positive pitch (leaning forward)
- Positive support drift (CoM ahead of wheel contact)

The APCR should command the wheels to create negative pitch_rate. This causes the robot to lean backward, which moves the support position back toward zero.

### Files Generated

- `docs/validation/apcr_torque_sign_verification.md` - This file
- `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/apcr_torque_sign_verification.json` - Already exists from prior analysis
