# K2 V3 Random Single-Push Test

**Date:** 2026-07-01
**Controller:** `K2_JAX_DEDICATED_DEFAULT_V3` (official default, promoted 2026-07-01)
**Test type:** Test-only diagnostic — random single-push survivability
**Status:** **FAIL** — 0/5 trials survived

---

## 1. Test Purpose

Test whether the K2 V3 controller can survive a **single random push** with randomized:
- Push timing (step 300–500)
- Force magnitude (50–100 N)
- Direction (random 3D unit vector, z clamped to [-0.25, 0.25])
- Application body (randomly selected from safe subset)

This isolates random single-push robustness at nominal mid-height (0.40 m) without confounding height randomization.

## 2. Controller Profile

```
Profile:  K2_JAX_DEDICATED_DEFAULT_V3
Variant:  mid_0p400 (0.40 m target CoM)
Drift:    ON (k_vel=10.0, k_pos=0.0, k_heading=0.0, max_tau=8.0)
Mode-div: ON (soft_gain=0.80, ref_source=target)
q_ref:    original-k2-exact (static)
```

**No controller code or parameters were modified for this test.**

## 3. Randomization Method

| Parameter | Range / Method |
|-----------|---------------|
| Seeds | [101, 102, 103, 104, 105] (deterministic, reproducible) |
| Push step | UniformInteger(300, 500) |
| Force magnitude | Uniform(50.0, 100.0) N |
| Direction (x, y) | Uniform(-1, 1), normalized |
| Direction (z) | Uniform(-0.25, 0.25), normalized |
| Target body | Uniform random from [torso, l_thigh, r_thigh, l_knee_link, r_knee_link] |
| Application point | Body COM only (MuJoCo `xfrc_applied` limitation) |
| Push duration | Fixed 8 steps (80 ms impulse) |
| Push count | Exactly 1 per trial |
| Simulation steps | 7000 per trial |
| Height setup | `mid_0p400_setup.json` (0.40 m CoM) |

**Limitation:** MuJoCo `xfrc_applied` applies forces at body center of mass only. Local application point offsets are not supported. All pushes apply at COM.

## 4. Sampled Push Configurations

| Trial | Seed | Push Step | Force (N) | Direction [x, y, z] | Target Body |
|-------|------|-----------|-----------|----------------------|-------------|
| 101 | 101 | 448 | 92.36 | [+0.992, +0.095, −0.086] | torso |
| 102 | 102 | 337 | 83.81 | [−0.902, +0.346, +0.256] | l_thigh |
| 103 | 103 | 482 | 72.81 | [+0.665, +0.714, −0.217] | torso |
| 104 | 104 | 305 | 93.59 | [+0.644, −0.756, −0.118] | l_thigh |
| 105 | 105 | 451 | 67.46 | [+0.539, +0.838, −0.087] | r_knee_link |

**Direction interpretation (robot faces −Y in world frame):**
- +x = leftwards lateral push
- −x = rightwards lateral push
- +y = backwards push (behind robot)
- −y = forwards push (in front of robot)

## 5. Results — All 5 Trials

### Summary Table

| Metric | Trial 101 | Trial 102 | Trial 103 | Trial 104 | Trial 105 |
|--------|-----------|-----------|-----------|-----------|-----------|
| **Seed** | 101 | 102 | 103 | 104 | 105 |
| **Push step** | 448 | 337 | 482 | 305 | 451 |
| **Push duration (steps)** | 8 | 8 | 8 | 8 | 8 |
| **Force (N)** | 92.36 | 83.81 | 72.81 | 93.59 | 67.46 |
| **Direction [x,y,z]** | [+0.99,+0.10,−0.09] | [−0.90,+0.35,+0.26] | [+0.67,+0.71,−0.22] | [+0.64,−0.76,−0.12] | [+0.54,+0.84,−0.09] |
| **Target body** | torso | l_thigh | torso | l_thigh | r_knee_link |
| **FALL** | YES | YES | YES | YES | YES |
| **Fall step** | 472 | 371 | 509 | 357 | 516 |
| **Termination reason** | orientation_fail (roll=45.7°) | orientation_fail (pitch=45.6°) | height_too_low (0.344<0.349) | height_too_low (0.346<0.349) | height_too_low (0.343<0.349) |
| **Recovery success** | NO | NO | NO | NO | NO |
| **Steps survived post-push** | 24 | 34 | 27 | 52 | 65 |

### Posture Metrics

| Metric | Trial 101 | Trial 102 | Trial 103 | Trial 104 | Trial 105 |
|--------|-----------|-----------|-----------|-----------|-----------|
| **Pitch peak (°)** | 3.4 | 45.6 | 21.1 | 32.6 | 42.6 |
| **Pitch RMS (°)** | 1.6 | 5.1 | 2.4 | 6.8 | 7.8 |
| **Roll peak (°)** | 45.7 | 27.7 | 47.1 | 31.9 | 5.3 |
| **Roll RMS (°)** | 4.6 | 4.4 | 4.4 | 4.8 | 0.9 |
| **Yaw max (°)** | 3.8 | 122.1 | 26.5 | 39.2 | 144.9 |
| **Min CoM Z (m)** | 0.393 | 0.399 | 0.344 | 0.346 | 0.343 |
| **CoM Z RMS error (m)** | 0.012 | 0.016 | 0.013 | 0.014 | 0.014 |

### Drift and Displacement

| Metric | Trial 101 | Trial 102 | Trial 103 | Trial 104 | Trial 105 |
|--------|-----------|-----------|-----------|-----------|-----------|
| **Sagittal final drift (m)** | +0.011 | +0.082 | +0.281 | −1.108 | +0.528 |
| **Lateral final drift (m)** | +0.301 | −0.352 | +0.165 | +0.572 | +0.199 |
| **Final displacement (m)** | 0.30 | 0.36 | 0.33 | 1.25 | 0.56 |
| **Max displacement (m)** | 0.30 | 0.37 | 0.33 | 1.11 | 0.56 |

### Torque and Divergence

| Metric | Trial 101 | Trial 102 | Trial 103 | Trial 104 | Trial 105 |
|--------|-----------|-----------|-----------|-----------|-----------|
| **Max torque (Nm)** | 30.00 | 30.00 | 30.00 | 30.00 | 30.00 |
| **Wheel torque peak (Nm)** | 10.23 | 10.23 | 10.23 | 10.23 | 10.23 |
| **Hip-yaw torque peak (Nm)** | 5.96 | 18.86 | 7.03 | 10.54 | 14.67 |
| **Hip-yaw div max (rad)** | 0.045 | 0.489 | 0.282 | 0.386 | 0.812 |
| **Hip-yaw div RMS (rad)** | 0.027 | 0.067 | 0.046 | 0.100 | 0.141 |
| **Contact loss (steps)** | 31 | 39 | 28 | 48 | 65 |

### Post-Push Window Metrics (500-step window)

| Metric | Trial 101 | Trial 102 | Trial 103 | Trial 104 | Trial 105 |
|--------|-----------|-----------|-----------|-----------|-----------|
| **Post-push pitch RMS (°)** | 1.20 | 17.83 | 8.95 | 18.45 | 22.93 |
| **Post-push support RMS (m)** | 0.026 | 0.055 | 0.165 | 0.479 | 0.328 |

### Performance

| Metric | Trial 101 | Trial 102 | Trial 103 | Trial 104 | Trial 105 |
|--------|-----------|-----------|-----------|-----------|-----------|
| **Mean Hz** | 61.3 | 58.9 | 61.9 | 60.5 | 62.6 |

## 6. Diagnosis

### Overall: ALL 5 TRIALS FAILED — 0/5 survival rate

### Worst Trial: **104**
- Earliest fall (step 357, only 52 steps after push at step 305)
- Largest displacement (1.25 m final, 1.11 m sagittal)
- Second-highest force (93.6 N) on l_thigh
- Severe pitch excursion (−32.6°) combined with high roll (31.9°)
- Second-worst hip-yaw divergence (0.386 rad)

### Runner-Up: **105**
- Worst hip-yaw divergence (0.812 rad, exceeding the 0.35 rad SAFETY_FAIL threshold frequently used in prior studies)
- Worst yaw excursion (144.9° — nearly half a rotation)
- Despite lowest force (67.5 N), the r_knee_link target body appears most destabilizing
- Most contact loss steps (65)

### Best (least bad): **101**
- Survived longest post-push (24 steps after push — still failed rapidly)
- Best hip-yaw divergence (0.045 rad max, 0.027 rad RMS)
- Moderate roll excursion (45.7°) — still a terminal orientation failure
- Torso push at 92.4 N — the controller partially resisted roll

### Failure Analysis by Direction

| Push direction | Trials | Outcome |
|---------------|--------|---------|
| Strongly lateral (+x dominant) | 101 | Roll failure |
| Strongly lateral + upward (−x, +z) | 102 | Pitch+yaw failure (l_thigh) |
| Diagonal forward-right (+x, +y) | 103 | Height collapse |
| Diagonal forward-left (+x, −y) | 104 | Height collapse + massive drift |
| Diagonal forward-right (+x, +y) | 105 | Yaw spinout + height collapse |

### Failure Analysis by Target Body

| Target Body | Trials | Outcome |
|-------------|--------|---------|
| torso | 101, 103 | 2/2 failed (roll, height) |
| l_thigh | 102, 104 | 2/2 failed (pitch+yaw, height+drift) |
| r_knee_link | 105 | 1/1 failed (yaw spinout) |

### Key Observations

1. **No trial survived.** The K2 V3 controller at mid-height (0.40 m) cannot recover from a single random push in the 50–100 N range.

2. **Failure modes varied by direction and body.** Torso pushes caused roll-dominant failures. Limb pushes (l_thigh, r_knee_link) caused more complex failures involving yaw spinout and height collapse.

3. **Hip-yaw divergence was pathologically high in trials 102, 104, 105** (0.28–0.81 rad). Only trial 101 maintained reasonable hip-yaw divergence (0.045 rad).

4. **Torque saturation was universal.** All trials hit the 30 Nm torque limit at multiple joints, indicating the controller could not generate sufficient corrective torque within its limits.

5. **Contact loss was significant** (28–65 steps), suggesting the wheels lost ground contact during the push response.

6. **Drift controller did not prevent failure.** Despite being enabled (k_vel=10.0), the drift controller was insufficient to arrest post-push drift. Trial 104 had 1.25 m displacement.

7. **Force magnitude did not strongly predict failure type.** Trial 105 failed severely (yaw 144.9°, hip-yaw div 0.812) despite the lowest force (67.5 N), while trial 101 had the "best" metrics despite 92.4 N.

8. **The knee_link target (trial 105) was the most destabilizing per Newton**, suggesting leg-segment pushes induce coupled pitch-yaw-roll modes the controller cannot decouple.

### Closest to Failure Threshold

- **Trial 101**: Roll peak 45.7° vs 45.0° termination threshold → failed by 0.7°
- **Trial 103**: CoM Z min 0.344 m vs 0.349 m floor → failed by 5 mm
- **Trial 104**: CoM Z min 0.346 m vs 0.349 m floor → failed by 3 mm

The margin is extremely thin. Even the "best" trial failed within 24 steps of the push.

### Root Cause Hypothesis

The K2 V3 controller's balancing authority is primarily sagittal (pitch stabilization via wheel torque). Random-direction pushes excite lateral (roll), yaw, and coupled modes that the controller has limited authority over:
- Roll control relies on hip-roll joints with limited torque (60 Nm motor limit but 30 Nm ctrlrange)
- Yaw control has no dedicated high-authority channel (hip-yaw max torque ~19 Nm observed)
- Limb-targeted pushes create kinematic chain disturbances that the posture PD cannot reject quickly enough

## 7. Exact Non-Visual Commands Used

```bash
# Trial 101
python scripts/run_k2_jax_realtime.py \
  --height-setup outputs/physical_target_height_setups/mid_0p400_setup.json \
  --steps 7000 --profile K2_JAX_DEDICATED_DEFAULT_V3 \
  --random-single-push --push-seed 101 \
  --push-step-min 300 --push-step-max 500 \
  --push-force-min 50 --push-force-max 100 \
  --push-duration-steps 8 \
  --save-random-push-config outputs/diag/k2_v3_random_single_push/trial_101/push_config.json \
  --telemetry full --output-dir outputs/diag/k2_v3_random_single_push/trial_101 \
  --quiet

# Trial 102
python scripts/run_k2_jax_realtime.py \
  --height-setup outputs/physical_target_height_setups/mid_0p400_setup.json \
  --steps 7000 --profile K2_JAX_DEDICATED_DEFAULT_V3 \
  --random-single-push --push-seed 102 \
  --push-step-min 300 --push-step-max 500 \
  --push-force-min 50 --push-force-max 100 \
  --push-duration-steps 8 \
  --save-random-push-config outputs/diag/k2_v3_random_single_push/trial_102/push_config.json \
  --telemetry full --output-dir outputs/diag/k2_v3_random_single_push/trial_102 \
  --quiet

# Trial 103
python scripts/run_k2_jax_realtime.py \
  --height-setup outputs/physical_target_height_setups/mid_0p400_setup.json \
  --steps 7000 --profile K2_JAX_DEDICATED_DEFAULT_V3 \
  --random-single-push --push-seed 103 \
  --push-step-min 300 --push-step-max 500 \
  --push-force-min 50 --push-force-max 100 \
  --push-duration-steps 8 \
  --save-random-push-config outputs/diag/k2_v3_random_single_push/trial_103/push_config.json \
  --telemetry full --output-dir outputs/diag/k2_v3_random_single_push/trial_103 \
  --quiet

# Trial 104
python scripts/run_k2_jax_realtime.py \
  --height-setup outputs/physical_target_height_setups/mid_0p400_setup.json \
  --steps 7000 --profile K2_JAX_DEDICATED_DEFAULT_V3 \
  --random-single-push --push-seed 104 \
  --push-step-min 300 --push-step-max 500 \
  --push-force-min 50 --push-force-max 100 \
  --push-duration-steps 8 \
  --save-random-push-config outputs/diag/k2_v3_random_single_push/trial_104/push_config.json \
  --telemetry full --output-dir outputs/diag/k2_v3_random_single_push/trial_104 \
  --quiet

# Trial 105
python scripts/run_k2_jax_realtime.py \
  --height-setup outputs/physical_target_height_setups/mid_0p400_setup.json \
  --steps 7000 --profile K2_JAX_DEDICATED_DEFAULT_V3 \
  --random-single-push --push-seed 105 \
  --push-step-min 300 --push-step-max 500 \
  --push-force-min 50 --push-force-max 100 \
  --push-duration-steps 8 \
  --save-random-push-config outputs/diag/k2_v3_random_single_push/trial_105/push_config.json \
  --telemetry full --output-dir outputs/diag/k2_v3_random_single_push/trial_105 \
  --quiet
```

## 8. Exact Visual Commands for Each Trial

These commands reproduce the **exact same random push** by loading the saved push config. They do NOT sample new random pushes.

```bash
# ── Trial 101 (seed=101, 92.4N torso, roll failure) ──
python scripts/run_k2_jax_realtime.py \
  --height-setup outputs/physical_target_height_setups/mid_0p400_setup.json \
  --steps 7000 --profile K2_JAX_DEDICATED_DEFAULT_V3 \
  --visual \
  --load-random-push-config outputs/diag/k2_v3_random_single_push/trial_101/push_config.json \
  --telemetry full --output-dir outputs/visual/k2_v3_random_single_push/trial_101

# ── Trial 102 (seed=102, 83.8N l_thigh, pitch+yaw failure) ──
python scripts/run_k2_jax_realtime.py \
  --height-setup outputs/physical_target_height_setups/mid_0p400_setup.json \
  --steps 7000 --profile K2_JAX_DEDICATED_DEFAULT_V3 \
  --visual \
  --load-random-push-config outputs/diag/k2_v3_random_single_push/trial_102/push_config.json \
  --telemetry full --output-dir outputs/visual/k2_v3_random_single_push/trial_102

# ── Trial 103 (seed=103, 72.8N torso, height collapse) ──
python scripts/run_k2_jax_realtime.py \
  --height-setup outputs/physical_target_height_setups/mid_0p400_setup.json \
  --steps 7000 --profile K2_JAX_DEDICATED_DEFAULT_V3 \
  --visual \
  --load-random-push-config outputs/diag/k2_v3_random_single_push/trial_103/push_config.json \
  --telemetry full --output-dir outputs/visual/k2_v3_random_single_push/trial_103

# ── Trial 104 (seed=104, 93.6N l_thigh, height collapse + massive drift) ──
python scripts/run_k2_jax_realtime.py \
  --height-setup outputs/physical_target_height_setups/mid_0p400_setup.json \
  --steps 7000 --profile K2_JAX_DEDICATED_DEFAULT_V3 \
  --visual \
  --load-random-push-config outputs/diag/k2_v3_random_single_push/trial_104/push_config.json \
  --telemetry full --output-dir outputs/visual/k2_v3_random_single_push/trial_104

# ── Trial 105 (seed=105, 67.5N r_knee_link, yaw spinout + height collapse) ──
python scripts/run_k2_jax_realtime.py \
  --height-setup outputs/physical_target_height_setups/mid_0p400_setup.json \
  --steps 7000 --profile K2_JAX_DEDICATED_DEFAULT_V3 \
  --visual \
  --load-random-push-config outputs/diag/k2_v3_random_single_push/trial_105/push_config.json \
  --telemetry full --output-dir outputs/visual/k2_v3_random_single_push/trial_105
```

## 9. Conclusion

**Verdict: FAIL — 0/5 trials survived**

The K2 V3 controller at nominal mid-height (0.40 m) cannot survive a single random push in the 50–100 N range from any tested direction or body target. All five trials terminated within 65 steps of the push.

### Worst Trial
**Trial 104** (seed=104): 93.6 N on l_thigh, fell at step 357 (52 steps post-push), 1.25 m final displacement, severe pitch (−32.6°) and roll (31.9°), height collapse.

### Most Informative Failure
**Trial 105** (seed=105): Despite the lowest force (67.5 N), the r_knee_link target caused extreme yaw spinout (144.9°) and hip-yaw divergence (0.812 rad) — suggesting knee-level pushes are disproportionately destabilizing.

### Recommended Next Diagnostic
1. **Visual inspection of trials 104 and 105** — understand coupling modes (why does knee push cause yaw spinout?)
2. **Reduced force sweep** — find the maximum survivable force at mid-height (likely < 30 N)
3. **Direction-only ablation** — test fixed-force (e.g., 40 N) with only direction randomized
4. **Body-only ablation** — test fixed-force, fixed-direction (sagittal forward) with only body randomized
5. **Compare with V2 rollback** — is V3 worse at random-direction pushes than V2?

---

**Controller code/params:** UNCHANGED. Only `scripts/run_k2_jax_realtime.py` was modified (test-only CLI flags and push infrastructure).
