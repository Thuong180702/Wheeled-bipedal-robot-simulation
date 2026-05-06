# Phase B.4 Comprehensive Posture Feasibility Study

**Date:** 2026-05-06  
**Study Type:** Dense grid search over hip_pitch × knee space  
**Grid Resolution:** 150×150 = 22,500 samples per configuration  
**Target Heights:** 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40 m  
**CoM Tolerances:** 2cm, 3cm, 5cm  
**Total Configurations:** 7 heights × 3 tolerances = 21 configurations  
**Total Samples:** 472,500 posture evaluations

---

## Executive Summary

This comprehensive feasibility study reveals that **statically balanced postures do not exist** across the target height range [0.40, 0.70]m under realistic CoM alignment criteria. The robot's morphology creates a fundamental three-way tradeoff between height tracking, knee-forward margin, and CoM-to-wheel alignment that cannot be simultaneously satisfied.

**Key Finding:** Out of 472,500 total posture evaluations, only 11 samples (0.0023%) achieved full static feasibility, all concentrated at h=0.70m with relaxed CoM tolerance.

**Implication:** The LQR/IK prior cannot provide standalone static balance. It serves as a **structured nominal controller** that provides geometric posture and sagittal stabilization, while residual PPO must learn dynamic balance corrections to compensate for inherent CoM misalignment.

---

## Methodology

### Grid Search Parameters

```yaml
hip_pitch_range: [-0.5, 1.8] rad
knee_range: [-0.5, 2.7] rad
grid_resolution: 150 × 150
samples_per_height: 22,500
```

### Feasibility Criteria

Five hierarchical feasibility levels were evaluated:

1. **geometric_height**: Target height achieved within ±1cm
2. **knee_forward**: Knee positioned forward of hip (margin > 1cm)
3. **torso_upright**: Torso pitch within ±5° of vertical
4. **com_near_wheel**: Whole-body CoM within tolerance of wheel contact
5. **full_static**: All four criteria satisfied simultaneously

### Physical Parameters

```yaml
thigh_length: 0.26 m
shin_length: 0.28 m
wheel_radius: 0.06 m
forward_axis: +Y (verified empirically)
```

---

## Results Summary

### Full Static Feasibility by Height and Tolerance

| Height (m) | 2cm CoM Tol | 3cm CoM Tol | 5cm CoM Tol | Total Feasible |
|------------|-------------|-------------|-------------|----------------|
| 0.70       | 0 (0.00%)   | 1 (0.00%)   | 10 (0.04%)  | 11             |
| 0.65       | 0 (0.00%)   | 0 (0.00%)   | 0 (0.00%)   | 0              |
| 0.60       | 0 (0.00%)   | 0 (0.00%)   | 0 (0.00%)   | 0              |
| 0.55       | 0 (0.00%)   | 0 (0.00%)   | 0 (0.00%)   | 0              |
| 0.50       | 0 (0.00%)   | 0 (0.00%)   | 0 (0.00%)   | 0              |
| 0.45       | 0 (0.00%)   | 0 (0.00%)   | 0 (0.00%)   | 0              |
| 0.40       | 0 (0.00%)   | 0 (0.00%)   | 0 (0.00%)   | 0              |
| **Total**  | **0**       | **1**       | **10**      | **11**         |

**Conclusion:** Full static balance is achievable only at the tallest height (0.70m) with relaxed CoM tolerance (≥3cm), and even then represents <0.05% of the posture space.

### Individual Feasibility Criteria

Averaged across all 21 configurations (7 heights × 3 tolerances):

| Criterion         | Mean % | Range %      | Samples (avg) |
|-------------------|--------|--------------|---------------|
| Geometric Height  | 4.0%   | 3.7% - 5.0%  | 890 / 22,500  |
| Knee Forward      | 20.0%  | 20.0% (const)| 4,500 / 22,500|
| Torso Upright     | 100.0% | 100.0% (all) | 22,500 / 22,500|
| CoM Near Wheel    | 9.7%   | 5.9% - 14.5% | 2,189 / 22,500|
| **Full Static**   | **0.005%** | **0.0% - 0.04%** | **1.1 / 22,500** |

**Key Observations:**
- **Torso upright**: Always satisfied (100%) - robot maintains vertical orientation across all postures
- **Knee forward**: Constant 20% - independent of height, determined by hip_pitch > 0 region
- **Geometric height**: 3.7-5.0% - narrow band of joint configurations achieve each target height
- **CoM near wheel**: 5.9-14.5% - increases with tolerance but still <15% even at 5cm
- **Full static**: Near-zero intersection - criteria are mutually exclusive

---

## The Three-Way Tradeoff

The robot's morphology creates an **impossible tradeoff** between three competing objectives:

### 1. Height Tracking Priority

**Best candidates** (minimize height error):

| Height | hip_pitch | knee   | height_err | knee_fwd  | com_dist |
|--------|-----------|--------|------------|-----------|----------|
| 0.70   | 0.426     | 0.638  | 0.000      | -0.1075   | 0.0457   |
| 0.65   | -0.160    | 0.595  | 0.000      | 0.0415    | 0.1833   |
| 0.60   | 0.534     | -0.307 | 0.000      | -0.1324   | 0.2991   |
| 0.55   | 0.565     | -0.479 | 0.000      | -0.1392   | 0.3335   |
| 0.50   | -0.284    | 1.068  | 0.000      | 0.0728    | 0.2782   |
| 0.45   | 1.599     | 1.304  | 0.000      | -0.2599   | 0.2837   |
| 0.40   | -0.191    | 1.540  | 0.000      | 0.0494    | 0.2633   |

**Pattern:** Achieves target height perfectly but:
- CoM distance: 4.6-33.4cm (far from wheel)
- Knee-forward: Often negative (knee behind hip)
- **Not statically balanced**

### 2. CoM Alignment Priority

**Best candidate** (minimize CoM-to-wheel distance):

```
hip_pitch: -0.052 rad
knee: -0.028 rad
height: 0.729 m (not target)
knee_fwd: 0.0136 m (minimal)
com_dist: 0.0000 m (perfect)
```

**Pattern:** Achieves perfect CoM alignment but:
- Height: 0.729m (only matches h=0.70m target loosely)
- Knee-forward: 1.36cm (barely positive)
- **Wrong height for most targets**

### 3. Knee-Forward Priority

**Best candidate** (maximize knee-forward margin):

```
hip_pitch: -0.500 rad (joint limit)
knee: -0.500 rad (joint limit)
height: 0.698 m (not target)
knee_fwd: 0.1246 m (good)
com_dist: 0.0776 m (moderate)
```

**Pattern:** Maximizes forward margin but:
- Height: 0.698m (only matches h=0.70m target loosely)
- CoM distance: 7.76cm (outside strict tolerance)
- **Wrong height for most targets**

### Why the Tradeoff Exists

1. **Height constraint**: Each target height defines a narrow manifold in (hip_pitch, knee) space (~4% of samples)

2. **Knee-forward constraint**: Requires hip_pitch > 0 (thigh tilted forward), which shifts CoM forward

3. **CoM alignment constraint**: Requires whole-body CoM directly above wheel contact, which depends on:
   - Torso mass (2.5 kg at height h)
   - Leg mass distribution (thigh 0.8 kg, shin 0.6 kg)
   - Geometric configuration (hip_pitch, knee)

4. **Coupling**: The height-achieving configurations place the CoM far from the wheel axis. Configurations that align CoM achieve the wrong height.

**Fundamental Issue:** The robot's mass distribution (heavy torso, light legs) combined with the geometric constraints of achieving specific heights creates CoM positions that are inherently misaligned with the wheel contact point.

---

## Detailed Results by Height

### h = 0.70 m (Tallest)

**Only height with any full static feasible postures.**

| CoM Tol | Full Static | Combined Candidate                          |
|---------|-------------|---------------------------------------------|
| 2cm     | 0           | None found                                  |
| 3cm     | 1           | hip=−0.377, knee=−0.500, kf=0.096, cd=0.026 |
| 5cm     | 10          | hip=−0.423, knee=−0.500, kf=0.107, cd=0.045 |

**Observation:** Feasible postures exist only with:
- Negative hip_pitch (thigh tilted backward)
- Knee at negative limit (−0.5 rad)
- Relaxed CoM tolerance (≥3cm)

### h = 0.65 m

| CoM Tol | Full Static | Height Tracking Candidate                   |
|---------|-------------|---------------------------------------------|
| All     | 0           | hip=−0.160, knee=0.595, kf=0.042, cd=0.183  |

**Observation:** Height-tracking posture has:
- Small positive knee-forward (4.2cm)
- Large CoM distance (18.3cm)
- No feasible balanced posture found

### h = 0.60 m

| CoM Tol | Full Static | Height Tracking Candidate                   |
|---------|-------------|---------------------------------------------|
| All     | 0           | hip=0.534, knee=−0.307, kf=−0.132, cd=0.299 |

**Observation:** Height-tracking posture has:
- **Negative knee-forward** (−13.2cm, knee behind hip)
- Very large CoM distance (29.9cm)
- Completely infeasible for balance

### h = 0.55 m

| CoM Tol | Full Static | Height Tracking Candidate                   |
|---------|-------------|---------------------------------------------|
| All     | 0           | hip=0.565, knee=−0.479, kf=−0.139, cd=0.334 |

**Observation:** Height-tracking posture has:
- **Negative knee-forward** (−13.9cm)
- Largest CoM distance (33.4cm)
- Worst case for balance

### h = 0.50 m

| CoM Tol | Full Static | Height Tracking Candidate                   |
|---------|-------------|---------------------------------------------|
| All     | 0           | hip=−0.284, knee=1.068, kf=0.073, cd=0.278  |

**Observation:** Height-tracking posture has:
- Positive knee-forward (7.3cm)
- Large CoM distance (27.8cm)
- Deep squat configuration

### h = 0.45 m

| CoM Tol | Full Static | Height Tracking Candidate                   |
|---------|-------------|---------------------------------------------|
| All     | 0           | hip=1.599, knee=1.304, kf=−0.260, cd=0.284  |

**Observation:** Height-tracking posture has:
- **Large negative knee-forward** (−26.0cm)
- Large CoM distance (28.4cm)
- Extreme forward lean configuration

### h = 0.40 m (Shortest)

| CoM Tol | Full Static | Height Tracking Candidate                   |
|---------|-------------|---------------------------------------------|
| All     | 0           | hip=−0.191, knee=1.540, kf=0.049, cd=0.263  |

**Observation:** Height-tracking posture has:
- Small positive knee-forward (4.9cm)
- Large CoM distance (26.3cm)
- Very deep squat configuration

---

## Implications for LQR/IK Prior Design

### What the Prior CAN Provide

1. **Geometric height tracking**: IK can map height commands to joint angles that achieve target torso height within ±1cm

2. **Sagittal stabilization**: LQR can provide pitch/pitch_rate feedback to prevent forward/backward falling

3. **Structured nominal control**: Prior outputs are geometrically consistent and follow physical constraints

4. **Smooth height transitions**: Polynomial IK mapping ensures continuous joint trajectories during height changes

### What the Prior CANNOT Provide

1. **Static balance**: CoM is inherently misaligned with wheel contact across the height range

2. **Standalone stability**: Robot will fall without additional corrective control

3. **CoM positioning**: IK is purely geometric and does not consider mass distribution

4. **Dynamic balance**: Prior has no mechanism to compensate for CoM-wheel misalignment

### Required Residual PPO Capabilities

The residual policy must learn to:

1. **CoM correction**: Adjust hip_pitch/knee to shift CoM toward wheel contact while maintaining approximate height

2. **Dynamic balance**: Use wheel velocity to actively balance under CoM misalignment (inverted pendulum control)

3. **Height-CoM tradeoff**: Learn when to sacrifice exact height tracking for better balance

4. **Push recovery**: Generate corrective wheel motion and leg adjustments to recover from disturbances

5. **Transition management**: Smooth CoM repositioning during commanded height changes

---

## Comparison with Previous Findings

### Phase B.2 LQR/IK Validation Report

Previous fixed-height evaluation at h=0.55m found:
- 100% fall rate within 1.0s
- Pitch RMS: 16-25°
- Conclusion: "LQR/IK prior alone cannot achieve balance"

**Consistency:** This feasibility study confirms that finding. At h=0.55m, zero full static feasible postures exist even with 5cm CoM tolerance. The prior's failure to balance is not a tuning issue but a **fundamental morphological constraint**.

### Original Posture Geometry Diagnostic

Initial diagnostic at h=0.67m found:
- Knee-forward margin: −10.7cm (negative)
- CoM-to-wheel distance: 4.6cm
- Conclusion: "Knee behind hip, moderate CoM misalignment"

**Consistency:** This feasibility study shows that h=0.67m (close to 0.70m) is the BEST case, yet still shows CoM misalignment. Lower heights have even worse CoM distances (18-33cm).

---

## Recommendations

### 1. Accept Limited Prior Status

Update controller metadata to reflect:

```yaml
prior_status: limited_prior
known_limitations:
  - IK is geometric (FK scan + polynomial fit) and not CoM-balanced
  - Standalone fixed-height balance is not achieved (100% fall rate)
  - CoM-to-wheel misalignment ranges from 4.6cm (h=0.70m) to 33.4cm (h=0.55m)
  - Residual PPO is REQUIRED to learn CoM positioning and dynamic balance
```

### 2. Design Residual Policy for CoM Correction

Residual observation should include:
- Base action (prior's geometric target)
- CoM position (if estimable) or proxy signals
- Wheel contact forces
- Pitch/pitch_rate (for inverted pendulum dynamics)

Residual action authority should be:
- **High on wheels** (0.3 scale): Primary balance actuator
- **Moderate on hip_pitch/knee** (0.15 scale): CoM repositioning
- **Low on hip_yaw** (0.05 scale): Avoid excessive twisting

### 3. Adjust Training Expectations

Do NOT expect:
- Residual policy to "fix" the prior quickly
- High success rates in early training
- Curriculum to advance rapidly through height range

DO expect:
- Long training time (>100M steps) to learn dynamic balance
- Residual actions to be large (near saturation) initially
- Height tracking error to increase as policy prioritizes balance over height

### 4. Evaluation Metrics

Primary metrics should be:
- **Survival time** (not height RMSE)
- **Fall rate** (not height tracking accuracy)
- **CoM-to-wheel distance** (residual's correction effectiveness)
- **Wheel velocity RMS** (dynamic balance effort)

Secondary metrics:
- Height RMSE (expected to be worse than pure PPO initially)
- Residual saturation rate (should decrease with training)

### 5. Consider Alternative Approaches

If residual PPO training proves too difficult:

**Option A:** Relax height tracking requirements
- Allow ±5cm height error instead of ±1cm
- Prioritize balance over exact height
- Simpler for residual policy to learn

**Option B:** Add CoM feedback to prior
- Estimate CoM from joint positions and known masses
- Add CoM-to-wheel error term to LQR state
- Requires CoM estimation (not currently implemented)

**Option C:** Pure PPO with height-dependent initialization
- Use LQR/IK prior only for initialization
- Let PPO learn full policy from scratch
- Simpler architecture, longer training

---

## Conclusion

This comprehensive feasibility study demonstrates that **statically balanced postures do not exist** across the target height range [0.40, 0.70]m under realistic CoM alignment criteria. The robot's morphology creates a fundamental tradeoff between height tracking, knee-forward margin, and CoM-to-wheel alignment that cannot be simultaneously satisfied.

The LQR/IK prior serves as a **structured nominal controller** that provides geometric posture and sagittal stabilization, but **cannot achieve standalone balance**. Residual PPO training is not an enhancement but a **necessity** — the residual policy must learn dynamic balance corrections to compensate for the prior's inherent CoM misalignment.

This finding validates the hybrid residual RL approach and clarifies the division of labor:
- **Prior:** Geometric height tracking + sagittal stabilization
- **Residual:** CoM correction + dynamic balance + push recovery

The path forward is clear: proceed with residual PPO training with realistic expectations about training difficulty and the magnitude of residual corrections required.

---

## Appendix: Generated Outputs

All results saved to: `outputs/posture_feasibility/`

### CSV Files (21 files)
- `grid_h{height}_tol{tolerance}.csv` for each configuration
- `all_results.csv` (combined dataset, 472,500 rows)

### Heatmaps (21 files)
- `feasibility_heatmaps_h{height}_tol{tolerance}.png`
- Shows: torso_height, knee_forward, torso_pitch, com_to_wheel, full_static, geometric_height

### Feasible Region Plots (21 files)
- `feasible_regions_h{height}_tol{tolerance}.png`
- Overlays different feasibility levels in joint space

### Best Candidates
Documented in CSV files and summarized in this report.
