# ACC Paper Design — Anchored Cascade Controller for Wheeled Biped

**Date:** 2026-07-27
**Venue:** IEEE ICRA/IROS, 6 pages
**Language:** English
**Author:** Van Thuong Nguyen, Independent Researcher
**Controller codename in repo:** V3_ANCHOR → paper name: **Anchored Cascade Controller (ACC)**

---

## 1. Paper Identity

- **Title:** Anchored Cascade Control with Conditional Activation for Wheeled Bipedal Balance and Disturbance Recovery
- **Keywords:** Wheeled biped robot, cascade control, anchored standing, proximity-gated integral, push recovery, flight recovery, terrain adaptation
- **Type:** Classical control — no RL, no learning
- **Novelty claim:** Proximity-gated anchor mechanism + unified gate-structured cascade architecture enable a purely classical controller to achieve (i) anchored standing (0.3 mm RMS idle), (ii) omnidirectional push recovery (F_min 70 N), (iii) contact-loss recovery (drop 100 cm, ledge 50 cm), and (iv) per-leg terrain adaptation (curb 20 cm) — all without learning.

---

## 2. Paper Structure (6 pages, IEEEtran)

| Section | Title | Pages | Content |
|---------|-------|-------|---------|
| I | Introduction | 0.5 | Problem, ACC overview, 3 contributions, paper outline |
| II | Related Work | 0.5 | WBC, cascade control, disturbance recovery, terrain adaptation |
| III | Robot Platform & TWIP Model | 0.4 | Morphology, simulation, TWIP linearization, stability remark |
| IV | Unified ACC Formulation | 2.0 | Controller architecture, gate-structured equations, all 5 components |
| V | Experimental Validation | 1.0 | 4 experiments + ablation table |
| VI | Discussion | 0.3 | Limitations, sim-to-real path |
| VII | Conclusion | 0.15 | Summary |
| — | References | 0.15 | ~20 compressed references |

---

## 3. Key Design Decisions

### 3.1 Notch filter naming
- NOT "2.5 Hz WIP oscillation mode"
- IS: "2.5 Hz closed-loop standing oscillation mode (induced by the P position term), identified from FFT of pitch under P-only standing — distinct from the open-loop WIP pole at 0.7 Hz (Sec. III)"

### 3.2 Push numbers — single definition, single source of truth
- **F_min:** Over all 24 directions, the maximum push magnitude the controller survives in *every* direction (weakest-direction survival threshold)
- **F_med:** Median over 24 directions
- Headline: F_min = 70 N, F_med = 115 N (400 Nm/s harness)
- Scheduled kp: F_min 30 N → 70 N (+133%)
- Fixed kp=35 can't be baseline (idle 55 mm RMS), mentioned as tradeoff only
- Every number traces to a cell in the ablation table (Sec. V)

### 3.3 Flight recovery — role separation
- Free-fall 0–100 cm: **stress test** (bounds flight PD authority; metric: peak pitch + flip avoidance)
- Ramp-ledge 20–50 cm: **use case** (real navigation; metric: standing recovery after landing)
- Explicit role-separation sentence in Sec. IV-E

### 3.4 No duplicate tables
- Novelty claims in Sec. IV-C are 3 inline sentences with pointers to abutment tables in Sec. V
- Table I (platform comparison) at end of Sec. IV, after all components presented

### 3.5 Gate language
- "Each component is activated continuously by smoothstep gates on physical conditions — not discrete switching"
- Avoid binary "AND" in overview; use "gated continuously by proximity and quietness"

### 3.6 Units explicit everywhere
- "CoM sagittal idle RMS: 0.3 mm (ACC) vs 55 mm (P-only limit cycle)"

### 3.7 Authority separation
- Balance → wheel torque
- Anchor/posture → leg-joint torque
- Terrain → per-leg height offset
- Flight PD → wheel torque override (only during contact loss)

---

## 4. Section IV — Detailed Blueprint

### IV-A: Controller Overview (0.25 pages)

Single equation:
```
τ_final = τ_balance + g_anchor · τ_anchor + g_flight · τ_flight
        + g_terrain · τ_terrain + τ_posture
```

Each gate `g_*` is a continuous smoothstep function of physical conditions. One figure: unified torque assembly diagram.

Authority separation paragraph.

### IV-B: Sagittal Balance Core (τ_balance) (0.3 pages)

```
τ_balance = τ_pitch + τ_damping + τ_position_P

τ_pitch      = kp(θ) · θ + kd · θ̇_notch
τ_damping    = −k_vel · v_sag
τ_position_P = −k_pos · clip(Δx, ±0.10m)
```

- Notch filter on pitch_rate: removes 2.5 Hz closed-loop oscillation mode
- kp scheduled: 50 (anchor) / 35 (ballistic recovery)
- P-only position: sufficient to keep robot within ±4-6 cm of home
- P-only creates ~1.3 Nm equilibrium bias → limit cycle → motivates anchor integral

### IV-C: Proximity-Gated Anchor (g_anchor · τ_anchor) (0.5 pages)

```
τ_anchor = K_i ∫ g_prox · g_env · (−Δx) dt + g_boost · K_boost · (−v_sag)

where:
  g_prox(Δx)   = smoothstep(|Δx|, 0.05, 0.15)
  g_env(v_sag)  = 1 − smoothstep(EMA(v_sag), 0.18, 0.30)
  g_boost       = g_prox · g_env · g_θ(θ, θ̇)

  EMA(v): attack α=0.35 (τ≈30 ms), release α=0.007 (τ≈1.5 s)
```

**Why separate P (B) and I (C):** P is global, always active, saturates at ±4 Nm — pulls robot toward home. I is local, gated, saturates at ±2 Nm — cancels equilibrium bias only when robot is near home and quiet. Global I windups during pushes and prevents recovery; the proximity gate protects I from windup while ensuring I does not fight the ballistic catch.

**3 novelty claims (inline, each pointing to Sec. V):**
1. *Asymmetric envelope follower* distinguishes true quiet stance from post-push ringdown (validated by symmetric-EMA ablation, Sec. V-1)
2. *Proximity-gated integral* confines I to anchor neighborhood while preserving P-only displaced-return outside 15 cm (validated by global-I ablation, Sec. V-1)
3. *Scheduled pitch stiffness* raises F_min from 30 N to 70 N (+133%) while maintaining 0.3 mm idle (validated by fixed-kp ablation, Sec. V-2)

### IV-D: Posture & Yaw Stability (τ_posture) (0.3 pages)

```
τ_posture = τ_jacobian_PD(q, q̇, h_cmd)
          + τ_divergence(q_hy^L, q_hy^R)
          + g_settled · τ_homing(q − q_ref)
          + τ_yaw_return(Δψ)
```

- Jacobian-based gain grid (23 height points) for consistent foot-level stiffness
- Homing PD on hip_roll + hip_yaw: stability-gated, restores narrow stance after push
- Wheel-differential yaw return
- Grid vs smoothstep remark: "posture uses a Jacobian grid because foot-level stiffness is nonlinear in height; balance gains are scalar and smoothstep-scheduled"

### IV-E: Contact-Loss Recovery (g_flight · τ_flight) (0.25 pages)

```
g_flight engages when Fz^L < 0.5mg AND Fz^R < 0.5mg (≥2 steps)
g_flight releases when min(Fz^L, Fz^R) ≥ 0.5mg (≥5 steps) + 150 ms ramp

τ_flight = clip(2.0·θ + 0.5·θ̇ − 0.02·ω_wheel, ±2 Nm)
```

**Role separation:** Free-fall drop 0–100 cm is a stress test (bounds flight PD attitude authority; metric: peak pitch + flip avoidance). Ramp-to-ledge drive-off 20–50 cm is the use case (real navigation; metric: standing recovery after landing).

Wheels act as reaction flywheels: spinning forward pitches nose up. 5-step release hysteresis + 150 ms soft handback prevent re-launch during bounce chatter.

### IV-F: Per-Leg Ground Adaptation (g_terrain · τ_terrain) (0.15 pages)

```
h_ground^L,R measured from contact-point z when loaded (Fz ≥ 0.2mg)
Δh = h_ground^L − h_ground^R
h_cmd^L,R = h_cmd ∓ Δh/2  (split height commands)
```

Ground measured at contact point (exact at any tilt), not wheel center. Unloaded wheel freezes estimate; seeks down only when wheel z < believed ground − 1 cm.

### IV-Foot: Platform Comparison Table (Table I)

| Feature | Ascento | DIABLO | Whleaper | SKATER | **ACC (Ours)** |
|---------|---------|--------|----------|--------|-----------------|
| Wheeled biped | ✓ | ✓ | ✓ | ✓ | ✓ |
| Height-variable stance | — | ✓ | — | — | ✓ |
| Anchored standing (mm-level) | — | — | — | — | ✓ |
| Omnidirectional push recovery | — | — | — | — | ✓ |
| Contact-loss recovery | — | — | — | — | ✓ |
| Per-leg terrain adaptation | — | — | — | — | ✓ |

---

## 5. Section V — Experimental Validation

### 5.1 Experiment-Ablation Mapping

| # | Experiment | Component | Ablation | Headline Metric |
|---|-----------|-----------|----------|-----------------|
| 1 | Anchored standing | Anchor (env+I+boost) | P-only, symmetric EMA, global I | Idle 0.3 mm, ringdown→0 in 9 s |
| 2 | Push recovery (24-dir, 10–160 N) | Scheduled kp + anchor gates | fixed kp=50, fixed kp=35, no boost | F_min 70 N, F_med 115 N |
| 3 | Contact loss (drop 100 cm, ledge 20–50 cm) | Flight PD + gate logic | No flight PD, no soft handback | Peak pitch, survival |
| 4 | Uneven terrain (curb 10–20 cm, oblique) | Per-leg adaptation | No adaptation, wheel-center estimate | Straddle roll, survival |

### 5.2 Unified Push Ablation Table (definitive source of push numbers)

| Configuration | F_min (N) | F_med (N) | Idle CoM RMS (mm) | Notes |
|---|---|---|---|---|
| **Full ACC (scheduled kp)** | **70** | **115** | **0.3** | Headline |
| Fixed kp = 50 | 30 | 65 | 0.3 | Idle good, capture narrow |
| Fixed kp = 35 | [?] | [?] | 55 (osc.) | Tradeoff only: capture wide, idle unusable |
| No anchor boost | [?] | [?] | ringdown ∞ | Post-push oscillation never decays |

### 5.3 Remaining Tables
- Standing idle comparison (ACC vs P-only vs HOMING)
- Contact-loss recovery results (drop + ledge scenarios)
- Terrain adaptation results (curb scenarios)
- Optional: ringdown time-series figure for push recovery

---

## 6. References

~20 references covering:
- Ascento (Klemm 2019, 2020) — wheeled biped WBC
- DIABLO (Liu 2024) — 6-DOF direct-drive wheeled biped
- Whleaper (Zhu 2024) — 10-DOF wheeled biped LQR+PPO
- SKATER (?) — wheeled biped
- Cui (?) — bipedal impact/contact-loss
- Cascade control literature
- WIP/TWIP analytical foundations
- Contact detection / flight phase control
- Terrain adaptation for legged robots

---

## 7. Implementation Plan

### Phase 1: ARS Plan (Socratic chapter planning)
### Phase 2: ARS Lit Review (annotated bibliography)
### Phase 3: Manual Technical Writing (Sec. III–VII)
### Phase 4: ARS Reviewer (simulated peer review)
### Phase 5: Manual Revision + Finalize

---

## 8. Self-Review

- [x] No TODOs — all placeholders are `[?]` for future measurement
- [x] Push numbers: single definition (F_min, F_med), single table (Sec. V-2)
- [x] Flight: role separation (stress test vs use case) explicit
- [x] No duplicate tables: novelty in C is inline, ablation only in V
- [x] Gate language: continuous smoothstep, not binary
- [x] Units on all metrics
- [x] Authority separation explicit in IV-A
- [x] Grid vs smoothstep remark in IV-D
- [x] P vs I separation rationale in IV-B + IV-C
- [x] 2.5 Hz notch correctly attributed to closed-loop P-induced oscillation
- [x] Table I (platform comparison) lives at end of IV
