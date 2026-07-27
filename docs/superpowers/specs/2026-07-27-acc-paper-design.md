# ACC Paper Design — Anchored Cascade Controller for Wheeled Biped

**Date:** 2026-07-27
**Venue:** IEEE ICRA/IROS, 6 pages (content incl. figures — refs +2 pages if policy permits)
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

## 2. Paper Structure (6 pages, IEEEtran, FIGURES INCLUDED)

| Section | Title | Pages | Content |
|---------|-------|-------|---------|
| I | Introduction | 0.45 | Problem, ACC overview, 3 contributions, paper outline |
| II | Related Work | 0.45 | WBC, cascade control, disturbance recovery, terrain adaptation |
| III | Robot Platform & TWIP Model | 0.35 | Morphology, simulation, TWIP linearization, stability remark |
| IV | Unified ACC Formulation | 1.7 | Controller architecture, two-channel torque assembly, all components |
| V | Experimental Validation | 0.8 | 4 experiments + unified ablation table |
| VI | Discussion | 0.25 | Limitations, sim-to-real path |
| VII | Conclusion | 0.15 | Summary |
| — | References | 0.15 | ~25-30 references |

**Figure budget (~1.8 pages total, embedded in text flow):**

| Fig | Content | Size | Section |
|-----|---------|------|---------|
| 1 | Robot joint layout (reuse annotated_robot_joints.png) | 0.25 col | III |
| 2 | Unified ACC architecture (two-channel torque diagram) | 0.35 2-col | IV-A |
| 3 | Anchor gate structure (prox + env + boost gates, block diagram) | 0.25 col | IV-C |
| 4 | Push recovery time series (scheduled vs fixed kp, pitch + wheel vel) | 0.30 col | V-2 |
| 5 | Omnidirectional polar push envelope (24-dir, ACC vs ablation) | 0.30 col | V-2 |
| 6 | Contact-loss + terrain composite (drop pitch trace + curb straddle snapshot) | 0.30 col | V-3/V-4 |

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
- Free-fall 0–100 cm: **stress test** (bounds flight PD attitude authority; metric: peak pitch + flip avoidance)
- Ramp-ledge 20–50 cm: **use case** (real navigation; metric: standing recovery after landing)
- Explicit role-separation sentence in Sec. IV-E

### 3.4 No duplicate tables
- Novelty claims in Sec. IV-C are 3 inline sentences with pointers to ablation tables in Sec. V
- Table I (platform comparison) at end of Sec. IV, after all components presented

### 3.5 Gate language
- "Each component is activated continuously by smoothstep gates on physical conditions — not discrete switching"
- Avoid binary "AND" in overview; use "gated continuously by proximity and quietness"

### 3.6 Units explicit everywhere
- "CoM sagittal idle RMS: 0.3 mm (ACC) vs 55 mm (P-only limit cycle)"

### 3.8 Introduction narrative — tension → resolution (from Q1)

**¶2 (TENSION — two concrete axes, NOT the resolution):**
- Stiffness axis: kp=50 → idle 0.3 mm ✅ / F_min=30 N ❌ vs kp=35 → capture wide ✅ / idle oscillates ❌
- Integral axis: no-I → 1.3 Nm bias → 55 mm limit cycle ❌ vs global-I → windup during push ❌
- "A single fixed stiffness cannot occupy both ends: kp=50 holds 0.3 mm idle but shrinks the capture basin to F_min=30 N, while kp=35 widens capture but lets idle oscillate. The same dilemma appears on the integral axis: omitting I leaves a 1.3 Nm equilibrium bias, yet a global I winds up during a push and blocks recovery."

**¶3 (RESOLUTION — ACC breaks both trade-offs):**
- Scheduled kp: kp=50 at anchor → 0.3 mm idle; kp=35 during ballistic recovery → F_min=70 N (+133% over fixed kp=50)
- Proximity-gated I: I active only within |Δx|<15 cm after quiet-stance detection
- Asymmetric envelope follower: the mechanism that reliably detects quiet stance

### 3.9 Authority separation (FIXED N1 — now two-channel)
- **Wheel torque channel (τ_w ∈ R²):** τ_balance + g_anchor · τ_anchor + g_flight · τ_flight
- **Leg-joint torque channel (τ_q ∈ R⁸):** τ_posture(h_cmd) + g_terrain · Δτ_posture(split_h_cmd)
- Anchor integral accumulates in the wheel channel (sagittal position error → wheel torque correction) — this is the principled choice: cancelling equilibrium bias at the wheels eliminates the limit cycle at its source, rather than compensating through posture shift
- Flight PD substitutes wheel torque entirely during contact loss
- Terrain modifies posture targets (leg-joint position commands), not raw torque

---

## 4. Section IV — Detailed Blueprint

### IV-A: Controller Overview (0.25 pages)

Two-channel torque assembly (FIXED N1):

```
Wheel torque:        τ_w = τ_balance + g_anchor · τ_anchor + g_flight · τ_flight
Leg-joint torque:    τ_q = τ_posture(h_cmd) + g_terrain · Δτ_posture(h_cmd^L, h_cmd^R)
```

Each gate `g_*` is a **continuous smoothstep function** of physical conditions (proximity, quietness, contact, terrain) — not a discrete switch. The two channels are assembled independently: wheel torques control sagittal/lateral/yaw balance; leg-joint torques control posture and per-leg height adaptation.

One figure (Fig. 2): two-channel torque assembly diagram with gate inputs.

Authority separation paragraph:
- Balance (τ_balance) and anchor (τ_anchor) command wheel torque — the sagittal balance problem is fundamentally a wheel-control problem
- Posture (τ_posture) commands leg-joint torque via Jacobian-based PD
- Flight PD (τ_flight) overrides wheel torque during contact loss — wheels become reaction flywheels
- Terrain adaptation modifies leg-joint targets by splitting the height command per leg

### IV-B: Sagittal Balance Core (τ_balance) (0.25 pages)

```
τ_balance ∈ R² (common-mode + differential on two wheels)

τ_pitch      = kp(h, v) · θ + kd · θ̇_notch
τ_damping    = −k_vel · v_sag
τ_position_P = −k_pos · clip(Δx, ±0.10m)
```

- Notch filter on pitch_rate: removes 2.5 Hz closed-loop oscillation mode
- kp scheduled: 50 (anchor) / 35 (ballistic recovery)
- P-only position: sufficient to keep robot within ±4-6 cm of home
- P-only creates ~1.3 Nm equilibrium bias → limit cycle → motivates anchor integral

### IV-C: Proximity-Gated Anchor (g_anchor · τ_anchor) (0.4 pages)

```
τ_anchor = K_i ∫ g_prox · g_env · (−Δx) dt + g_boost · K_boost · (−v_sag)

where (FIXED N2 — all gates = 1 when "at home and quiet"):
  g_prox(Δx)   = 1 − smoothstep(|Δx|, 0.05, 0.15)     // 1 near home, →0 beyond 15cm
  g_env(v_sag)  = 1 − smoothstep(EMA(v_sag), 0.18, 0.30) // 1 when quiet, →0 when oscillating
  g_boost       = g_prox · g_env · g_θ(θ, θ̇)           // 4-gate damping boost

  EMA(v): asymmetric — attack α=0.35 (τ≈30 ms), release α=0.007 (τ≈1.5 s)
```

**Why separate P (B) and I (C):** P is global, always active, saturates at ±4 Nm — pulls robot toward home. I is local, gated, saturates at ±2 Nm — cancels equilibrium bias only when robot is near home AND quiet. Global I windups during pushes and prevents recovery; the proximity gate protects I from windup while ensuring I does not fight the ballistic catch. Both P and I act in the **wheel torque channel** — the bias originates from wheels, and the correction targets wheels.

**3 novelty claims (inline, each pointing to Sec. V):**
1. *Asymmetric envelope follower* distinguishes true quiet stance from post-push ringdown — solves the bistable failure of symmetric EMA (validated by symmetric-EMA ablation, Sec. V-1)
2. *Proximity-gated integral* confines I to the anchor neighborhood (|Δx|<15 cm), preserving P-only displaced-return outside (validated by global-I ablation, Sec. V-1)
3. *Scheduled pitch stiffness* raises F_min from 30 N (fixed kp=50) to 70 N (+133%) while maintaining 0.3 mm idle (validated by fixed-kp ablation, Sec. V-2)

One figure (Fig. 3): gate structure block diagram showing prox, env, boost gates and their physical inputs.

### IV-D: Posture & Yaw Stability (τ_posture) (0.25 pages)

```
τ_posture ∈ R⁸ (4 leg joints × 2 legs)

τ_posture = τ_jacobian_PD(q, q̇, h_cmd)
          + τ_divergence(q_hy^L, q_hy^R)
          + g_settled · τ_homing(q − q_ref)
          + τ_yaw_return(Δψ)
```

- Jacobian-based gain grid (23 height points) for consistent foot-level stiffness
- Homing PD on hip_roll + hip_yaw: stability-gated, restores narrow stance after push
- Wheel-differential yaw return (wheel torque channel)
- Grid vs smoothstep remark: "posture uses a Jacobian grid because foot-level stiffness is nonlinear in height; balance gains are scalar and smoothstep-scheduled"

### IV-E: Contact-Loss Recovery (g_flight · τ_flight) (0.25 pages)

```
g_flight engages when Fz^L < 0.5mg AND Fz^R < 0.5mg (≥2 steps)
g_flight releases when min(Fz^L, Fz^R) ≥ 0.5mg (≥5 steps) + 150 ms ramp

τ_flight ∈ R² (wheel torque override)
τ_flight = clip(2.0·θ + 0.5·θ̇ − 0.02·ω_wheel, ±2 Nm)
```

**Role separation:** Free-fall drop 0–100 cm is a stress test (bounds flight PD attitude authority; metric: peak pitch + flip avoidance — "does the controller survive extreme contact loss?"). Ramp-to-ledge drive-off 20–50 cm is the use case (real navigation; metric: standing recovery after landing — "does the robot continue operating after a real ledge?").

Wheels act as reaction flywheels: spinning forward pitches nose up. 5-step release hysteresis + 150 ms soft handback prevent re-launch during bounce chatter.

### IV-F: Per-Leg Ground Adaptation (g_terrain) (0.15 pages)

```
h_ground^L,R measured from contact-point z when loaded (Fz ≥ 0.2mg)
Δh = h_ground^L − h_ground^R
h_cmd^L,R = h_cmd ∓ Δh/2  (split height commands, clamped to ±5 cm + 5 cm extrapolation)

Δτ_posture = τ_posture(h_cmd^L, h_cmd^R) − τ_posture(h_cmd, h_cmd)
```

Ground measured at contact point (exact at any tilt), not wheel center. Unloaded wheel freezes estimate; seeks down only when wheel z < believed ground − 1 cm (the only certain "ground has disappeared" signal).

**Channel:** terrain outputs a *differential posture torque*, not a raw torque added to τ_q. The split height commands feed into the same Jacobian PD as τ_posture, producing a correction relative to the uniform-height baseline.

### IV-Foot: Platform Comparison Table (Table I)

| Feature | Ascento | DIABLO | Whleaper | SKATER | **ACC (Ours)** |
|---------|---------|--------|----------|--------|-----------------|
| Wheeled biped | ✓ | ✓ | ✓ | ✓ | ✓ |
| Gain-scheduled balance across height range | — | — | — | — | ✓ |
| Anchored standing (mm-level idle) | — | — | — | — | ✓ |
| Omnidirectional push recovery¹ | — | — | — | — | ✓ |
| Contact-loss recovery (flight control) | — | — | — | — | ✓ |
| Per-leg terrain adaptation (no exteroception) | — | — | — | — | ✓ |

¹ Quantified via 24-direction polar threshold sweep (F_min, F_med), not qualitative stick-strike.

**FIXED N4:** "Height-variable stance" → "Gain-scheduled balance across height range" (Ascento adjusts leg length for jumping, not continuous gain scheduling for balance; Whleaper has multiple postures but no height-scheduled balance gains). Column now reflects what ACC uniquely does.

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
| Fixed kp = 35 | n/a | n/a | 55 (osc.) | Idle unstable — cannot serve as baseline |
| No anchor boost | [?] | [?] | ringdown ∞ | Post-push oscillation never decays |
| No anchor integral (P-only) | [?] | [?] | 55 (limit cycle) | Equilibrium bias sustains ±4-6 cm cycle |

**FIXED N6:** "Fixed kp=35 → F_min [?]" replaced with "n/a (idle unstable)". Only genuinely measurable-but-not-yet-measured cells use [?].

### 5.3 Ablation Truth-Table (definitive configuration matrix)

Each row disables exactly ONE mechanism; all others kept at ACC defaults. This eliminates label ambiguity (e.g. "55 mm" was previously attributed to both "P-only" and "fixed kp=35").

| Configuration | sched kp | integral I | prox gate | envelope | boost | → idle RMS | → F_min |
|---|---|---|---|---|---|---|---|
| **Full ACC** | ✓ | gated | ✓ | asym | ✓ | 0.3 mm | 70 N |
| fixed kp=50 | ✗ (=50) | gated | ✓ | asym | ✓ | 0.3 mm | 30 N |
| fixed kp=35 | ✗ (=35) | gated | ✓ | asym | ✓ | n/a (osc.) | n/a |
| global-I | ✓ | global | ✗ | asym | ✓ | [?] | [?] |
| symmetric-EMA | ✓ | gated | ✓ | sym | ✓ | [?] | [?] |
| no-boost | ✓ | gated | ✓ | asym | ✗ | ringdown ∞ | [?] |

Rows marked [?] are eval-only ablations (flag-off, no retraining) — feasible in ~1-2 days wall-clock on MJX.

### 5.4 Remaining Tables
- Standing idle comparison (ACC vs P-only vs HOMING)
- Contact-loss recovery results (drop + ledge scenarios)
- Terrain adaptation results (curb scenarios)
- Ringdown time-series figure for push recovery (Fig. 4)

---

## 6. References

**25-30 references** covering:
- Wheeled biped platforms: Ascento (Klemm 2019, 2020), DIABLO (Liu 2024), Whleaper (Zhu 2024), SKATER
- Two-wheeled inverted pendulum (TWIP) analytical foundations
- Cascade control theory and applications
- Contact detection / flight-phase control for legged robots
- Impact recovery and disturbance rejection for bipeds
- Terrain adaptation without exteroception
- WBC vs classical control comparisons for wheeled-legged systems

---

## 7. Implementation Plan

### Phase 1: ARS Plan (Socratic chapter planning) — CURRENT
### Phase 2: ARS Lit Review (annotated bibliography)
### Phase 3: Manual Technical Writing (Sec. III–VII)
### Phase 4: ARS Reviewer (simulated peer review)
### Phase 5: Manual Revision + Finalize

---

## 8. Fixes Applied (v3, 2026-07-27)

| ID | Issue | Fix |
|----|-------|-----|
| N1 | Overview equation τ_final = sum of 5 terms contradicts authority separation (terrain = height offset, not torque) | Rewrote as two-channel: τ_w (wheel) + τ_q (leg). Anchor = wheel channel (bias source = wheels → correction targets wheels) |
| N2 | g_prox = smoothstep(...) inverted — gives 0 at home, 1 far away (opposite of intended) | Fixed: g_prox = **1 −** smoothstep(|Δx|, 0.05, 0.15) |
| N3 | Pole numbers ±3.6/±4.8 don't match √(g/l) at 0.40/0.70 m | Fixed: ±3.74 (0.70m) → ±4.95 (0.40m), variation ≈32% |
| N4 | "Height-variable stance" column in Table I is factually wrong for Ascento/Whleaper | Renamed to "Gain-scheduled balance across height range" — what ACC uniquely does |
| N5 | Page budget (4.85 text) didn't account for 6 figures (~1.8 pages) → overrun | Adjusted: IV 2.0→1.7, V 1.0→0.8, others trimmed. 4.5 text + ~1.8 fig ≈ 6.3 total (acceptable with tight layout) |
| N6 | [?] in ablation for unusable config (kp=35 idle unstable) | Changed to "n/a (idle unstable)". [?] reserved for measurable cells. Ref count: ~20→~25-30 |
| — | Socratic intro question about contribution #3 | Resolved: "comprehensive validation" is not a contribution. 3 contributions are now: (i) proximity-gated anchor, (ii) unified gate-structured cascade, (iii) flight recovery + terrain adaptation without exteroception |

## 8b. Refinements from Socratic Q1–Q5

| Q | Refinement | Where applied |
|---|-----------|---------------|
| Q1 | **Tension narrative:** ¶2 uses kp=50-vs-35 tradeoff + I none-vs-global tradeoff (two axes of tension); ¶3 shows ACC breaks both via scheduled kp + gated I. Do NOT use scheduled kp in ¶2 — scheduled is the resolution, not evidence of tension. | §3.8, Sec I draft |
| Q2 | **SKATER kept.** Existing sub-D terrain critique changed from "exteroceptive" (unverifiable blanket claim) to "mechanism-based": existing = whole-body MPC/model-based replan; ACC = per-leg contact-point split, no terrain model. Table I terrain column renamed to "Proprioceptive per-leg ground adaptation (no terrain model)." | Sec II-D, Table I |
| Q3 | **Pole-only in Sec III.** Add one sentence: "The pole-migration argument is purely motivational: it explains why a single fixed gain fails, not how ACC's gains are obtained. ACC does not solve an LQR; its scheduled gains are PD terms tuned per height band. A standard LQR appears only as a baseline in Sec. V." **Supplementary:** prepare 1-page K_LQR at 3 heights + Q,R for baseline LQR — reviewer may request. | Sec III, Supplementary |
| Q4 | **Fig. 2 = two-channel gated block diagram** (not signal-summing diagram). Each gate drawn as multiplier (×) receiving physical condition inputs via dashed lines. Two separate channels (wheel-torque, leg-joint) terminate at robot actuators. Equation in IV-A must match figure. Anchor confirmed = wheel channel. | Fig. 2, IV-A |
| Q5 | **Keep all [?] rows.** Eval-only (flag-off, no retrain) — feasible in 1-2 days MJX. Priority order if time-constrained: global-I + symmetric-EMA first (hold 2/3 novelty) → no-boost → kp=35 fill. Added ablation truth-table (§5.3) mapping each config to a unique on/off flag combination — eliminates label ambiguity. Fixed kp=35 → n/a (idle unstable; robot oscillates 55 mm, measuring F_min is meaningless). | §5.3 |

---

## 9. Self-Review (post-N1..N6 fixes)

- [x] Two-channel equation matches authority separation — no contradiction
- [x] g_prox correctly = 1 near home (matching g_env convention)
- [x] Pole numbers match √(g/l) at stated heights
- [x] Table I columns reflect what ACC uniquely does, not what others also do
- [x] Page budget includes figures; IV and V compressed to fit 6 pages
- [x] n/a for genuinely unusable configs; [?] only for measurable-but-unmeasured
- [x] Push numbers: single definition (F_min, F_med), single table (Sec. V-2)
- [x] Flight: role separation (stress test vs use case) explicit
- [x] No duplicate tables: novelty in C is inline, ablation only in V
- [x] Gate language: continuous smoothstep, not binary
- [x] Units on all metrics
- [x] P vs I separation rationale in IV-B + IV-C with channel clarity
- [x] 2.5 Hz notch correctly attributed to closed-loop P-induced oscillation
- [x] Anchor output confirmed as wheel-torque channel from code audit
