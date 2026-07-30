# Review Round 13 — Full 7-Agent Academic Panel Review

**Date**: 2026-07-29
**Paper**: "Anchored Conditionally-Gated Control for Wheeled Bipedal Balance and Disturbance Recovery"
**Author**: Van Thuong Nguyen
**Review Mode**: `academic-paper-reviewer` v1.10.0 — full mode (7 agents)
**Target Venues**: IEEE RA-L, ICRA/IROS, IEEE/ASME TMECH, IEEE T-RO, Robotica

> **Neutrality constraint**: Old review files (ROUND_7–ROUND_12, PEER_REVIEW_REPORT, REVIEW_PANEL, EDITORIAL_DECISION) were NOT read. This review is independent.

---

## Panel Composition

| # | Role | Identity | Recommendation | Score |
|---|------|----------|---------------|-------|
| 1 | **EIC** | Senior Editor, IEEE RA-L — locomotion/balancing control | Major Revision | 63.8 |
| 2 | **R1 (Methodology)** | Control systems / experimental validation | Major Revision | 73.2 |
| 3 | **R2 (Domain)** | Wheeled biped robotics, 10+ yr | Minor Revision | ~72 |
| 4 | **R3 (Perspective)** | Process control / industrial automation | Minor Revision | 73.1 |
| 5 | **DA** | Adversarial stress-test | (5 CRITICAL, 5 MAJOR) | — |
| 6 | **Synthesizer** | Editorial synthesis | **Major Revision** | **~70.6** |

---

## Decision: MAJOR REVISION

**Panel average**: ~70.6/100 → rubric maps to Minor Revision (65–79), but upgraded to Major Revision due to specific experiments required, not text-only fixes.

---

## Top 3 Blocking Issues

### 1. Control Rate Confound (CRITICAL)
ACC runs at 100 Hz; all LQR/PPO baselines at 50 Hz. The 2× bandwidth advantage is acknowledged but never isolated. No ACC-at-50Hz ablation exists. The paper's own robustness data shows sharp delay bifurcation (96N→47N at any non-zero delay), making the 10ms timestep difference a major confound.

**Required fix**: Run ACC at 50 Hz and report degradation across all primary metrics.

### 2. Incomplete Additive Factorial (CRITICAL)
L2→L3 transition bundles 4 mechanisms simultaneously (prox gate, asym EMA, boost, pitch gate). The deferred L2.5 cell (L2 + prox gate only) would isolate the gate's contribution.

**Required fix**: Measure L2.5 configuration.

### 3. Comparative Claims Exceed Evidence (MAJOR)
- "53× RMS improvement" compares against deliberately crippled P-only baseline (39mm RMS with known 1.3Nm bias)
- LQR baselines use simplified 4-state TWIP model on a 10-DOF platform — need "TWIP-based" qualifier
- PPO baseline: single seed, 5.4M steps — insufficient for any comparative claim
- Table V lacks control rate column

**Required fix**: Qualify all comparative claims; add control rate column; address PPO baseline.

---

## DA CRITICAL Adjudications

| ID | Finding | Verdict |
|----|---------|---------|
| C1 | LQR baselines use structurally mismatched 4-state TWIP model | **VALIDATED** → R3 |
| C2 | 2× bandwidth confound unresolved | **VALIDATED** → R1 |
| C3 | "53× RMS improvement" rhetorically inflated | **REJECTED** as CRITICAL (downgraded to MAJOR) — number is arithmetically correct; issue is rhetorical framing → R3(a) |
| C4 | L2.5 ablation deferred | **VALIDATED** → R2 |
| C5 | Simulation-only with hardware framing | **PARTIALLY VALIDATED** — paper already acknowledges this; text revision suffices |

---

## Required Revisions (Must Fix)

| # | Item | Effort |
|---|------|--------|
| **R1** | Run ACC at 50 Hz, report degradation | 1 day |
| **R2** | Measure L2.5 (L2 + proximity gate only) | 1 day |
| **R3** | Qualify all comparative claims (53×, TWIP-LQR, PPO, control rate column) | 1–2 days text; 1–2 weeks if re-training PPO |

## Suggested Revisions (Should Fix) — 12 items

Key ones:
- **S1**: Resolve Whleaper positioning (ACC is NOT a residual RL architecture)
- **S2**: Deepen competing platform architecture comparison (Ascento WBC, SKATER MPC, DIABLO cascaded LQR)
- **S3**: Add hybrid-automaton formulation (Liberzon 2003)
- **S4**: Explicit comparison to Åström conditional integration
- **S5**: Timing budget as table + deployment risk clarification
- **S6**: Parameter sensitivity ranking for 50+ gains
- **S7**: Expand Hyon et al. (2007) gated control discussion
- **S8**: Cross-domain applicability paragraph
- **S9–S12**: Minor fixes (ringdown stats, Welch code, settle-time sweep, terminology)

---

## Score Breakdown by Dimension

| Dimension | Weight | EIC | R1 | R2 | R3 | Mean |
|-----------|--------|-----|----|----|----|------|
| Originality | 20% | 72 | 78 | 78 | 70 | **74.5** |
| Methodology | 25% | 58 | 65 | 72 | 78 | **68.3** |
| Evidence | 25% | 52 | 68 | 68 | 65 | **63.3** |
| Coherence | 15% | 78 | 82 | 75 | 82 | **79.3** |
| Writing | 15% | 68 | 80 | 74 | 78 | **75.0** |
| **Weighted** | | **63.8** | **73.2** | **~72** | **73.1** | **~70.6** |

---

## Venue Readiness

| Venue | Current State | After Fixes |
|-------|--------------|-------------|
| **IEEE RA-L** | Not ready | Ready after Major Revision (R1+R2+R3) |
| **ICRA / IROS** | Close | Ready after Minor Revision (text-only) |
| **IEEE/ASME TMECH** | Not ready | Ready after Major Revision |
| **IEEE T-RO** | Not ready | Needs hardware validation |
| **Robotica** | Close | Ready after Minor Revision (text-only) |

---

## Consensus Strengths (All 4 Reviewers Agree)

1. Proximity-gated anchor with asymmetric envelope follower is genuinely novel
2. Factorial ablation (L0→L3 + S0→S5) is methodologically strong
3. Limitations disclosure is unusually honest (9 enumerated)
4. 100 Hz vs 50 Hz confound must be resolved
5. Hardware validation is absent; claims should be scoped accordingly

## Consensus Weaknesses (3+ Reviewers Agree)

1. LQR baselines cannot support "LQR fails" claim without TWIP qualifier
2. PPO baseline (5.4M steps, single seed) is insufficient
3. "53× RMS improvement" frame is rhetorically inflated vs P-only

---

## Key Reviewer Quotes

**EIC**: "The gate-activation mechanism with its fast-attack/slow-release envelope follower is the paper's most interesting technical contribution and could plausibly transfer to other platforms."

**R1**: "The comparative claims against LQR/PPO are weakened by the control rate confound and baseline insufficiency. A major revision addressing these three items would substantially strengthen the paper's methodological foundation."

**R2**: "The paper's strongest domain-level assets are: correctly identifying the stiffness-precision-windup trilemma as a real, under-discussed problem; providing detailed, reproducible simulation specification; offering transferable tuning heuristics; and honestly disclosing simulation-only status."

**R3**: "ACC is an honest, carefully ablated piece of engineering that solves a real problem. From a process control perspective, the paper does something increasingly rare — it explicitly acknowledges its debt to Shinskey and Åström."

**DA**: "The paper is, at root, a well-executed engineering study of a conditional integrator with envelope detection. As an engineering contribution it has merit. But the rhetorical framing...is not supported by the evidence."

---

## Path Forward

### Fast path (no new experiments, ~2 days text)
→ Narrow all comparative claims → Submit to **ICRA/IROS** or **Robotica**
→ Paper is already near acceptance standard for these venues with text-only revision

### Strong path (experiments + text, ~1 week)
→ Run R1 (50Hz ACC), R2 (L2.5), R3 (claim qualification) → Submit to **IEEE RA-L**
→ Highest impact venue achievable with current data + minimal new experiments

### Full path (experiments + text + hardware prep, ~2-3 months)
→ All required + suggested revisions → Add hardware validation (even partial) → Submit to **IEEE T-RO** or **TMECH**
→ Requires hardware access; not recommended as immediate next step

---

## Revision Checklist (Priority-Ordered)

### 🔴 Priority 1 — Must do before resubmission
- [ ] R1: ACC at 50 Hz evaluation
- [ ] R2: L2.5 ablation measurement
- [ ] R3a: Qualify "53×" as "vs. P-only baseline" everywhere
- [ ] R3b: Add "TWIP-based" qualifier to all LQR failure claims
- [ ] R3c: Add control rate column to Table V
- [ ] R3d: Move PPO from Table V to discussion OR train 3-seed PPO at ≥50M steps

### 🟡 Priority 2 — Strongly recommended
- [ ] S1: Resolve Whleaper positioning
- [ ] S2: Deepen platform architecture comparison
- [ ] S5: Timing budget as table
- [ ] S6: Parameter sensitivity ranking
- [ ] S4: Explicit comparison to Åström conditional integration

### 🟢 Priority 3 — Nice to have
- [ ] S3: Hybrid-automaton formulation
- [ ] S7: Expand Hyon et al. discussion
- [ ] S8: Cross-domain applicability paragraph
- [ ] S9–S12: Minor fixes (ringdown stats, Welch code, settle-time sweep, terminology)
- [ ] Abstract: shorten, front-load mechanism, carry simulation-only caveat
- [ ] Table I: remove ACC bolding where no comparator exists
- [ ] Table II: separate tunable from fixed parameters
- [ ] Standardize CI computation to t-distribution
- [ ] Bibliography key consistency
- [ ] Smoothstep citation (Perlin 2002)

---

## Reviewer Reports (Full)

### EIC Review Report
- **Recommendation**: Major Revision
- **Confidence**: 4
- **Score**: 63.8
- **Key findings**: Asymmetric baseline comparison (Critical), platform comparison table misleading (Major), no hardware validation for RA-L (Major), insufficient PPO baseline (Major), "LQR fails" conflates reduced-order failure with structural insufficiency (Major), bandwidth confound unresolved (Major), no WBC/MPC baseline (Minor-Major)
- **Main strength**: Genuinely novel gate mechanism, methodical factorial ablation, honest limitations section, clean flight/terrain extensions

### R1 Methodology Review Report
- **Recommendation**: Major Revision
- **Confidence**: 4
- **Score**: 73.15
- **Key findings**: Incomplete additive factorial L2→L3 bundle (Major), control rate confound unresolved (Major), insufficient PPO baseline (Major), Welch t-test not in executable code (Minor), CI uses z=1.96 not t-distribution (Minor), sample size adequacy for precision claim (Minor), 46-config sweep not full factorial (Minor)
- **Main strength**: Principled factorial ablation, transparent limitations, deterministic seed management, multiple uncertainty quantification methods, controller implementation validation

### R2 Domain Review Report
- **Recommendation**: Minor Revision
- **Confidence**: 4
- **Score**: ~72
- **Key findings**: Whleaper positioning mismatch (Major — ACC is NOT residual RL), platform comparison surface-level (Major), "no single classical controller" is partial straw man (Moderate), PPO baseline insufficient for RL claims (Moderate), Hyon et al. underexplored (Minor)
- **Main strength**: Solves recognized but under-documented domain problem, simulation spec detail exceeds field norms, transferable tuning heuristics, genuine LQR sweep negative result, honest limitations

### R3 Perspective Review Report
- **Recommendation**: Minor Revision
- **Confidence**: 4
- **Score**: 73.1
- **Key findings**: 0.5ms timing margin is deployment risk not clearance (Major), 50+ hand-tuned parameters without systematic methodology (Major), insufficient distinction from industrial override/selector (Moderate), hybrid/switched-systems connection unexploited (Moderate), asymmetric EMA lineage deeper than acknowledged (Minor), smoothstep C0 transition implications unexplored (Minor)
- **Main strength**: Honest intellectual lineage, three-condition gate synthesis exceeds standard conditional integration, factorial ablation exceeds process control norms, gate FMEA mapping, hardware transfer discussion detailed

### DA Devil's Advocate Report
- **5 CRITICAL findings**: LQR baseline structural mismatch, bandwidth confound, "53×" claim inflation, deferred L2.5 ablation, simulation-only with hardware framing
- **5 MAJOR findings**: PPO baseline inadequacy, gate parameter overfitting, statistical power for key claims, high-noise "improvement" is degradation, platform comparison table misleading
- **5 MINOR findings**: Terminology inflation ("anchor"), settle-time inconsistency, flight/terrain not truly independent, incomplete DOB/ADRC reference, EMA₀=0 startup bias
- **Strongest counter-argument**: The baselines are structurally misaligned with the platform; the evidence supports a much narrower claim than the paper makes; the precision is a simulation artifact, not a robust property
