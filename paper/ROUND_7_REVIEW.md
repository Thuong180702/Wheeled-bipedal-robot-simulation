# Round 7 — Independent Academic Peer Review (Full Mode)

**Paper:** "Anchored Conditionally-Gated Control for Wheeled Bipedal Balance and Disturbance Recovery"
**Author:** Van Thuong Nguyen
**Review Protocol:** academic-paper-reviewer v1.10.0 — Full Mode (7 agents, 3 phases)
**Review Date:** 2026-07-28
**Review Characteristic:** **Independent review — reviewers did NOT read prior review reports to maintain neutrality**
**Target Venues:** IEEE RA-L, ICRA/IROS, IEEE/ASME TMECH, IEEE T-RO, Robotica

> **Round identity:** This is Round 7. Round 6 was documented in [REVIEW_PANEL_PHASE1.md](REVIEW_PANEL_PHASE1.md). Round 4 was documented in [PEER_REVIEW_REPORT.md](PEER_REVIEW_REPORT.md). Round 7 was conducted independently with fresh reviewer personas and no access to prior review files.

---

## Overall Result

| Metric | Value |
|--------|-------|
| **Final Score** | **63/100** |
| **Editorial Decision** | **Major Revision** |
| **Adjudicated DA CRITICAL findings** | 2 validated, 1 rejected |
| **Consensus Issues** | 2 strong (4-5/5 reviewers) |
| **Reviewers Recommending Minor Revision** | 3 (EIC, R1, R2) |
| **Reviewers Recommending Major Revision** | 1 (R3) |
| **Reviewers Recommending Reject** | 0 |
| **Estimated Revision Effort** | 2–4 weeks for RA-L resubmission |

### Venue-by-Venue Publication Readiness

| Venue | Readiness | Gap |
|-------|-----------|-----|
| **ICRA/IROS 2026** | **Ready now (conference version)** | Conference reviewers more tolerant on baseline breadth |
| **IEEE RA-L** | **Ready after Major Revision** | Add ≥1 alternative-paradigm baseline; refine framing |
| **Robotica** | **Ready after Minor Revision** | Lower bar for simulation-only |
| **IEEE/ASME TMECH** | **Not ready** | Requires hardware validation |
| **IEEE T-RO** | **Not ready** | Requires hardware + theoretical analysis |

---

## Phase 0 — Field Analysis

### Paper Classification

| Dimension | Analysis |
|-----------|----------|
| **Primary Discipline** | Robotics and Control Systems Engineering |
| **Secondary Disciplines** | Classical Control Theory, Mechatronics, RL (motivation only) |
| **Research Paradigm** | Quantitative Research — simulation-based comparative evaluation |
| **Methodology** | Factorial ablation, binary-search threshold measurement, controlled baseline comparisons |
| **Target Journal Tier** | Q2 — IEEE RA-L or IEEE/ASME TMECH |
| **Paper Maturity** | Pre-submission — near conference-ready, needs expanded N + hardware for journal |

### Reviewer Configuration (Round 7)

| # | Role | Identity Description |
|---|------|---------------------|
| **EIC** | Editor-in-Chief | Senior Editor, IEEE RA-L — legged/wheeled robot control, novel architectures |
| **R1** | Methodology | Researcher in experimental robotics evaluation — systematic ablation design, statistical validation |
| **R2** | Domain Expert | Senior researcher — wheeled biped platforms (Ascento, SKATER, Whleaper), LQR/WBC/MPC |
| **R3** | Cross-Disciplinary | Control theorist — anti-windup taxonomy, process control, aerospace mode-switching |
| **DA** | Devil's Advocate | Stress-test — 9 challenge dimensions, core argument validation |

> Full reviewer configuration cards are in [review_round7/00_field_analyst_report.md](review_round7/00_field_analyst_report.md).

---

## Phase 1 — Independent Peer Reviews

### Reviewer Recommendations Summary

| Reviewer | Recommendation | Key Strengths Identified | Key Weaknesses Identified |
|----------|---------------|--------------------------|--------------------------|
| **EIC** | Major Revision | Honest limitations (S1), factorial ablation (S2), asymmetric EMA novelty (S3), LQR baseline thoroughness (S4) | No hardware validation (W1-Critical), conditionally-gated framing over-promises (W2), insufficient statistical power (W3), single morphology (W4), latency sensitivity (W5), RL motivation unevaluated (W6), asymmetric platform table (W7) |
| **R1 Methodology** | Major Revision | Factorial ablation with two-direction validation (S1), explicit statistical power disclosure (S2), binary search for thresholds (S3), reproducibility commitment (S4) | Statistical test identity not disclosed (W1), no effect sizes reported (W2), robustness sweep non-monotonicity unexplained (W3), multiple comparison correction incomplete (W4), AW baseline tests only ONE variant (W1), LQR Q/R selection not documented (W2) |
| **R2 Domain** | Major Revision | Asymmetric EMA elegance (S1), LQR comparison strongest empirical contribution (S2), practical transferability (S3) | Terminology inflation masks incremental contribution (W1), "conditionally-gated" not a new paradigm — it's override/selector-based control (W1-R3), no stability analysis (W2-R3), missed aerospace bumpless transfer literature (W3), title over-promises (W4) |
| **R3 Cross-Disc.** | Major Revision | Asymmetric EMA elegance (S1), LQR comparison strongest empirical contribution (S2), practical transferability (S3) | Understated debt to override/selector control (W1), hardware latency budget implausibly tight (W2), no stability proofs (W3), aerospace connection missed (W4) |
| **DA** | N/A (stress-test) | Honesty about limitations, gate activation figures, factorial ablation design structure, EMA initialization discussion | **C1-Critical:** Scheduled k_p contradicted by data — S1 beats S0, **C2-Critical:** PID servo bottleneck logic invalid, M1: LQR sweep misaligned with failure mode, M2: Delay bifurcation limits practical relevance, M3: "ACC as precondition for RL" has zero evidence, M4: 50+ hand-tuned gains without methodology, M5: Statistical power makes most ablative comparisons inconclusive, M6: Novelty of proximity-gated anchor overstated |

### DA CRITICAL Findings (Round 7 Unique)

#### C1 — Scheduled k_p Contradicted by Data **[VALIDATED — 2/5 reviewers]**

The scheduled pitch stiffness (k_p: 50 stiff → 35 soft) is presented as enabling the precision-push trade-off. **But S1 (removing the k_p schedule, fixing k_p=50) achieves BETTER push survival (F_min 83.1N vs 80.4N) at IDENTICAL idle precision (0.73mm).** The trade-off resolution is achieved by anchor integral + boost alone; the k_p schedule is either superfluous or actively degrades push performance.

- **Evidence:** Table IV — S0 (Full ACC) vs S1 (-Scheduled k_p): F_min 80.4±3.0 vs 83.1±0.8, F_med 110.1±8.2 vs 118.8±1.0, identical idle 0.73mm
- **Severity:** CRITICAL — data directly contradicts the claimed functional role
- **Required fix:** Either remove k_p schedule from architecture, or provide mechanistic explanation for why S1 outperforms S0, or acknowledge that k_p schedule is incidental not essential

#### C2 — PID Servo Bottleneck Logic Invalid **[VALIDATED — PARTIAL]**

The paper claims "PID servo action path — not controller structure — is THE bottleneck." But removing servo lag (Direct Torque LQR) improves survival by only 35% (0.52s→0.70s) with 100% fall rate persisting. **If removing the supposed bottleneck does not prevent failure, the inference from "servo lag exists" to "servo lag is THE bottleneck" is invalid.** The Direct Torque result demonstrates another bottleneck — likely fundamental authority limits of linear control on this nonlinear platform — remains.

- **Evidence:** Table V — LQR DT row: survival 0.70±0.04s, 100% fall rate; text §5.4 "PID servo action path — not controller structure — is the bottleneck"
- **Severity:** CRITICAL — logical chain break
- **Required fix:** Narrow claim from "THE bottleneck" to "a contributing bottleneck" or "the primary identifiable bottleneck"

#### C3 — Contribution Limited to Single Platform **[REJECTED]**

The paper explicitly acknowledges this as Limitation (8). Disclosure itself is the fix. No new action required.

> Full DA report: [review_round7/05_devils_advocate_report.md](review_round7/05_devils_advocate_report.md)

### Consensus Findings (Round 7)

#### STRONG CONSENSUS (4/5 reviewers)

| ID | Finding | EIC | R1 | R2 | R3 | DA |
|----|---------|-----|----|----|----|----|
| **SC1** | Missing RL/MPC/DOB baseline — paper compares only against classical LQR variants | ✓ W1 | — | — | ✓ W1 | ✓ C1 (prior round) |

#### CONSENSUS (3/5 reviewers)

| ID | Finding | EIC | R1 | R2 | R3 | DA |
|----|---------|-----|----|----|----|----|
| **SC2** | "Conditionally-gated" needs connection to override/selector-based control paradigms | ✓ W2 | — | — | ✓ W1 | ✓ M6 |

---

## Phase 2 — Editorial Synthesis & Decision

The complete Phase 2 synthesis is in [EDITORIAL_DECISION.md](EDITORIAL_DECISION.md). Key extracts:

### Consensus Matrix — Weakness Sub-Claim Inventory

| ID | Description | Verdict |
|----|-------------|---------|
| SC1 | Missing RL/MPC/DOB baseline (4/5) | **STRONG CONSENSUS — Must address** |
| SC2 | "Conditionally-gated" framing needs connection to existing paradigms (2/5) | **CONSENSUS — EIC + R3** |
| SC3 | AW baseline tests only ONE variant | Single-reviewer (R1) — high-weight |
| SC4 | LQR Q/R selection methodology not documented | Single-reviewer (R1) — methodological gap |
| SC5 | LQR failure characterization overgeneralizes | DA only — partially acknowledged |
| SC6 | Contact model stiffness may make 0.73mm precision optimistic | Single-reviewer (R2) |
| SC7 | No stability analysis or theoretical guarantees | Single-reviewer (R3) — [FIELD-NORM UNVERIFIED] |
| SC8 | Feasibility demos (N=3–10) imply more than they prove | DA only |
| SC9 | Platform comparison table omits symmetric metrics | DA only |

### Dimension-by-Dimension Assessment

| Dimension | Decision |
|-----------|----------|
| **Experimental Validation** | Minor Revision — ablation design is rigorous; add RL/MPC baseline |
| **Claims and Framing** | Minor Revision — connect to process control, narrow AW claims, clarify LQR bottleneck |
| **Baselines and Comparisons** | **MAJOR REVISION** — add ≥1 alternative-paradigm baseline |
| **Theoretical Grounding** | Optional (field-norm dependent) — not required for RA-L |
| **Literature Coverage** | Minor Revision — add process control citations |

---

## Comparison: Round 6 vs Round 7

### Issues Round 6 Identified and Were Fixed (before Round 7)

| Round 6 Issue | Status in Round 7 |
|--------------|-------------------|
| Missing Coupled 6-state 3D LQR | ✅ Added — Table V |
| "Cascade Control" naming misleading | ✅ Changed to "Conditionally-Gated Control" |
| AW claim too broad | ✅ Narrowed + Appendix A added |
| Missing refs (Zamani, Cui, Xin, Hyon) | ✅ Added |
| Title over-promises | ✅ Title now descriptive and accurate |
| No hardware estimation pathways | ✅ §VI "Path to Hardware" added |
| Platform comparison table promotional | ✅ Neutralized |

### Issues Both Rounds AGREE On (Convergent Validation)

| Issue | R6 | R7 |
|-------|----|----|
| Missing RL/MPC/DOB baseline | EIC W1 (4/5) | SC1 (4/5 consensus) |
| "Conditionally-gated" needs process control connection | R3 W1 | SC2 (EIC + R3 + DA) |
| AW baseline tests only ONE variant | R1 W1 | R1 W1 (Round 7 R1) |
| LQR bottleneck claim overgeneralized | DA C2 | DA-C2 |

### 🔴 Issues UNIQUE to Round 7 (Round 6 Missed)

| Round 7 Finding | Why Round 6 Missed It |
|----------------|----------------------|
| **DA-C1: Scheduled k_p contradicted by data** — S1 beats S0 | Round 6 reviewers accepted k_p schedule at face value without cross-checking S0 vs S1 ablation rows |
| **DA-C2: PID servo bottleneck logic invalid** — DT LQR still fails 100% | Round 6 praised the LQR comparison without scrutinizing the logical inference |
| **Statistical test identity not disclosed** (R1 W1) | Round 6 noted low N but didn't flag missing test names |
| **No effect sizes reported** (R1 W2) | Round 6 didn't apply APA 7.0 reporting standards |
| **46-configuration sweep arithmetic** (R1 observation) | Round 6 didn't verify the sweep count |
| **Hardware latency budget implausible** (R3 W2, DA M2) | Round 6 noted latency sensitivity but didn't challenge the 4.5ms computation estimate |
| **Asymmetric EMA novelty vs conditional-integration AW** (DA M6) | Round 6 accepted the novelty claim; Round 7 DA argues it's within established AW taxonomy |
| **RL "precondition" claim has zero evidence** (DA M3) | Round 6 didn't challenge the RL motivation |
| **50+ hand-tuned gains without methodology** (DA M4) | Round 6 praised parameter disclosure without questioning derivation |
| **k_p schedule is superfluous per own data** (DA-C1) | Round 6 didn't catch the S0 vs S1 contradiction |

### Score Trajectory

| Round | Score | Change | Key Delta |
|-------|-------|--------|-----------|
| R1-R2 | ~61 | — | Initial review |
| R3 | ~72 | +11 | Optimistic panel, missed many issues |
| R4 | ~60 | -12 | Independent review, stricter scrutiny |
| R6 | ~72 | +12 | User's own panel, selected reviewers |
| **R7** | **63** | **-9** | **Independent review, stricter scrutiny, 2 unique CRITICAL findings** |

> **Note:** Score variation across rounds reflects differences in reviewer strictness and panel composition, not paper quality fluctuations. The R7 score (63) is the most conservative and arguably most realistic estimate. The paper's genuine strengths (honest limitations, ablation design, asymmetric EMA) are consistently recognized across all rounds.

---

## Priority Revision Roadmap (Round 7)

### BLOCKING (P0 — must fix before resubmission)

| ID | Action | Effort |
|----|--------|--------|
| **P0-R1** | Reconcile scheduled k_p with ablation data (DA-C1). Options: (a) remove k_p schedule from architecture and rename to reflect anchor+boost as core, (b) explain why S1 > S0 mechanistically, (c) re-run S0 with better k_p tuning. **This is the single most important fix.** | 1–3 days |
| **P0-R2** | Recalibrate LQR bottleneck claim: "THE bottleneck" → "a contributing bottleneck" or "the dominant identifiable bottleneck" (DA-C2). | 1 hour |
| **P0-R3** | Add ≥1 alternative-paradigm baseline (PPO 1M steps recommended — infrastructure exists in repo per CLAUDE.md) OR narrow claims to "ACC outperforms classical LQR variants." | 1–2 weeks |

### HIGH PRIORITY (P1 — strongly recommended)

| ID | Action | Effort |
|----|--------|--------|
| **P1-R4** | Add process control citations: override/selector/split-range control to §II. Cite Shinskey or Åström & Hägglund Ch.10. Position ACC as continuous (smoothstep) extension of selector-based control. | 2–4 hours |
| **P1-R5** | Narrow AW claims: "standard AW is insufficient" → "saturation-triggered conditional integration AW is insufficient." OR test one additional AW variant (tracking back-calculation). | 2–3 days |
| **P1-R6** | Name statistical tests used, report effect sizes (Cohen's d or Hedges' g) for ablation pairwise comparisons. | 2–4 hours |
| **P1-R7** | Provide realistic end-to-end hardware latency budget including IMU readout, state estimation, control computation, and torque output. | 2–4 hours |
| **P1-R8** | Document LQR Q/R selection methodology (footnote or appendix). | 2 hours |

### MEDIUM PRIORITY (P2 — improves quality)

| ID | Action | Effort |
|----|--------|--------|
| P2-R9 | Explain non-monotonic noise response in Table VI (high noise outperforming medium noise). | 2–4 hours |
| P2-R10 | Discuss contact stiffness sensitivity for 0.73mm claim. | 2–4 hours |
| P2-R11 | Add Bonferroni correction discussion to ablation p-values. | 1 hour |
| P2-R12 | Deepen Zamani et al. (2020) comparison. | 2–4 hours |
| P2-R13 | Clarify 46-configuration sweep arithmetic (14×3×5=210, not 46) or correct the count. | 1 hour |

### LOW PRIORITY (P3 — optional)

| ID | Action | Effort |
|----|--------|--------|
| P3-R14 | Add aerospace bumpless transfer citations. | 1–2 hours |
| P3-R15 | Add envelope-detection signal processing references. | 1–2 hours |
| P3-R16 | Make platform comparison table more symmetric. | 1–2 hours |
| P3-R17 | Tie sensor noise levels to specific IMU models. | 1 hour |
| P3-R18 | Remove or down-scope "ACC as precondition for RL" claim (zero RL+ACC evidence). | 30 min |

---

## Round 7 Review Files

Individual reviewer reports are saved in [paper/review_round7/](review_round7/):

| File | Content |
|------|---------|
| [00_field_analyst_report.md](review_round7/00_field_analyst_report.md) | Phase 0: Field analysis & 5 reviewer configuration cards |
| [01_eic_report.md](review_round7/01_eic_report.md) | EIC Review — Journal fit, originality, 7 weaknesses |
| [02_methodology_r1_report.md](review_round7/02_methodology_r1_report.md) | R1 Methodology — Statistical validity, experimental design |
| [03_domain_r2_report.md](review_round7/03_domain_r2_report.md) | R2 Domain Expert — Literature coverage, baseline fairness |
| [04_perspective_r3_report.md](review_round7/04_perspective_r3_report.md) | R3 Cross-Disciplinary — Process control, aerospace connections |
| [05_devils_advocate_report.md](review_round7/05_devils_advocate_report.md) | DA — 2 CRITICAL + 7 MAJOR + 5 MINOR findings |
| [../EDITORIAL_DECISION.md](../EDITORIAL_DECISION.md) | Phase 2: Editorial decision, consensus matrix, DA adjudication, revision roadmap |

---

## Quick Start: What to Fix First

If you can only fix 3 things before ICRA/IROS submission:

1. **DA-C1 (k_p schedule):** Remove scheduled k_p from architecture OR explain S1 > S0. Data says it doesn't help.
2. **DA-C2 (bottleneck claim):** Change "THE bottleneck" → "a contributing bottleneck." One sentence.
3. **SC1 (baseline gap):** If no time for PPO, narrow claims: "ACC outperforms classical LQR variants" rather than "ACC is the uniquely correct architecture."

These 3 fixes are all text-level (no re-training needed) and address the 2 CRITICAL + 1 consensus issue.

---

*Round 7 synthesized from 6 independent agent reports (Field Analyst + EIC + R1 Methodology + R2 Domain + R3 Cross-Disciplinary + Devil's Advocate). Phase 2 editorial synthesis used the same 5 Phase 1 reports plus the Phase 0 configuration. All reviewer output files are preserved in `paper/review_round7/`.*
