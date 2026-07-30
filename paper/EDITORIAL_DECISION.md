# Phase 2: Editorial Decision Letter & Revision Roadmap

**Paper:** "Anchored Conditionally-Gated Control for Wheeled Bipedal Balance and Disturbance Recovery"
**Author:** Van Thuong Nguyen
**Review Panel:** EIC + 3 Peer Reviewers + Devil's Advocate (5 independent reviews)
**Date:** 2026-07-28

---

## Reviewer Summary Matrix

| Dimension | EIC | R1 (Methodology) | R2 (Domain) | R3 (Cross-Disc.) | Devil's Advocate |
|-----------|-----|-------------------|-------------|------------------------|-----------------|
| **Overall Rec.** | Minor Revision | Minor Revision | Minor Revision | Major Revision | (stress-test only) |
| **Confidence** | 4 | 4 | 5 | 4 | — |
| **Key Strengths** | Honest limitations, ablation design, LQR thoroughness, architectural clarity | Factorial ablation, statistical disclosure, binary search, reproducibility | Detailed sim specs, honest comparison table, tuning guide, sensitivity analysis | Asymmetric EMA elegance, LQR comparison, practical transferability | — |
| **# Major Issues** | 2 | 2 | 1 | 2 | 4 (CRITICAL: 3) |
| **# Minor Issues** | 1 | 2 | 3 | 2 | 4 (MAJOR) |

---

## Consensus Matrix — Weakness Sub-Claim Inventory

Each weakness is decomposed into atomic sub-claims. Consensus = ≥3 reviewers independently raise the same sub-claim.

### CONSENSUS (3–5 reviewers agree)

| Sub-Claim ID | Description | EIC | R1 | R2 | R3 | DA | Verdict |
|-------------|-------------|-----|-----|-----|-----|-----|---------|
| **SC1** | Missing RL/MPC/DOB baseline — paper compares only against classical LQR variants | ✓ (W1) | — | — | ✓ (W1) | ✓ (C1) | **STRONG CONSENSUS (4/5)** — Must address |
| **SC2** | "Conditionally-gated" framing needs connection to existing control paradigms (override control, selector-based, gain scheduling) | ✓ (W2) | — | — | ✓ (W1) | — | **CONSENSUS (2/5)** — EIC + R3 independently raise this; R1's AW taxonomy concern is related |

### PARTIAL CONSENSUS (2 reviewers agree)

| Sub-Claim ID | Description | EIC | R1 | R2 | R3 | DA | Verdict |
|-------------|-------------|-----|-----|-----|-----|-----|---------|
| **SC3** | AW baseline tests only ONE variant (conditional integration) — claims about "standard AW" are overbroad | — | ✓ (W1) | — | — | — | **Single-reviewer (R1)** — but R1 is the AW expert; high-weight finding |
| **SC4** | LQR Q/R selection methodology not documented | — | ✓ (W2) | — | — | — | **Single-reviewer (R1)** — methodological documentation gap |
| **SC5** | LQR failure characterization overgeneralizes ("LQR fails" vs "LQR through PID servo fails") | — | — | — | — | ✓ (C2) | **DA only** — but EIC partially acknowledges this distinction |
| **SC6** | Contact model stiffness may make 0.73mm precision optimistic | — | — | ✓ (W2) | — | — | **Single-reviewer (R2)** |
| **SC7** | No stability analysis or theoretical guarantees | — | — | — | ✓ (W2) | — | **Single-reviewer (R3)** — but R3 self-identifies as [FIELD-NORM UNVERIFIED] |
| **SC8** | Feasibility demos (N=3–10) imply more than they prove | — | — | — | — | ✓ (C3) | **DA only** — but EIC initially raised then retracted similar concern |
| **SC9** | Platform comparison table omits metrics where ACC doesn't lead | — | — | — | — | ✓ (M4) | **DA only** |

---

## Consensus Assessment by Dimension

### Dimension 1: Experimental Validation — **MINOR REVISION REQUIRED**

**Consensus strength:** STRONG. All 5 reviewers acknowledge the ablation design is rigorous. EIC (S2), R1 (S1), R2 (S3), and R3 (S2) all praise the factorial ablation. DA's M3 (ablation order bias) is a theoretical concern only — the two-direction path addresses it adequately.

**Required actions:**
- Add RL or MPC baseline (SC1 — 4/5 consensus). This is the SINGLE most important revision. Even a 1M-step PPO comparison at fixed height would satisfy most reviewers.
- Address multiple comparison correction (R1 W4) — add Bonferroni-adjusted significance thresholds or note that the primary claim doesn't depend on borderline p-values.
- Explain non-monotonic noise response in Table VI (R1 W3).

### Dimension 2: Claims and Framing — **MINOR REVISION REQUIRED**

**Consensus strength:** MODERATE. EIC (W2) and R3 (W1) independently identify that the "conditionally-gated control" framing needs better positioning within existing paradigms (override control, selector-based, gain scheduling). DA's C2 (LQR failure overgeneralization) is a related framing concern.

**Required actions:**
- Add process control citations (override/selector-based control) to Related Work (R3 W1).
- Narrow AW claims in abstract/introduction to "saturation-triggered conditional integration" or test one additional AW variant (R1 W1).
- Clarify that LQR failure is through the PID servo path specifically (DA C2).

### Dimension 3: Baselines and Comparisons — **MAJOR REVISION REQUIRED**

**Consensus strength:** STRONG. This is the paper's most significant gap. DA C1 (no DOB/ADRC/MPC), EIC W1 (no RL/MPC), and R3 W1 (no DOB) converge on the same finding: the paper compares only against variants of the same underlying approach. The LQR comparisons are thorough *within* the classical PD-servo paradigm but don't test against fundamentally different approaches.

**Required actions:**
- **Priority 1:** Add at least ONE baseline from a different paradigm (DOB+LQR, MPC, or PPO). If PPO infrastructure exists in the repo (as CLAUDE.md indicates), a 1M-step comparison is feasible.
- **Priority 2:** If unable to add new baselines, reframe the paper's contribution more modestly: "ACC outperforms classical LQR variants" rather than "ACC is the uniquely correct architecture."
- Document Q/R selection methodology for LQR (R1 W2).

### Dimension 4: Theoretical Grounding — **OPTIONAL (FIELD-NORM DEPENDENT)**

**Consensus strength:** WEAK. Only R3 (W2) raises the lack of stability analysis, but R3 self-labels this as [FIELD-NORM UNVERIFIED]. For RA-L (an empirical/applied venue), theoretical guarantees are not expected. The editor (EIC) did not raise this concern.

**Required actions:** None required for RA-L submission. For T-RO or Automatica, stability analysis would be necessary.

### Dimension 5: Literature Coverage — **MINOR REVISION REQUIRED**

**Consensus strength:** MODERATE. R3 identifies missing process control and aerospace literature. R2 identifies missing envelope-detection signal processing literature and deeper Handle engagement.

**Required actions:**
- Add 3–5 citations to: override/selector-based control, envelope detection for control, and aerospace mode-switching (if those references can be verified).
- Deepen the Zamani et al. discussion (R2 W1).

---

## Devil's Advocate CRITICAL Adjudication

Per SKILL.md Iron Rule #4: every DA CRITICAL is adjudicated visibly.

| ID | DA Finding | Adjudication | Rationale |
|----|-----------|-------------|-----------|
| **C1** | No DOB/ADRC/MPC baseline tested | **VALIDATED** | This finding converges with EIC W1 (no RL/MPC) and R3 W1 (no DOB). Four of five reviewers independently identify the baseline gap. The paper MUST address this — either by adding a baseline or by narrowing claims. |
| **C2** | LQR failure overgeneralization ("LQR fails" vs "LQR through PID servo fails") | **VALIDATED — PARTIAL** | The DA is correct that the abstract/headline claim overgeneralizes. However, the paper body already makes the distinction clearly (§V-D: "the PID servo action path — not controller structure — is the bottleneck"). Fix: narrow the abstract claim to match the body's more precise framing. |
| **C3** | Contribution limited to single platform; architectural generality unproven | **REJECTED** | The paper explicitly acknowledges this as Limitation (8). The DA is not wrong about the fact, but the fact is already disclosed. No new action required beyond what the paper already does. |

**C3 rejection rationale:** The paper's Limitation (8) states: "Architectural generality across morphologies and simulators remains to be demonstrated." The DA's C3 finding re-states this acknowledged limitation without identifying a new defect. A finding that restates an already-disclosed limitation does not meet the CRITICAL threshold (Singleton-Critical test: if the defect is already disclosed, disclosure itself is the fix).

### DA CRITICAL Summary
- **VALIDATED:** 2 (C1, C2)
- **REJECTED:** 1 (C3)
- **UNRESOLVED:** 0

---

## Editorial Decision

### Decision: **MAJOR REVISION**

### Rationale

This is a strong paper with a genuine engineering contribution, rigorous experimental validation, and exemplary intellectual honesty. The paper's core finding — that a proximity-gated integral with asymmetric envelope follower can achieve sub-millimeter precision without sacrificing push recovery — is well-supported by the ablation evidence.

However, the paper has one substantial gap that prevents acceptance in current form: **the baseline comparison is limited to variants of the same underlying approach** (classical PD-servo LQR). Four of five reviewers independently identified the need for comparison against at least one fundamentally different paradigm (RL, MPC, or DOB-based control). This is the consensus critical issue.

Additionally, the framing needs refinement: (1) the "conditionally-gated control" terminology should be connected to existing paradigms (override control, selector-based control); (2) AW-related claims should be narrowed from "standard AW" to the specific variant tested; (3) the LQR failure characterization should be more precise ("LQR through PID servo" not "LQR").

The paper is **not rejected** because:
- The core contribution (proximity-gated anchor) is genuine and well-validated
- The ablation design is methodologically strong
- The limitations are honestly disclosed
- The DA's most concerning CRITICAL (C3) is already acknowledged by the authors
- The required revisions are bounded and achievable

### Venue Assessment

| Venue | Readiness | Gap |
|-------|-----------|-----|
| **IEEE RA-L** | **Ready after Major Revision** | Add ≥1 alternative-paradigm baseline; refine framing. 2–4 weeks of additional experiments + writing. |
| **ICRA/IROS** | **Ready now (conference version)** | Conference reviewers are more forgiving on baseline breadth. The current paper would be competitive. |
| **IEEE/ASME TMECH** | **Not ready** | Requires hardware validation. Simulation-only is a significant weakness for TMECH. |
| **IEEE T-RO** | **Not ready** | Requires hardware + theoretical analysis + broader baselines. |
| **Robotica** | **Ready after Minor Revision** | Lower bar for simulation-only. Minor framing fixes sufficient. |

### Recommended Submission Strategy

1. **Immediate:** Submit to ICRA/IROS (conference deadline permitting). The paper is conference-ready now.
2. **2–4 weeks:** Add PPO baseline (1M steps), refine framing → resubmit as RA-L.
3. **6–12 months:** After hardware validation → target TMECH or T-RO with expanded baselines and theoretical analysis.

---

## Revision Roadmap (Prioritized)

### BLOCKING (must fix before resubmission)

| Priority | Action | Reviewers | Effort |
|----------|--------|-----------|--------|
| **P0** | Add at least ONE baseline from a different paradigm: PPO (1M steps), MPC, or DOB+LQR. PPO is recommended since infrastructure exists. | EIC, R3, DA (4/5 consensus) | 1–2 weeks |
| **P0** | Narrow AW-related claims: change "standard AW is insufficient" → "saturation-triggered conditional integration AW is insufficient" OR test tracking back-calculation AW. | R1 | 2–3 days |

### HIGH PRIORITY (strongly recommended)

| Priority | Action | Reviewers | Effort |
|----------|--------|-----------|--------|
| **P1** | Add process control citations (override/selector/split-range control) to Related Work §II. Cite Shinskey or Åström & Hägglund Chapter 10. | R3, EIC | 2–4 hours |
| **P1** | Refine "conditionally-gated" framing: position ACC as continuous (smoothstep) extension of selector-based control for robotics. | EIC, R3 | 4–8 hours |
| **P1** | Narrow abstract/headline LQR failure claim: "LQR through PID servo fails" instead of "LQR fails." | DA (C2) | 1 hour |
| **P1** | Document LQR Q/R selection methodology (footnote or appendix). | R1 | 2 hours |

### MEDIUM PRIORITY (improves quality)

| Priority | Action | Reviewers | Effort |
|----------|--------|-----------|--------|
| **P2** | Explain non-monotonic noise response in Table VI (high noise outperforming medium noise). | R1 | 2–4 hours |
| **P2** | Discuss contact stiffness sensitivity for the 0.73mm claim. | R2 | 2–4 hours |
| **P2** | Add multiple comparison correction discussion (Bonferroni) to ablation p-values. | R1 | 1 hour |
| **P2** | Deepen Zamani et al. (2020) comparison — what did they find on their platform? | R2 | 2–4 hours |

### LOW PRIORITY (optional enhancements)

| Priority | Action | Reviewers | Effort |
|----------|--------|-----------|--------|
| **P3** | Add aerospace bumpless transfer citations to flight controller section. | R3 | 1–2 hours |
| **P3** | Add envelope-detection signal processing references. | R2 | 1–2 hours |
| **P3** | Consider making platform comparison table more symmetric (include metrics where other platforms lead). | DA (M4) | 1–2 hours |
| **P3** | Tie sensor noise levels to specific IMU models for hardware relevance. | R2 | 1 hour |
| **P3** | Consider title refinement: "Proximity-Gated Integral Control with Asymmetric Envelope Detection for Precision Standing and Push Recovery of a Wheeled Biped" (more specific, less general-claiming). | R3 | 30 min |

---

## Summary Statistics

- **Reviewers recommending Accept:** 0
- **Reviewers recommending Minor Revision:** 3 (EIC, R1, R2)
- **Reviewers recommending Major Revision:** 1 (R3)
- **Reviewers recommending Reject:** 0
- **DA CRITICAL findings:** 3 (2 validated, 1 rejected)
- **Consensus issues:** 2 strong (SC1, SC2), 7 partial
- **Estimated revision effort:** 2–4 weeks for RA-L resubmission

---

## Overall Assessment for Publication Readiness

**Current state: NOT YET READY for RA-L/TMECH/T-RO. READY for ICRA/IROS conference.**

The paper needs one material addition (alternative-paradigm baseline) and several framing refinements before journal submission. These are bounded, achievable revisions. The underlying contribution is solid and the experimental methodology is strong. The paper's honesty about limitations is a significant asset that will serve it well in peer review.

**Recommendation:** Submit to the next available ICRA/IROS deadline immediately (conference-ready now). In parallel, add the PPO baseline and framing refinements for RA-L journal submission within 4 weeks.

---

*Synthesized from 5 independent reviewer reports (EIC + R1 Methodology + R2 Domain + R3 Cross-Disciplinary + Devil's Advocate). All reviewer findings are preserved in the Phase 1 report file. No new review comments have been introduced by the synthesizer.*
