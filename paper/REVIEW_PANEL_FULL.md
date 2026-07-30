# Editorial Decision Package — ACC Paper Review

**Review Date:** 2026-07-29
**Review Panel:** 5 independent reviewers (EIC + 3 Peer Reviewers + Devil's Advocate)
**Mode:** Full (academic-paper-reviewer v1.10.0)
**Prior Rounds:** Excluded per author request (neutrality)

---

## Decision: **MAJOR REVISION** — Score: 72.5/100

---

## Venue Readiness

| Venue | Readiness | Verdict |
|-------|-----------|---------|
| **IEEE RA-L** | **75%** | Submit after Major Revision |
| **ICRA / IROS** | **80%** | Ready after Minor Revision |
| **Robotica** | **78%** | Ready after Minor Revision |
| **IEEE/ASME TMECH** | **70%** | Submit after Major Revision + supplements |
| **IEEE T-RO** | **45%** | Not ready (needs hardware + formal analysis) |

---

## Scoring Summary (0-100 Rubric)

| Dimension | EIC | R1 | R2 | R3 | DA | Panel Mean |
|-----------|-----|----|----|----|----|------------|
| Originality (20%) | 72 | 75 | 78 | 65 | 70 | **72** |
| Methodological Rigor (25%) | 80 | 68 | 75 | 70 | 72 | **73** |
| Evidence Sufficiency (25%) | 75 | 72 | 70 | 68 | 65 | **70** |
| Argument Coherence (15%) | 82 | 78 | 75 | 72 | 60 | **73** |
| Writing Quality (15%) | 82 | 80 | 78 | 76 | 75 | **78** |
| **Weighted Average** | **77.5** | **73.1** | **74.7** | **69.4** | **67.7** | **72.5** |

**After revision (projected): 78-82/100 → Accept / Minor Revision**

---

## Reviewer Configuration

### EIC — Senior Editor, IEEE RA-L
- **Specialization:** Wheeled mobile manipulation and balancing robots
- **Recommendation:** Minor Revision
- **Confidence:** 4/5

### R1 (Methodology) — DLR Control Systems Validation Researcher
- **Specialization:** Experimental controller benchmarking, simulation fidelity, factorial ablation design
- **Recommendation:** Major Revision
- **Confidence:** 4/5

### R2 (Domain) — Wheeled Bipedal Robotics PI
- **Specialization:** Ascento/Handle-era LQR-WBC hybrid controllers, coupled pitch-roll dynamics
- **Recommendation:** Minor Revision
- **Confidence:** 5/5

### R3 (Perspective) — Senior Process Control Engineer (ExxonMobil/BASF → Robotics)
- **Specialization:** Industrial override/selector/split-range control, anti-windup taxonomy
- **Recommendation:** Minor Revision
- **Confidence:** 4/5

### DA (Devil's Advocate) — Critical Robotics Experimentalist
- **Specialization:** Simulation fidelity, statistical power, architectural generality stress-testing
- **Recommendation:** Major Revision
- **Confidence:** 4/5
- **Final Tally:** 2 CRITICAL + 5 MAJOR + 2 MINOR

---

## CONSENSUS-5 (All Reviewers Agree)

1. **Core mechanism (asymmetric envelope-gated anchor integral) is genuinely novel and well-validated**
2. **Factorial ablation methodology is exemplary — exceeds field standards**
3. **Intellectual honesty and limitation disclosure are exceptional — rare in the field**

## CONSENSUS-4 (4/5 Reviewers Agree)

4. **Scheduled k_p is superfluous — S1 (fixed k_p=50) outperforms S0 on all metrics; should be removed from architecture**
5. **"ACC as precondition for RL" has zero supporting evidence — must be qualified as hypothesis**
6. **No alternative-paradigm baseline (PPO, MPC, DOB) — comparison only against LQR variants**

---

## Top 3 Blocking Issues

### #1 [CRITICAL] Architectural Framing Inflated
- **Evidence:** S1 (fixed k_p=50, F_min=83.1N) outperforms S0 (scheduled k_p, F_min=80.4N). L0 P-only achieves F_med=121.8N (higher than ACC's 110.1N). Flight/terrain are independent modules with zero interaction with anchor.
- **Fix:** Remove scheduled k_p from architecture. Promote S1 as canonical ACC. Narrow framing from "unified architecture" to "PD balance + proximity-gated integral + independent extensions."
- **Sources:** DA(C1,C2), EIC(W3), R1(W1), R2
- **Effort:** 5-7 hours

### #2 [MAJOR] 0.42mm vs 0.73mm Idle Precision Discrepancy
- **Evidence:** Table III (standing test): 0.73±0.07mm (N=10, 5s settle). Table VI (robustness sweep): 0.42±0.03mm (N=5, 3s settle). Same controller, same sim params. 1.74× discrepancy unexplained.
- **Root cause (suspected):** EMA initialization transient — EMA_0=0 biases g_env=1 for ~7.5s. Different settle times (3s vs 5s) → different points on transient curve.
- **Fix:** Run settle-time sweep (1s-15s) with EMA_0=1.0 (hardware-realistic). Report precision at each settle time. Reconcile values.
- **Sources:** R1(W3), DA(M6)
- **Effort:** 4-6 hours

### #3 [MAJOR] Paired t-test Assumptions Invalid
- **Evidence:** Configurations use independent deterministic seeds (`20260728 * 100 + rep`). "Pairing" is by rep index, not shared initial conditions. Each rep has independent `rng.normal(0.0, 0.005)` perturbation.
- **Fix:** Switch to Welch's t-test (unpaired, unequal variance) OR implement true paired design with stored/replayed initial conditions.
- **Sources:** R1(W2)
- **Effort:** 3-4 hours

---

## Summary of Each Reviewer's Findings

### EIC — 7 Strengths, 7 Weaknesses
**Strengths:** Proximity-gated anchor novelty, factorial ablation design, 4-variant LQR comparison, intellectual honesty, reproducibility infrastructure, practical porting guidance, architectural clarity.
**Weaknesses:** No alternative-paradigm baseline (Major), conditional-gating framing over-promises (Major), scheduled k_p superfluous (Major), statistical power insufficient for secondary claims (Major), 50+ hand-tuned parameters without methodology (Minor), no formal stability analysis (Minor), platform comparison table asymmetrical (Minor).

### R1 (Methodology) — 5 Strengths, 7 Weaknesses
**Strengths:** Factorial ablation design, statistical transparency, 46-config LQR gain sweep, deterministic seeding disclosure, code/data versioning.
**Weaknesses:** L2→L3 confound bundles 4 mechanisms (High), invalid paired t-tests (High), 0.42mm vs 0.73mm discrepancy (Moderate), b_r_h=-5.0 not physically derived (Moderate), LQR N=60 vs N=20 discrepancy (Low-Moderate), noise injection doesn't test estimation dynamics (Low), within-run vs between-trial CV ambiguity (Low).

### R2 (Domain) — 5 Strengths, 3 Weaknesses
**Strengths:** Asymmetric envelope follower novelty, coupled 6-state LQR bottleneck identification, factorial ablation with additive+subtractive decomposition, gate activation figure pedagogy, 9 explicit limitations.
**Weaknesses:** Higher-rate PID servo not tested (Major), hybrid/switched systems literature absent (Major), platform comparison table qualitative only (Minor), Handle dynamics under-cited (Minor), gate_activation.pdf unverified (Minor).

### R3 (Perspective — Process Control Engineer) — 6 Weaknesses
**Weaknesses:** ACC not sufficiently distinguished from industrial override control (Major), asymmetric EMA lineage deeper than acknowledged (Minor), 0.5ms latency margin not credible for industrial deployment (Major), 50+ hand-tuned parameters is industrial maintainability concern (Major), stuck-gate failure modes not analyzed (Major), "ACC as precondition for RL" unevidenced (Major).

### DA (Devil's Advocate) — 2 CRITICAL, 5 MAJOR, 2 MINOR
**CRITICAL:**
- C1: Scheduled k_p contradicted by own data (S1 > S0)
- C2: "Unified architecture" claim inflated — components are independent, not unified

**MAJOR:**
- M1: "PID servo is THE bottleneck" overstatement — removing it doesn't fix the problem
- M2: "ACC is precondition for RL" has zero evidence
- M3: No alternative-paradigm baseline (PPO, MPC, DOB)
- M4: Ablations confirm design process, not architectural necessity
- M5: Only ONE symmetric EMA configuration tested — single-point ablation overgeneralized
- M6: 0.73mm conditional on idealized contact — headline claim misleading without qualification

**MINOR:**
- M7: Application need for sub-mm precision not established
- (M5 downgraded to MINOR after field-norm check)

---

## Full Revision Roadmap

### Priority 1 — Structural Revisions (Must Fix, 5-7 days)

| # | Item | Severity | Source | Effort |
|---|------|----------|--------|--------|
| R1 | Remove scheduled k_p from architecture, promote S1 (fixed k_p=50) as canonical ACC | Critical | DA(C1), EIC(W3), R1(W1) | 3-4h |
| R2 | Narrow architectural framing to match ablation data | Critical | DA(C2), EIC(W2), R2 | 2-3h |
| R3 | Characterize and reconcile 0.42mm vs 0.73mm idle precision discrepancy | Major | R1(W3), DA(M6) | 4-6h |
| R4 | Switch to Welch's t-test or implement true paired design | Major | R1(W2) | 3-4h |

### Priority 2 — Content Supplementation (Should Fix, 3-5 days)

| # | Item | Severity | Source | Effort |
|---|------|----------|--------|--------|
| S1 | Add hybrid/switched systems connection to Related Work | Major | R2(W2) | 1-2h |
| S2 | Qualify "ACC as precondition for RL" as hypothesis | Major | DA(M2), R3(W6), EIC | 0.5h |
| S3 | Distinguish ACC from standard industrial override control | Major | R3(W1) | 1-2h |
| S4 | Discuss higher-rate PID servo as simpler alternative | Major | R2(W1), DA(Alt Path 2) | 1h |
| S5 | Add gate failure mode analysis (stuck-at-1, stuck-at-0) | Major | R3(W5) | 1-2h |
| S6 | Restructure additive hierarchy — add L2.5 to isolate L3 confound | Major | R1(W1) | 4-6h |
| S7 | Reconcile LQR baseline N values (N=60 vs N=20) | Minor | R1(W5) | 2-3h |
| S8 | Derive or sensitivity-analyze b_r_h = -5.0 in coupled LQR model | Minor | R1(W4) | 3-4h |
| S9 | Add quantitative context to platform comparison table | Minor | R2(W3), EIC(W7) | 1h |
| S10 | Discuss latency margin credibility for industrial deployment | Major | R3(W3) | 1h |

### Priority 3 — Text and Formatting (Nice to Fix, 2-3 days)

- Fix "Within-run CV" → "Between-trial CV" in Table III header (R1 M1)
- Add statistical qualifier to abstract: distinguish N≥5 core results from N=3-10 qualitative (EIC M4)
- Qualify "Differences <5N cannot be distinguished from noise" with "at N=5" (R1 M2)
- Use "LQR-servo" terminology consistently (R1 M3)
- Cite peak-hold/envelope detector lineage (EIC M5, R3 W2)
- Add coordinate frame indicator + CoM marker to Fig. 1 (EIC M9)
- Verify `biped_flight_control` citation accuracy (R2 M3)
- Clarify Table I "---" footnote: "not reported in cited work" vs "not tested" (R2 M8, EIC M2)
- Add hardware latency timing budget table (EIC M8)
- Discuss Whleaper LQR+PPO hybrid connection (R2 M2)
- Remove/complete placeholder code in `gate_sensitivity.py` (R1 M4)
- Check robustness sweep high-noise data completeness (R1 M5)
- Verify Cui et al. (2019) DOB reference domain (R2 M4)
- Verify Liu et al. (2024) review journal name (R2 M5)
- Verify `gain_scheduling_survey` resolves to Rugh & Shamma (2000) (R2 M6)
- Consider "is hypothesized to provide" for sim-to-real claim (R2 M7)
- Report compound disturbance CI (R1 M6)
- Add F_max degradation to abstract for push performance under delay (R1 M7)

---

## DA Critical Adjudications

| ID | Finding | Adjudication | Rationale |
|----|---------|-------------|-----------|
| C1 | Scheduled k_p contradicted by own data | **VALIDATED** | Objective, verified: S1 F_min=83.1 > S0 F_min=80.4, identical precision. 3/4 non-DA reviewers independently flag this. |
| C2 | "Unified architecture" claim inflated | **VALIDATED** | Evidence-supported: L0 F_med=121.8 > L3 F_med=110.1 shows PD core handles push independently. Flight/terrain modules have zero interaction with anchor. 2/4 non-DA reviewers flag the framing concern. |

**DA-CRITICAL-VS-ACCEPT: 2 validated** — Blocks silent Accept. Requires architectural reframing (R1, R2) before re-review.

---

## Recommended Submission Strategy

1. **IEEE RA-L** — Best fit. Focused controller contribution fits letter format. Simulation-only acceptable with honest disclosure. Submit after addressing Top 3 blocking issues + S1-S5.
2. **ICRA 2027** — Fallback. Conference reviewers less likely to flag architectural framing nuance. Rigorous ablation will impress.
3. **Robotica** — Alternative. More space if paper grows during revision. Accepts well-executed simulation studies.

---

## Total Estimated Revision Effort

| Priority | Items | Effort |
|----------|-------|--------|
| P1 — Structural (R1-R4) | Remove scheduled k_p, narrow framing, characterize precision, fix t-tests | 5-7 days |
| P2 — Content (S1-S10) | Hybrid systems lit, RL qualification, override comparison, servo discussion, gate FMEA, L3 restructure, N reconciliation, b_r_h derivation | 3-5 days |
| P3 — Text/Formatting | ~16 minor fixes | 2-3 days |
| **Total** | | **10-15 days (2-3 weeks)** |

---

*Review conducted 2026-07-29 by 5-agent academic paper review panel (academic-paper-reviewer v1.10.0, full mode). All findings are evidence-anchored to paper/main.tex. Prior 7 review rounds excluded per author request to maintain neutrality.*
