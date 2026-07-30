# Editorial Decision Package

## Part 1: Editorial Decision Letter

---

### Decision: **MAJOR REVISION**

---

### Consensus Analysis

#### Reviewer Summary Matrix

| | EIC | R1 (Methodology) | R2 (Domain) | R3 (Perspective) | DA |
|---|---|---|---|---|---|
| **Recommendation** | Major Revision | Major Revision | Major Revision | Weak Accept (RA-L) / Major (T-RO) | (score 72.7) |
| **Confidence** | 4/5 | 5/5 | — | 4 | — |
| **Overall Score** | 78.85 | 68/100 | 4.05/10 | — | 72.7/100 |
| **Core contribution valid?** | YES | YES | YES | YES | YES |

---

#### STEP 1b: Sub-Claim Inventory

I decompose all weaknesses across the 4 non-DA reviewers into atomic sub-claims and map them.

| ID | Sub-Claim | EIC | R1 | R2 | R3 | Consensus |
|----|-----------|-----|----|----|----|-----------|
| **SC1** | RL-precondition framing is untested promissory note — must move to Future Work | W1 (Major) | Silent | 3.5/10 positioning | W4 (utility) | **CONSENSUS-3** (EIC+R2+R3) |
| **SC2** | Abstract 0.73mm claim lacks contact-model / noise qualification | W2 (Major) | Silent | "sub-mm" needs qual | W2 (sim-to-real) | **CONSENSUS-3** (EIC+R2+R3) |
| **SC3** | "Primary LQR bottleneck" overstatement vs. DT-LQR 35% improvement | W3 (Major) | M2 (Major) | 4.0/10 comparative | Silent | **CONSENSUS-3** (EIC+R1+R2) |
| **SC4** | Sim config description (Newton/100 vs implicitfast/4) misrepresents setup | Silent | C1 (CRITICAL) | Silent | Silent | **SINGLE** (R1, C=5/5) |
| **SC5** | Control frequency: paper claims 100Hz, code uses 50Hz | Silent | C2 (CRITICAL) | Silent | Silent | **SINGLE** (R1, C=5/5) |
| **SC6** | PPO 100M-step claim unsubstantiated | Silent | C3 (CRITICAL) | Weak baseline | 7/10 technical | **CORROBORATED** (R1+R2+R3 partial) |
| **SC7** | LQR baselines under-tuned / architecturally asymmetric / Q-R undisclosed | Silent | M2 (Major) | 4.0/10 comparative | Silent | **CORROBORATED** (R1+R2) |
| **SC8** | N=5 underpowers subtractive ablation conclusions | Silent | M3 (Major) | Silent | Silent | **CORROBORATED** (R1+DA M3) |
| **SC9** | Flight/terrain extensions overclaimed as contributions | W6 (scope) | Md4 (Moderate) | 5.0/10 contrib | Silent | **CONSENSUS-3** (EIC+R1+R2) |
| **SC10** | Whleaper (Zhu 2024) — closest prior work — not quantitatively compared | Silent | Silent | 3.0/10 lit coverage | Silent | **SINGLE** (R2, C=high) |
| **SC11** | Table I asymmetrically reports metrics | W4 (Minor) | Silent | 4.0/10 positioning | Silent | **CORROBORATED** |
| **SC12** | Latency sensitivity (<10ms) under-discussed given 0.5ms margin | W5 (Minor) | M1 (Major) | Silent | W1 (HW readiness) | **CONSENSUS-3** (EIC+R1+R3) |
| **SC13** | "Omnidirectional" → "Multidirectional" (30% direction asymmetry) | Silent | Silent | "Multidirectional" | Silent | **CORROBORATED** (R2+DA M4) |
| **SC14** | No comparison to DCM/capture-point push-recovery literature | Silent | Silent | 3.0/10 lit gap | Silent | **SINGLE** (R2) |
| **SC15** | No PI+dead-zone anti-windup baseline tested | Silent | Silent | 3.0/10 lit gap | Silent | **CORROBORATED** (R2+DA M2) |
| **SC16** | Asymmetric EMA not contextualized in audio/signal-processing lineage | Silent | Silent | 4.0/10 framework | W3 (cross-disc) | **CORROBORATED** (R2+R3) |
| **SC17** | Factorial ablation claimed as Contribution #3 (inflated) | Silent | Silent | 5.0/10 contrib | Silent | **SINGLE** (R2) |
| **SC18** | Hybrid systems (Liberzon) citation is decorative, not analytical | Silent | Silent | 4.0/10 framework | Silent | **SINGLE** (R2) |
| **SC19** | EMA₀=0 settle-time not fully characterized (3 different idle numbers) | Silent | Md2 (Moderate) | "sub-mm" qual | Silent | **CORROBORATED** (R1+R2+DA C3) |
| **SC20** | 53× improvement conflates limit-cycle elimination with precision | Silent | Md3 (Moderate) | Silent | Silent | **SINGLE** (R1) |
| **SC21** | DA C1: ACC F_med is WORSE than P-only — push framing is misleading | — | — | — | — | **DA CRITICAL** |
| **SC22** | DA C2: PID servo ~0.25s lag is never justified/measured | — | — | — | — | **DA CRITICAL** |
| **SC23** | DA C3: Three different idle precision numbers with no canonical value | — | — | — | — | **DA CRITICAL** |

---

#### Points of Agreement (CONSENSUS-3 or CONSENSUS-4)

The following issues have strong multi-reviewer consensus and **MUST** be addressed:

**CONSENSUS-3 issues (3+ reviewers agree, action-bearing):**

1. **RL-precondition framing (SC1).** EIC W1, R2 positioning 3.5/10, R3 W4 all agree: the claim that ACC is a "precondition for RL" has zero evidence. Move to Future Work. Frame ACC as a standalone classical controller.

2. **Abstract 0.73mm qualification (SC2).** EIC W2, R2 claims-check 4.5/10, R3 W2 all agree: the headline precision claim must carry contact-model, sensor-noise, and settle-time qualifiers in the abstract.

3. **LQR bottleneck overstatement (SC3).** EIC W3, R1 M2, R2 comparative 4.0/10 all agree: the claim that the PID servo path is the "primary" LQR bottleneck is in tension with Direct-Torque LQR data (only 35% improvement, 100% fall persists). Language must be softened.

4. **Flight/terrain scope inflation (SC9).** EIC W6, R1 Md4, R2 contribution 5.0/10 all agree: flight and terrain are trivial bolt-ons presented as architectural contributions. Demote or move to supplementary.

5. **Latency sensitivity under-discussed (SC12).** EIC W5, R1 M1, R3 W1 all agree: the 10ms bifurcation with 0.5ms projected margin is the paper's most practically consequential finding and must be elevated in prominence and discussed with realistic hardware jitter.

**CORROBORATED issues (2 reviewers, actionable):**

6. **PPO 100M-step substantiation (SC6).** R1 C3 + R2 + R3 partial. The checkpoint evidence (max 5.39M steps) contradicts the 100M claim. Must provide training log or correct the number.

7. **LQR baseline fairness (SC7).** R1 M2 + R2. Q/R matrices undisclosed. Coupled LQR not re-derived for DT plant. PID servo tuning unvalidated. The comparison needs transparency or bounding.

8. **N=5 underpowered for subtractive ablation (SC8).** R1 M3 + DA M3. Fine-grained claims about S2-S4 push contributions are below minimum detectable effect.

9. **"Omnidirectional" → "Multidirectional" (SC13).** R2 + DA M4. 30% direction asymmetry (F_min/F_med = 0.70) contradicts "omnidirectional."

10. **No PI+dead-zone baseline (SC15).** R2 + DA M2. The simplest engineering alternative to ACC is untested.

11. **Asymmetric EMA lineage (SC16).** R2 + R3. The envelope follower's audio-engineering and collision-detection precedents are uncited.

12. **EMA settle-time characterization (SC19).** R1 Md2 + DA C3. Three different idle precision numbers (0.42mm/0.73mm/unmeasured for EMA₀=1.0) across the paper create confusion.

#### Points of Disagreement (SPLIT — arbitration required)

**SPLIT-1: Simulator configuration description accuracy (SC4)**

- **R1 (C=5/5):** CRITICAL. Paper claims Newton/100 iterations/elliptic cone; XML shows implicitfast/4 iterations/pyramidal. "Significant research integrity concern."
- **EIC, R2, R3:** Silent — none flagged this as a concern.

**Arbitration (EIC):** This is a documentation error of moderate severity. The paper describes a solver configuration that does not match the XML. However, the substantive question is whether the actual solver (implicitfast, 4 iterations, pyramidal) produces meaningfully different results from what was described. R1 raises this as a validity concern but does not demonstrate that results would change. The fix is straightforward: correct the paper to match the XML. If the implicitfast/4-iteration configuration produces solver artifacts (as R1_Md1 separately notes), this should be documented as a limitation. The "integrity concern" framing is disproportionate for what appears to be a documentation mismatch rather than fabricated data. **Verdict: Must fix — correct configuration description to match XML. Add solver-iteration sensitivity note (R1_Md1). Not a research-integrity issue; a documentation error.**

**SPLIT-2: Control frequency discrepancy (SC5)**

- **R1 (C=5/5):** CRITICAL. Paper says 100Hz; base_env.py uses 50Hz. This cascades into latency claims, EMA parameters, hardware budget.
- **EIC, R2, R3:** Silent.

**Arbitration (EIC):** R1's code audit reveals a genuine discrepancy. The evaluation scripts (push_ci.py, robustness_sweep.py) run at 100Hz, but training environments run at 50Hz. The 10ms delay bifurcation threshold is exactly one eval step at 100Hz but half a training step at 50Hz. This is a factual inconsistency that must be resolved. The paper cannot claim 100Hz control throughout while the training infrastructure runs at 50Hz. **Verdict: Must fix — reconcile control rate globally. Either (a) change code to 100Hz and re-run all experiments, or (b) correct paper to 50Hz, re-derive all timing claims, and acknowledge the 20ms (not 10ms) delay threshold. Given the extensive re-running required by option (a), option (b) is more practical for a revision cycle if the 20ms threshold still supports the hardware argument. The authors must report the actual control rate used and verify all timing-dependent claims at that rate.**

**SPLIT-3: Whleaper as mandatory comparator (SC10)**

- **R2 (C=high):** Major gap. Whleaper is the single closest prior work. Must be quantitatively compared.
- **EIC, R1, R3:** Silent — none raised this as a requirement.

**Arbitration (EIC):** R2 is correct that Whleaper (Zhu 2024, LQR+PPO residual for wheeled biped) is the closest prior work. However, Whleaper's published results may not report metrics directly comparable to ACC's (idle precision, push force thresholds). R2 acknowledges this by suggesting approximate comparison from published figures. I agree the paper must engage Whleaper substantively, but a fully quantitative comparison may be impossible if Whleaper does not report compatible metrics. **Verdict: Should fix — add a dedicated comparison paragraph or subsection discussing Whleaper. Report what metrics are available from Whleaper's published results. Explicitly identify metric gaps that prevent direct numerical comparison. This is "should fix" (not "must fix") because metric incompatibility may make quantitative comparison infeasible, and the paper already acknowledges Whleaper. However, the characterization must move from one sentence to substantive engagement.**

**SPLIT-4: Factorial ablation as Contribution #3 (SC17)**

- **R2:** The ablation is good experimental practice, not a contribution. Inflates contribution count.
- **EIC S2:** "The factorial ablation design is exemplary... exceeds the norm in RA-L control papers."
- **R1:** "Factorial ablation design is methodologically strong" (strength #1).
- **R3 S2:** "Factorial ablation is the correct experimental design."

**Arbitration (EIC):** R2 is technically correct that an ablation is methodology, not a contribution per se. However, EIC, R1, and R3 all identify the ablation as a distinguishing strength. The tension is semantic: is "contribution" something novel contributed to the literature, or something valuable the paper provides? I side with R2: the ablation supports the contribution but is not itself the contribution. **Verdict: Should fix — restructure to two contributions: (1) the proximity-gated anchor mechanism with asymmetric envelope follower, and (2) the composable architecture demonstrated via extensions. The factorial ablation then serves as primary evidence for Contribution 1, presented as a methodological strength rather than Contribution 3. This is not "must fix" because the distinction is largely presentational, but it improves the paper's credibility.**

---

### Decision Rationale

This paper presents a genuinely novel mechanism -- a proximity-gated position integral with asymmetric envelope detection -- that addresses a real trade-off between idle precision and push recovery on a simulated 10-DOF wheeled biped. The factorial ablation (additive L0-L3 + subtractive S0-S5) is methodologically exemplary and exceeds norms in the field. The gate sensitivity analysis providing transferable tuning heuristics (tau_release sensitivity 220.4, ~20x the next parameter) makes the contribution actionable. The explicit acknowledgment of 10 limitations and statistical power constraints builds genuine reader trust. The S5 pitch-gate finding (removal induces 56mm parametric oscillation) is a genuinely insightful emergent-behavior result.

However, three issues prevent acceptance in current form and collectively justify Major Revision rather than Minor:

1. **Documentation integrity of simulation setup.** Two factual discrepancies exist between the paper and the codebase: (a) the solver description (Newton/100 iterations vs. implicitfast/4 iterations), and (b) the control frequency (100Hz claimed vs. 50Hz in training infrastructure). The control-frequency discrepancy cascades into the paper's central latency-sensitivity finding. These must be resolved -- either correct the paper to match the code or change the code and re-validate results. The EIC judges these as documentation errors, not research misconduct, but they must be corrected before publication.

2. **Untested claims that inflate significance.** The RL-precondition hypothesis (SC1) has zero supporting evidence and four separate reviewers across the current and prior rounds have flagged it. The "primary bottleneck" language for the PID servo path (SC3) is in tension with the Direct-Torque LQR evidence. The flight/terrain extensions (SC9) are trivial bolt-ons presented as architectural contributions. These collectively inflate the paper's apparent contribution beyond what the data support. Narrowing the claims to what is robustly demonstrated would substantially strengthen the paper.

3. **Comparative evaluation gaps.** The LQR baselines' Q/R matrices and tuning process are undisclosed (SC7). N=5 underpowers the subtractive ablation's fine-grained claims (SC8). The simplest engineering alternative (PI + position-dead-zone anti-windup) is untested (SC15). The closest prior work (Whleaper) receives only one sentence (SC10). These gaps do not invalidate ACC's core contribution but prevent readers from assessing its relative merit.

All identified issues are resolvable within one revision cycle. None requires new theoretical development. None invalidates ACC's empirical performance data. The revisions are primarily: (a) correcting documentation to match the codebase, (b) narrowing claims to what the data robustly support, and (c) adding transparency to the comparative evaluation. The paper's intellectual honesty and methodological rigor are its strongest assets and should be preserved throughout revision.

---

### Top Blocking Issues (ranked, with evidence anchors)

| Rank | Issue | Severity | Confidence | Evidence Anchor |
|------|-------|----------|------------|-----------------|
| **B1** | **Reconcile control frequency.** Paper claims 100Hz; training code uses 50Hz. Cascades into latency claims, EMA parameters, hardware budget. | CRITICAL | 5/5 (R1 code audit) | base_env.py:61 (CONTROL_DT=0.02); paper line 111 (Δ_ctrl=0.01s); robustness_sweep.py:30 (DT=0.01) |
| **B2** | **Correct solver configuration description.** Paper claims Newton/100 iterations/elliptic cone; XML shows implicitfast/4 iterations/pyramidal cone. | CRITICAL | 5/5 (R1 XML audit) | wheeled_biped_real.xml:38-41; paper Section III, line 111 |
| **B3** | **Move RL-precondition hypothesis to Future Work.** Zero evidence for "ACC as precondition for RL." 4 reviewers (EIC+R2+R3+DA) across 2 rounds have flagged. | MAJOR | 5/5 (consensus) | Paper line 66; CLAUDE.md confirms no residual RL trained |
| **B4** | **Qualify abstract 0.73mm claim with contact-model and noise caveats.** Degrades 60% under realistic tire compliance (1.2mm). Degrades to 2.30mm under 10ms delay. | MAJOR | 5/5 (consensus) | Abstract line 49; Section III lines 112-113; Table VI |
| **B5** | **Substantiate or correct PPO 100M-step claim.** Max checkpoint is 5.39M steps. | MAJOR | 4/5 (R1 code audit) | backup_checkpoints/Day2/step_5390336 |
| **B6** | **Disclose LQR Q/R matrices and tuning process.** Four LQR variants' gains are opaque. Coupled 6-state LQR not re-derived for DT plant. | MAJOR | 4/5 (R1+R2) | Table V; Limitation 6 |
| **B7** | **Soften "primary bottleneck" language for PID servo path.** DT-LQR shows only 35% improvement with persistent 100% fall rate. | MAJOR | 4/5 (EIC+R1) | Abstract line 50; Table V (DT-LQR: 0.76s survival) |

---

### Venue Readiness Assessment

| Venue | Readiness | Rationale |
|-------|-----------|-----------|
| **IEEE RA-L** | **Needs Major Revision (current target)** | Core contribution fits letter format. After B1-B7 fixes, the narrowed claims and corrected documentation would meet RA-L standards. The 6-page constraint requires moving flight/terrain to supplementary. |
| **ICRA/IROS** | **Needs Revision** | Conference format acceptable. Competitive landscape means the Whleaper comparison gap (SC10) and underpowered ablation (SC8) become more salient. Would benefit from N=10 on subtractive ablation. |
| **IEEE/ASME TMECH** | **Needs Substantial Revision + Hardware** | Longer format suits the detailed architecture. However, TMECH expects at minimum benchtop validation. The simulation-only nature and 0.5ms latency margin would face significant scrutiny. Hardware demonstration strongly recommended. |
| **IEEE T-RO** | **Not Ready** | Requires hardware validation OR formal stability analysis (Lyapunov, ROA estimation) OR both. The simulation-only, single-morphology validation is insufficient for T-RO. The decorative hybrid-systems citation would need to become analytical. |
| **Robotica** | **Needs Minor Revision** | Applied robotics focus. Simulation-only is acceptable if the path to hardware is convincing. The 5-step tuning procedure and gate sensitivity heuristics fit the applied emphasis. Primary concern is the unvalidated latency budget. |

**Recommendation:** Target RA-L after Major Revision. The letter format forces the necessary scope-narrowing (flight/terrain to supplementary, RL hypothesis to Future Work). The core anchor mechanism and factorial ablation are well-suited to the RA-L contribution model.

---

### Final Score: **73.8 / 100**

| Dimension | Weight | EIC | R1 | R2 | R3 | DA | **Adjudicated** |
|-----------|--------|-----|----|----|----|----|-----------------|
| Originality | 0.20 | 78 | — | — | 6/10 | 72 | **73** |
| Methodological Rigor | 0.25 | 82 | 70* | — | 8/10 | 78 | **76** |
| Evidence Sufficiency | 0.25 | 72 | — | — | — | 62 | **68** |
| Argument Coherence | 0.15 | 85 | — | — | — | 75 | **78** |
| Writing Quality | 0.15 | 80 | — | — | 7/10 | 80 | **78** |
| **Weighted** | | **78.85** | **68** | **4.05/10** | — | **72.7** | **73.8** |

*R1's "Experimental Design & Statistics" (70) and "Controller Validation & Claim Bounding" (85) mapped to Rigor.

The adjudicated score reflects penalties for: (a) the two CRITICAL documentation gaps that must be resolved before any claims can be fully trusted (control frequency, solver config), (b) the evidence-to-claims ratio gap (flight/terrain as contributions, RL precondition, ablation as contribution), and (c) underpowered fine-grained ablation conclusions. The score would rise to ~80 after B1-B7 fixes, consistent with the upper Minor Revision range.

---

## Part 2: Revision Roadmap

### Required Revisions (Must Fix — structural, blocking)

| # | Requirement | Traces To | Severity | Evidence | Action |
|---|------------|-----------|----------|----------|--------|
| **R1** | **Reconcile control frequency globally.** | SC5 (R1 C2), SC12 (consensus) | CRITICAL | base_env.py:61 CONTROL_DT=0.02; push_ci.py:27 DT=0.01 | Option A (recommended): Correct paper to state 50Hz training, 100Hz evaluation. Re-derive EMA α at 50Hz. Update latency threshold to 20ms (one step at 50Hz). Update hardware latency budget. Add note explaining eval runs at 100Hz for finer delay resolution. Option B: Change code to 100Hz, re-run all experiments. |
| **R2** | **Correct solver description to match XML.** | SC4 (R1 C1) | CRITICAL | XML: integrator="implicitfast", iterations="4", cone="pyramidal" | Replace paper's "Newton, 100 iterations; cone: pyramidal (elliptic for wheels)" with "implicitfast, 4 iterations, ls_iterations=8; cone: pyramidal." Correct friction description from Coulomb μs/μk to MuJoCo tangential/torsional/rolling. If solver-iteration effects are material, document as a limitation (R1_Md1). |
| **R3** | **Move RL-precondition hypothesis to Future Work.** | SC1 (CONSENSUS-3: EIC W1, R2, R3 W4) | MAJOR | Paper line 66; 4 reviewers across 2 rounds flagged | Delete "ACC is thus hypothesized as a precondition for RL" from Introduction (line 66). Replace with: "ACC achieves sub-millimeter precision and omnidirectional push recovery where four LQR variants and pure PPO fail." Add to Conclusion/Future Work: "ACC may serve as a structured prior for residual RL; this hypothesis awaits empirical validation." |
| **R4** | **Add contact-model qualifier to abstract precision claim.** | SC2 (CONSENSUS-3: EIC W2, R2, R3 W2) | MAJOR | Section III: solref softening → +60% RMS | Revise abstract to: "...sub-millimeter idle precision (0.73 +/- 0.07 mm CoM RMS, N=10, stiff ground contact; degrades to ~1.2 mm under tire-like compliance and to 2.30 mm under 10 ms delay)." |
| **R5** | **Substantiate or correct PPO training step count.** | SC6 (R1 C3, corroborated) | MAJOR | backup_checkpoints max = 5,390,336 | Either: (a) provide training log / wandb curve showing 100M steps, or (b) correct the paper to state the actual trained steps with an explanation. Document what "steps" means (env steps vs. updates × num_envs × steps_per_update). |
| **R6** | **Disclose LQR Q/R matrices and tuning process.** | SC7 (R1 M2, R2 comparative) | MAJOR | Table V; LQR code | Add appendix or footnote with Q/R matrices for all four LQR variants. Document tuning methodology (taken from cited papers? tuned on this platform? grid search?). State explicitly that coupled 6-state LQR uses CT-derived gains, not re-derived for DT plant (per Limitation 6). |
| **R7** | **Soften PID-servo bottleneck language.** | SC3 (CONSENSUS-3: EIC W3, R1 M2, R2) | MAJOR | Abstract line 50; DT-LQR 0.76s survival | Replace "a primary LQR bottleneck is the PID servo action path" with "The PID servo action path is a key identifiable constraint -- removing it via direct torque output improves survival 35% but does not prevent the fall, indicating additional factors also limit LQR performance." |

### Suggested Revisions (Should Fix — improve quality, not blocking)

| # | Requirement | Traces To | Action |
|---|------------|-----------|--------|
| **S1** | **Compare quantitatively with Whleaper (Zhu 2024).** | SC10 (R2) | Add a dedicated paragraph or subsection. Report available Whleaper metrics (push force, idle precision, if reported). If metrics are incompatible, explicitly state the gap. Move from one-sentence mention to substantive engagement. |
| **S2** | **Narrow "omnidirectional" to "multidirectional" or quantify asymmetry.** | SC13 (R2, DA M4) | Report F_min/F_med ratio (0.70) in abstract. Use "multidirectional (8 directions, F_min=83 N backward, F_med=119 N forward)." |
| **S3** | **Test PI + position-dead-zone anti-windup baseline, or explain its absence.** | SC15 (R2, DA M2) | Either: (a) implement and measure under identical push protocol (recommended -- ~10 lines of code), or (b) add a paragraph explaining why this baseline was omitted, citing specific AW schemes and their predicted failure modes for this problem. |
| **S4** | **Increase N for subtractive ablation, or narrow claims.** | SC8 (R1 M3, DA M3) | Either: (a) increase N to 10+ for S2-S5 rows, or (b) restructure narrative: present S2-S4 as exploratory trends, emphasize the two unambiguous findings (L0→L1 +5.5N, L2→L3 for precision, S5 collapse). |
| **S5** | **Demote flight/terrain from Contribution 2 to "Extensions."** | SC9 (CONSENSUS-3) | Move flight/terrain sections to a brief "Extensions" subsection or supplementary. Restructure contributions to: (1) proximity-gated anchor with asymmetric envelope follower, (2) composable architecture validated by push recovery and robustness. |
| **S6** | **Restructure ablation from Contribution #3 to supporting evidence.** | SC17 (R2, arbitrated) | Reframe: two contributions. The factorial ablation is primary evidence for Contribution 1, presented as methodological strength, not Contribution 3. |
| **S7** | **Elevate latency sensitivity finding in abstract and conclusion.** | SC12 (CONSENSUS-3) | Add to abstract: "ACC exhibits a critical latency sensitivity (<10 ms threshold; F_max bifurcates from 96 N to 47 N), placing projected hardware latency (~9.5 ms) within 5% of its degradation threshold." Add to conclusion: specific hardware validation priority. |
| **S8** | **Characterize EMA₀ settle-time with EMA₀=1.0 (hardware-recommended init).** | SC19 (R1 Md2, DA C3) | Run settle-time sweep at EMA₀=1.0 (1s, 3s, 5s, 10s, 15s). Report cold-start precision. Unify idle precision reporting around one canonical number. |
| **S9** | **Cite audio-engineering and collision-detection precedents for asymmetric envelope.** | SC16 (R2, R3 W3) | Add citations: Giannoulis et al. 2012 (digital compressor design), Zolzer 2011 (DAFX), Haddadin et al. 2008/2017 (momentum observers with asymmetric filtering for collision detection). |
| **S10** | **Reframe 53× claim as limit-cycle elimination + precision.** | SC20 (R1 Md3) | Replace "~53× idle precision improvement" with "ACC eliminates the 55 mm peak-to-peak limit cycle observed in P-only control and achieves 0.73 mm CoM RMS idle precision (a ~53× reduction in RMS error)." |
| **S11** | **Address DA-C1: acknowledge that ACC's F_med is slightly worse than P-only.** | DA C1 | Add sentence: "ACC's median push survival (F_med=118.8 +/- 1.0 N) is comparable to the P-only baseline (121.8 +/- 2.5 N); the anchor is primarily a precision mechanism that preserves, rather than enhances, push recovery performance." |
| **S12** | **Provide measured PID step response or analytical settling time.** | DA C2 | Measure or analytically derive the ~0.25s PID servo lag from the disclosed gains. Add to appendix or footnote. If the 0.25s figure is an estimate, state it as such. |

### Text and Formatting (Nice to Fix)

| # | Item | Traces To |
|---|------|-----------|
| **T1** | Define "ringdown" at first use in main text body. | EIC Minor 1 |
| **T2** | Add units to Figure 3 y-axis labels for standalone readability. | EIC Minor 2 |
| **T3** | Specify which base gains the dimensionless multipliers in Table II multiply. | EIC Minor 3 |
| **T4** | Specify finite-difference perturbation magnitudes for gate sensitivity gradients. | EIC Minor 4 |
| **T5** | Ensure consistent N reporting for LQR baselines (abstract vs. Table V). | EIC Minor 5 |
| **T6** | Verify all bibliography entries are cited in text (orphan check). | EIC Minor 6 |
| **T7** | Verify IEEEtran compliance for compact spacing overrides. | EIC Minor 7 |
| **T8** | Fix reference [7] journal name to official IEEE format. | R1 m2 |
| **T9** | Add world-frame convention statement to Section III (robot faces -Y, X left, Z up, pitch sign convention). | R1 Md5 |
| **T10** | Add "--- indicates not reported in cited work; absence does not imply incapability" caveat to Table I caption. | EIC W4, R2 |
| **T11** | Clarify non-monotonic noise behavior explanation beyond "EMA disengagement." | R1 m3 |
| **T12** | Clarify whether between-trial CV 10.0% is of the CoM RMS metric or of an underlying per-trial statistic. | R1 m4 |
| **T13** | Report default armature 0.01 in Table II (currently only reports overridden values). | R1 m1 |
| **T14** | Remove or develop the hybrid systems (Liberzon 2003) connection. If no formal analysis, remove the citation and framing. | R2 (framework 4.0/10) |
| **T15** | Add one sentence to Conclusion closing the loop on the RL-precondition framing (now in Future Work). | EIC |
| **T16** | Add "on a 10-DOF Wheeled Biped" scope qualifier to title or abstract. | EIC W6 |

---

### Revision Checklist

The authors should confirm each item in their response letter.

**Phase 1: Documentation Corrections (Must Complete)**
- [ ] R1: Control frequency reconciled globally (paper + code)
- [ ] R2: Solver configuration description matches XML
- [ ] R5: PPO step count substantiated or corrected

**Phase 2: Claim Narrowing (Must Complete)**
- [ ] R3: RL-precondition hypothesis moved to Future Work
- [ ] R4: Abstract 0.73mm claim qualified
- [ ] R7: PID-servo bottleneck language softened
- [ ] S2: "Omnidirectional" → "Multidirectional" with asymmetry quantified
- [ ] S5: Flight/terrain demoted to "Extensions"
- [ ] S6: Ablation reframed as evidence, not Contribution #3
- [ ] S11: Acknowledge ACC F_med is comparable to (not better than) P-only

**Phase 3: Evidence Supplementation (Should Complete)**
- [ ] R6: LQR Q/R matrices and tuning process disclosed
- [ ] S1: Whleaper substantive engagement
- [ ] S3: PI+dead-zone baseline tested or absence explained
- [ ] S4: N increased for subtractive ablation OR claims narrowed
- [ ] S8: EMA₀=1.0 settle-time characterized
- [ ] S12: PID step response measured or analytically derived

**Phase 4: Positioning and Citations (Should Complete)**
- [ ] S7: Latency sensitivity elevated in abstract/conclusion
- [ ] S9: Audio/collision-detection envelope precedents cited
- [ ] S10: 53× claim reframed as limit-cycle elimination
- [ ] T14: Hybrid systems citation removed or developed

**Phase 5: Polish (Nice to Complete)**
- [ ] T1-T16: All text and formatting items addressed

---

### Estimated Revision Timeline

| Phase | Effort | Calendar |
|-------|--------|----------|
| Phase 1: Documentation Corrections | 1-2 days | 1 week |
| Phase 2: Claim Narrowing | 2-3 days (text-only) | 1-2 weeks |
| Phase 3: Evidence Supplementation | 1-3 weeks (if new experiments) | 2-4 weeks |
| Phase 4: Positioning and Citations | 1-2 days | 1 week |
| Phase 5: Polish | 1 day | 3 days |
| **Total** | **2-6 weeks** | **4-8 weeks** |

The wide range depends on whether the authors choose to run new experiments (N increase for ablation, PI+dead-zone baseline, EMA₀=1.0 settle-time sweep, PID step response measurement). If the authors choose the text-only path (narrow claims rather than add data), the revision can be completed in 2-3 weeks. If they add new data, 4-8 weeks is realistic.

---

## Part 3: Reviewer Report Summary

### EIC Report
- **Recommendation:** Major Revision
- **Score:** 78.85 / 100
- **Key strengths identified:** Asymmetric envelope follower synthesis (S1), factorial ablation design (S2), intellectual honesty about limitations (S3), thorough LQR comparison (S4), gate sensitivity heuristics (S5), reproducibility infrastructure (S6)
- **Key weaknesses identified:** RL-precondition framing untested (W1), abstract precision unqualified (W2), primary bottleneck overstatement (W3), asymmetric platform comparison table (W4), latency sensitivity under-discussed (W5), single morphology/simulator (W6)
- **Confidence:** 4/5
- **6 dimension scores provided**

### R1: Methodology / Statistical Validation Reviewer
- **Recommendation:** Major Revision
- **Score:** 68 / 100
- **3 CRITICAL findings:** Sim config misrepresentation (C1), control frequency discrepancy (C2), PPO step count unsubstantiated (C3)
- **3 MAJOR findings:** Timing cascade from C2 (M1), LQR architecture mixing (M2), N=5 underpowered (M3)
- **5 MODERATE findings:** Solver iterations (Md1), EMA settle-time (Md2), 53× claim (Md3), flight/terrain framing (Md4), sign convention documentation (Md5)
- **Confidence:** 5/5 (code-level audit at commit f92640c)
- **5 dimension scores with sub-scores provided**

### R2: Domain Reviewer (Wheeled-Legged Locomotion / WBC)
- **Recommendation:** Major Revision
- **Score:** 4.05 / 10 (different scale, see dimension weights)
- **Key gaps identified:** Whleaper not quantitatively compared, DCM/capture-point literature unaddressed, anti-windup literature straw-manned, hybrid systems citation decorative, no MPC baseline
- **Key strengths identified:** Asymmetric envelope follower, gate sensitivity analysis, factorial ablation, honest limitation disclosure
- **6 dimension scores provided**

### R3: Cross-Disciplinary / Sim-to-Real Reviewer
- **Recommendation:** Weak Accept (RA-L) / Major Revision (T-RO/TMECH)
- **Key strengths:** Asymmetric envelope follower (S1), factorial ablation (S2), torque rate-limit invariance (S3), honest limitations (S4), LQR failure analysis nuance (S5)
- **Key weaknesses:** Hardware path dangerously optimistic (W1), 0.73mm contingent on contact (W2), cross-disciplinary connections underdeveloped (W3), practical utility unclear (W4), energy/compute budgets absent (W5), safety interlock analysis incomplete (W6), PPO comparison problematic (W7)
- **10 dimension scores provided**
- **Notable contribution:** Audio-engineering connection for asymmetric envelope follower

### DA: Devil's Advocate
- **Score:** 72.7 / 100 (DA rubric)
- **3 CRITICAL issues:** F_med is WORSE than P-only (C1), PID servo ~0.25s lag unjustified (C2), three different idle precision numbers (C3)
- **5 MAJOR issues:** Flight/terrain trivial (M1), no PI+dead-zone baseline (M2), N=5 underpowered (M3), omnidirectional misnomer (M4), RL-precondition non-sequitur (M5)
- **5 ignored alternative paths identified**
- **4 missing stakeholder perspectives provided**
- **5 observations (non-defects) noted**

---

### Aggregate Recommendation

| Reviewer | Rec | Score |
|----------|-----|-------|
| EIC | Major Revision | 78.85 |
| R1 | Major Revision | 68 |
| R2 | Major Revision | 4.05/10 |
| R3 | Weak Accept (RA-L) | — |
| DA | (score only) | 72.7 |
| **Consolidated** | **MAJOR REVISION** | **73.8** |

The EIC consolidates to **MAJOR REVISION**. The two CRITICAL documentation issues (R1 C1/C2) prevent Minor Revision despite the EIC's own score (78.85) landing at the upper Minor boundary. The control-frequency discrepancy (SC5) is factual and cascading; until resolved, no timing-dependent claim in the paper can be considered reliable. Once R1-R7 are addressed, the paper would enter Minor Revision territory and be suitable for RA-L acceptance.

**The core contribution -- the proximity-gated anchor with asymmetric envelope follower -- is valid, novel, and well-ablated. The paper needs scope-narrowing, not re-execution.**