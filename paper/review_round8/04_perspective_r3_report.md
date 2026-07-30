# Peer Review Report — R3 Cross-Disciplinary / Process Control (Round 8)

## Reviewer Identity
Control theorist and industrial automation researcher. 15+ years in anti-windup compensation, override control, split-range architectures, and nonlinear control for chemical process control and energy systems. Publications in *Automatica*, *Control Engineering Practice*, *Journal of Process Control*, and *IEEE Transactions on Control Systems Technology*. Applied work on conditional integration, tracking back-calculation, and observer-based anti-windup in refinery distillation and exothermic batch reactor control.

## Overall Assessment
### Recommendation: Minor Revision
### Confidence Score: 4
### Summary Assessment

This manuscript has improved substantially since earlier rounds. The renaming from "cascade control" to "conditionally-gated control" is a genuine improvement, and the citations to Shinskey (1996) and Astrom & Hagglund (2005) for override/selector-based control show the authors have engaged with the process control literature. The Bonferroni-adjusted statistical reporting (alpha=0.007 for 7 pairwise comparisons, with Cohen's d effect sizes) is properly integrated into both the ablation table notes and the Discussion. The coupled 6-state 3D LQR baseline resolves the most significant prior concern about LQR model scope.

My remaining concerns are narrower than in prior rounds and center on (1) whether the paper fully acknowledges how much of "conditionally-gated control" is already standard practice in process control under different names, (2) whether the Anti-Windup taxonomy in Appendix A is correctly applied, and (3) whether the 5-step porting procedure is empirically validated or merely plausible. These are framing and acknowledgment issues, not threats to the core empirical contribution. The proximity-gated anchor with asymmetric envelope follower is a genuinely novel synthesis for wheeled-biped control, and the ablation evidence supports its efficacy.

## Strengths

**S1: Honest engagement with process control lineage.** The paper now cites Shinskey (1996) and Astrom & Hagglund (2005) for override/selector-based control in the Related Work and positions ACC's smoothstep gates as extending "from discrete selection to continuous blending." This is a meaningful improvement over earlier rounds that lacked any process control citations. The framing in Section IV-A explicitly distinguishes ACC from cascade control (nested loops at fixed time scales), selector-based control (discrete switching), and gain scheduling (continuous variation with one variable). *Evidence: Section II-A lines 88-89, Section IV-A lines 139.*

**S2: Coupled 6-state 3D LQR baseline is the fair comparison the paper needed.** The 6-state LQR (pitch+roll+forward, with Q/R extended via Bryson's rule) replicates the coupled structure from Zamani et al. (2020) on the authors' legged morphology. The finding that it performs identically to the 4-state variant -- confirming the PID servo path as the dominant bottleneck -- is a valuable negative result. The 46-configuration gain sweep documentation is now adequate. *Evidence: Section V-D, Table VIII (LQR baselines), footnote on Bryson's rule.*

**S3: Bonferroni context properly integrated (P2-R11 fix confirmed).** The ablation reporting now includes explicit Bonferroni-adjusted thresholds (alpha=0.007 for 7 comparisons), Cohen's d effect sizes (d~6.5 for L0 to L1, d~1.9 for L2 to L3), and the honest acknowledgment that only L0 to L1 survives correction. The Discussion section (VI) re-states this with appropriate caveats: "The primary claims do not depend on borderline significance: the ~70x idle precision improvement and S5 collapse (80N to 9N) are unambiguous regardless of correction." This is properly done. *Evidence: Table IV notes (line 485), Section VI (lines 666-667).*

**S4: Architecture figure and two-channel decomposition are clear.** Figure 2 presents the wheel-torque and leg-joint-torque channels with gates shown as dashed lines -- this is legible and communicates the conditional-activation concept effectively. The decomposition into always-active core (balance, posture) and conditionally-gated extensions (anchor, flight, terrain) is structurally clean. *Evidence: Figure 2, Equation 1.*

**S5: Parameter sensitivity analysis revealing alpha_r dominance is practically valuable.** The finding that the release coefficient dominates (sensitivity 220.4, 20x the next parameter, 130x pitch gates) is actionable. It tells practitioners exactly where to focus tuning effort when porting ACC. *Evidence: Section V-E gate sensitivity analysis.*

## Weaknesses

**W1: "Conditionally-gated control" relabels established process control practice without fully acknowledging the depth of prior art.** [Major, Confidence: 4]
- **Problem:** The paper positions ACC's gates as extending selector-based and override control "from discrete selection to continuous blending" (line 89). While the specific smoothstep-over-EMA mechanism is novel in the wheeled-biped context, the *concept* of continuous blending between control modes based on process conditions is not new. The process control literature has used: (a) split-range control with continuous valve splitting (Shinskey, 1996, Chapter 7), where two actuators share a control signal with overlapping proportional bands -- functionally identical to ACC's smoothstep gates; (b) override control with bumpless transfer (Astrom & Hagglund, 2005, Section 10.5), where switching between controllers uses tracking to prevent bumps -- analogous to ACC's C0/C1 smoothstep transitions; (c) gain scheduling with multiple scheduling variables and nonlinear interpolation (Rugh & Shamma, 2000, Section V), where the scheduling function can respond to "qualitatively distinct physical regimes." ACC's gates are distinguished by their *triggering conditions* (spatial proximity, velocity envelope, pitch stability) and *shape* (smoothstep), but the *architecture* of conditionally activating control components is standard.
- **Evidence Anchor:** Lines 88-89: "ACC's smoothstep gates extend this principle from discrete selection to continuous blending." Lines 139: "a continuous generalization of selector-based and override control from discrete switching to smooth blending."
- **Why it matters:** The paper would be stronger if it positioned ACC as a specific, well-engineered *application* of continuous override control to wheeled-biped balancing, rather than implying that continuous blending itself is novel. The contribution is the synthesis and physical-condition design, not the architecture concept. This distinction matters for scholarly credit assignment and for readers from process control who will immediately recognize the pattern.
- **Suggestion:** Add one paragraph to Related Work or Section IV-A explicitly acknowledging: "Continuous blending of control modes via overlapping activation functions is standard in split-range control (Shinskey, 1996, Ch. 7) and bumpless-transfer override control (Astrom & Hagglund, 2005, Sec. 10.5). ACC's contribution is not the blending concept itself but the specific design of physically-motivated gating conditions (spatial proximity, asymmetric velocity envelope, pitch stability) that map the precision-vs-recovery trade-off onto activation functions -- a problem-specific synthesis for which off-the-shelf process control patterns provide no direct recipe."
- **Severity:** Major (framing, not empirical)
- **Confidence:** 4

**W2: Anti-windup taxonomy in Appendix A is correctly structured but the distinctiveness claim needs qualification.** [Major, Confidence: 5]
- **Problem:** Appendix A correctly identifies the three main AW families: conditional integration, tracking back-calculation, and observer-based AW. The claim that ACC is "distinct from conditional integration in gating on physical state rather than control error" is technically accurate as stated but potentially misleading. In process control practice, conditional integration frequently gates on *process state* (temperature zones in exothermic reactors, level bands in surge tanks, pressure regions in compressors) -- not just control error. A temperature-zone-gated integral in a batch reactor controller is structurally identical to ACC's proximity-gated integral: both gate an integral term based on a physical state variable entering a defined band. The distinctive features of ACC's anchor are: (i) the *asymmetric envelope follower* as the gating variable (rather than a direct state measurement), and (ii) the *coupling of the gated integral with envelope-driven damping boost* into a joint mechanism. These are genuine distinctions, but they are more specific than "gating on physical state rather than control error" implies.
- **Evidence Anchor:** Appendix A lines 704-706: "ACC's anchor is thus a state-gated integral with envelope-detected damping -- complementary to saturation-triggered AW and distinct from conditional integration in gating on physical state rather than control error."
- **Why it matters:** The AW community will scrutinize this claim. The paper should be precise about what is and is not novel relative to industrial AW practice to avoid appearing to claim novelty for standard techniques.
- **Suggestion:** Revise Appendix A to state: "ACC's anchor differs from standard conditional integration in two respects: (i) the gating variable is derived from an asymmetric velocity envelope rather than directly from a process state, and (ii) the gated integral is coupled with an envelope-driven damping boost that is simultaneously activated, creating a joint integral-plus-damping mechanism not found in standard AW. Gating on physical state per se is common in industrial conditional integration (e.g., temperature-zone-gated integrals in batch reactors); ACC's specific envelope-detected, boost-coupled form is the distinction."
- **Severity:** Major (control-theoretic precision)
- **Confidence:** 5 (this is my primary research area)

**W3: The 5-step porting procedure is heuristic; generality claims are untested.** [Major, Confidence: 4]
- **Problem:** The 5-step tuning procedure (Section V-E) provides pragmatic guidance but its claim to generality is not empirically supported: "A fixed kp and anchor integral gain Ki require minimal adjustment across platforms; the critical parameter is tau_release" (line 647). This claim has not been tested on any platform other than the authors' 8.1kg, 0.54m-CoM-height wheeled biped. The procedure's dependence on identifying a "dominant ringdown period Tr" assumes the post-push dynamics are well-approximated by a single dominant mode -- an assumption that holds for a rigid inverted pendulum but may fail on morphologies with significant leg compliance, multi-link dynamics, or coupled pitch-roll modes.
- **Evidence Anchor:** Line 647: "A fixed kp and anchor integral gain Ki require minimal adjustment across platforms; the critical parameter is tau_release." Limitation (8) line 678: "Architectural generality across morphologies and simulators remains to be demonstrated."
- **Why it matters:** The paper's practical impact claim rests on transferability. The limitation is honestly disclosed (Limitation 8), but the tuning-procedure text in Section V-E makes stronger claims than the limitation section permits. Internal consistency requires either hedging the tuning claims or removing the generality language.
- **Suggestion:** Either: (a) add a sentence to the tuning procedure: "This procedure has been validated only on the 8.1kg, 0.54m-CoM platform described in Section III; validation on morphologies with substantially different dynamics (e.g., quadrupeds, tall humanoids, compliant-leg platforms) is future work," or (b) rephrase Step 1 to acknowledge the single-mode assumption: "For platforms whose post-push dynamics are dominated by a single ringdown mode (as in the rigid inverted-pendulum approximation), identify the dominant ringdown period Tr from free decay simulation."
- **Severity:** Major (claim-evidence alignment)
- **Confidence:** 4

**W4: Smoothstep-over-EMA continuity claim is technically imprecise.** [Minor, Confidence: 5]
- **Problem:** The paper states (line 135, footnote): "Smoothstep itself is C1; composed with the C0 asymmetric EMA (Eq. 6), composite g_env is C0." This is a corrected statement from earlier rounds and is technically correct. However, the practical implication -- that C0 continuity matters for control smoothness -- is not explored. The paper could strengthen this by noting that C0 continuity is sufficient to prevent torque discontinuities (since the torque equation involves g_anchor * tau_anchor, and a C0 gate produces a C0 torque signal), while C1 would additionally prevent "kinks" in the torque rate, which matter only if the torque rate limit binds at the switching point.
- **Evidence Anchor:** Line 135 footnote.
- **Why it matters:** Minor, but a control audience expects precise treatment of smoothness claims.
- **Suggestion:** Add a brief note: "C0 continuity of the composite gate ensures torque continuity (no step changes in commanded torque); the torque rate limit (400 Nm/s) provides additional smoothing, making C1 continuity of the gate itself unnecessary for this application."
- **Severity:** Minor
- **Confidence:** 5

**W5: The bidirectional coupling between kp schedule and anchor integral is disclosed but under-analyzed.** [Minor, Confidence: 4]
- **Problem:** The paper acknowledges (Limitation 7, line 677) that the scheduled-kp variant "introduces bidirectional coupling with the anchor integral, forming an implicit algebraic loop resolved at 100 Hz by one-sample computational delay." This is honest disclosure. However, the paper then notes (S1 vs S0 in Table IV) that the scheduled variant does not improve push survival over fixed kp=50, effectively rendering the scheduled variant unnecessary. A process control reviewer would ask: if the scheduled variant provides no benefit and introduces an algebraic loop, why retain it? The paper keeps it "for completeness" (line 188), but completeness of a component that adds coupling without benefit is not a control-theoretic virtue.
- **Evidence Anchor:** Line 188: "The schedule is retained for completeness." Limitation 7, line 677.
- **Why it matters:** Minor, but retaining a component with documented coupling and no demonstrated benefit weakens the architectural clarity. The paper's own data (S1 vs S0) shows fixed kp=50 is equivalent or better.
- **Suggestion:** Consider moving the scheduled-kp variant to an appendix or explicitly recommending fixed kp=50 as the default configuration, with the schedule documented only for reference. This simplifies the architecture and eliminates the algebraic loop concern entirely.
- **Severity:** Minor
- **Confidence:** 4

## Detailed Comments

### ACC Formulation vs. Process Control Literature

ACC's two-channel torque assembly (Equation 1) structurally resembles **split-range control with overlapping proportional bands**. In chemical process control, a split-range setup assigns two actuators (e.g., a heating valve and a cooling valve) to a single controller output, with each actuator active in a defined band of the control signal. The transition between actuators uses overlapping proportional bands to prevent dead zones -- functionally identical to ACC's smoothstep gates creating overlapping activation regions between control components.

The key difference ACC brings is that the *gating variable* is not the control signal itself (as in split-range) but an independently computed physical condition (proximity, velocity envelope, pitch stability). This is structurally closer to **override control with external reset** (Astrom & Hagglund, 2005, Section 10.5), where an external process variable overrides the primary controller. ACC generalizes this by using *continuous* (smoothstep) rather than discrete (min/max selector) override, and by having *multiple simultaneous* overrides rather than a single min/max selection.

The paper's framing at lines 88-89 and 139 is close to correct but could be more precise: "ACC's smoothstep gates extend selector-based and override control from discrete selection to continuous blending" should be "ACC applies continuous blending -- a standard technique in split-range control -- to the problem of selecting among control components based on physically-motivated gating conditions, extending override control from single-variable min/max selection to multi-condition smooth activation." The difference is acknowledging that continuous blending itself is standard, while the *application* to physically-gated control component selection is the contribution.

### Appendix A: AW Taxonomy Assessment

The taxonomy in Appendix A is structurally correct (three families identified) but shallow. Specific observations:

1. **Conditional integration:** The paper correctly notes that standard conditional integration freezes or resets the integrator based on actuator saturation or control error sign. ACC's anchor adds: (a) spatial gating (proximity), (b) envelope-detected quietness gating, and (c) coupled damping boost. These are genuine extensions, but the paper should acknowledge that state-based (not just error-based) conditional integration is already practiced in industry.

2. **Tracking back-calculation:** The paper mentions this family but does not test it. Tracking back-calculation (also called "conditioning technique" or "Hanus conditioning") feeds the difference between saturated and unsaturated control signal back to the integrator through a tracking time constant. This is the most widely deployed industrial AW scheme and would be straightforward to implement on ACC's anchor integral. A brief explanation of why tracking back-calculation is expected to fail (or a commitment to test it) would strengthen the Appendix.

3. **Observer-based AW:** The paper mentions this family but does not engage with it. Observer-based AW (Hippe, 2006) estimates the unsaturated plant state and uses it for feedback, preventing windup by maintaining a consistent internal model. This is the most theoretically sophisticated AW family and is acknowledged as beyond scope -- which is acceptable for an empirically-focused paper.

4. **The "no direct AW analogue" claim:** The claim that the coupled integral-plus-boost mechanism has "no direct AW analogue" (line 704-706) is reasonable. In standard AW, the anti-windup mechanism acts on the integrator alone; it does not simultaneously activate a damping term. The joint activation of integral and damping is genuinely outside the standard AW taxonomy and could be described as a "conditionally-gated integral-damping pair" -- a term that accurately captures its distinctiveness.

### "Conditionally-Gated" as a Term

From a process control perspective, "conditionally-gated control" is a reasonable descriptive term for what ACC does, but it is not a new *category* of controller. It is a specific *design pattern* within the broader family of multi-mode/multi-region controllers that includes:

- Gain scheduling (Rugh & Shamma, 2000): parameters vary with scheduling variables
- Override/selector control (Shinskey, 1996): one controller active at a time based on process conditions
- Split-range control (Astrom & Hagglund, 2005): actuators share control signal with defined activation bands
- Hybrid control / switched systems (Liberzon, 2003): discrete modes with continuous dynamics
- State-dependent integral gating (industrial practice): integrator active only in specified process state bands

ACC combines elements of all five. The novelty is in the specific synthesis -- not in the concept of conditional activation. The term "conditionally-gated" is useful as a *descriptor* of ACC's mechanism but should not be presented as a new control paradigm. The paper largely avoids this overclaim now (the shift from "cascade control" to "conditionally-gated control" is already a significant improvement), but residual language in the abstract and introduction occasionally implies paradigm-level novelty.

**Recommendation:** The paper should explicitly position ACC as a "multi-region control architecture combining continuous gain scheduling, state-dependent integral gating, and envelope-driven mode switching" -- acknowledging each established technique's contribution to the synthesis. This would strengthen the paper's scholarly positioning without weakening its contribution claim.

### Gate Design Methodology Assessment

The 5-step procedure (Section V-E) is:
1. Identify dominant ringdown period Tr from post-push free decay
2. Set tau_release ~ 3-5 x Tr
3. Set quiet-stance envelope band to 2-3x idle velocity noise floor
4. Set pitch stability gates conservatively (+/-12 deg, +/-15 deg/s)
5. Tune kp for desired idle precision

**Assessment of each step:**

- **Step 1:** Well-justified for the rigid inverted-pendulum class. The ringdown period Tr is identifiable from a single push simulation. Caveat: assumes single dominant mode. On a compliant platform with multiple closely-spaced modes, identifying "the" ringdown period is ambiguous.

- **Step 2:** The factor 3-5x Tr is empirically supported by the sensitivity analysis (alpha_r dominates, sensitivity 220.4) but has no theoretical derivation. A describing-function analysis of the smoothstep gate nonlinearity could derive this factor from the gain-reduction region width, but this is beyond the paper's empirical scope. The heuristic is acceptable for an empirical paper.

- **Step 3:** 2-3x idle velocity noise floor is a standard signal-to-noise margin. Well-justified.

- **Step 4:** Conservative pitch stability thresholds are sensible as safety interlocks. The sensitivity analysis confirms they are effectively insensitive (sensitivity ~0), so the exact values are not critical.

- **Step 5:** Fixed kp=50 is empirically validated as sufficient. The scheduled variant is shown to provide no additional benefit.

**Overall:** The procedure is well-documented, empirically grounded for this platform, and transparent about its heuristic nature. The only concern is the untested generality claim (see W3 above).

## Questions for Authors

**Q1:** In process control, split-range configurations routinely use overlapping proportional bands to achieve continuous blending between actuators. ACC's smoothstep gates serve the same function -- continuous activation based on a scheduling variable. Given that the smoothstep function itself is standard (used in computer graphics since the 1990s), what specifically distinguishes ACC's gating from a multi-variable split-range configuration with smoothstep scheduling functions? Is the distinction in the choice of gating variables (spatial proximity, velocity envelope, pitch stability), or in the specific mapping of these variables to component activation?

**Q2:** Appendix A mentions tracking back-calculation AW as one of the three main AW families but does not test it. Tracking back-calculation would feed the difference between the saturated and unsaturated position-integral command back to the integrator. Given that ACC already has an integral cap (2.0 Nm) and a leak term (5e-3 per step), would tracking back-calculation provide any additional benefit? Or does the spatial proximity gate achieve the same effect through a different mechanism?

**Q3:** The 5-step porting procedure assumes a single dominant ringdown period Tr. On a morphology with significant leg compliance (e.g., a quadruped in stance), the post-push dynamics may involve multiple coupled modes (body pitch, leg spring, wheel-hop). How would you extend the procedure to multi-mode ringdown? Would multiple envelope followers tuned to different frequency bands be needed, or does the single asymmetric EMA remain sufficient?

**Q4:** The paper reports that fixed kp=50 matches or exceeds scheduled kp in all metrics (S1 vs S0). Given that the scheduled variant introduces an implicit algebraic loop (Limitation 7), requires additional parameters, and provides no benefit, what is the rationale for retaining it? Is there a scenario not tested in the current evaluation (e.g., very large pushes beyond 160N, or a different platform morphology) where scheduling would become necessary?

## Minor Issues

**M1:** The footnote on line 135 correctly states the composite gate is C0, but the text at line 139 says "continuous smoothstep" without distinguishing C0 from C1. Consider "C0-continuous smoothstep" or "C0-smoothstep" for precision.

**M2:** Line 647: "A fixed kp and anchor integral gain Ki require minimal adjustment across platforms." The word "require" implies demonstrated fact. Replace with "are hypothesized to require" or "we expect to require" since cross-platform validation is pending (Limitation 8).

**M3:** The term "ringdown" is used extensively but never formally defined in the main text body (first appears in a table caption). Add a one-sentence definition at first use: "Ringdown refers to the post-disturbance decay of velocity oscillations to the quiet-stance envelope band."

**M4:** The asymmetric EMA (Eq. 6) is mathematically a standard peak-hold/envelope detector used in audio dynamics processing since the 1970s. Citing one reference from this lineage (e.g., Giannoulis et al., 2012, "Digital Dynamic Range Compressor Design -- A Tutorial and Analysis," JAES) would strengthen the signal-processing context.

**M5:** Section V-E sensitivity analysis reports raw finite-difference gradients. Consider reporting sensitivity as *elasticity* (percentage change in outcome per percentage change in parameter) for parameters with different units, which is more interpretable for practitioners.

## Dimension Scores

| Dimension | Score (0-100) | Notes |
|-----------|--------------|-------|
| Originality (20%) | 72 | Proximity-gated anchor + asymmetric envelope is genuinely novel synthesis for wheeled bipeds; individual components (envelope detection, conditional integration, smoothstep blending) have prior art in process control and signal processing |
| Argument Coherence (15%) | 82 | Clear narrative from P-only limitation through anchor to ablation validation; Bonferroni and effect-size reporting is properly integrated |
| Significance & Impact (20%) | 68 | Practical value for wheeled-biped experimentalists is high; claim to generality across morphologies is untested; process control practitioners will recognize the architecture pattern as familiar |
| Methodological Rigor (20%) | 78 | Factorial ablation (additive + subtractive) is exemplary; statistical power analysis is honest; AW taxonomy engagement is correct but shallow; 5-step porting procedure is heuristic, not validated |
| Control-Theoretic Soundness (15%) | 58 | No Lyapunov/passivity/describing-function analysis; empirical validation compensates partially; C0-vs-C1 continuity distinction is now correct; implicit algebraic loop is disclosed but not analyzed |
| Literature Integration (10%) | 65 | Process control citations (Shinskey, Astrom & Hagglund) are present but under-engaged; envelope-detection signal processing lineage is not cited; aerospace bumpless-transfer literature is absent |
| **Weighted Total** | **71** | Solid empirical contribution with framing that needs further precision on what is and is not novel relative to process control practice |

## Revision Priority Summary

**Must address (blocking):**
- W2: Qualify the "gating on physical state rather than control error" distinctiveness claim in Appendix A; acknowledge that state-gated conditional integration is already practiced in industry
- W1: Add explicit acknowledgment that continuous blending of control modes is standard in split-range control; reposition ACC's contribution as the specific synthesis of physically-motivated gating conditions

**Should address (strongly recommended):**
- W3: Hedge the generality claims in the 5-step porting procedure; acknowledge single-mode assumption and single-platform validation
- Q1-Q4: Answer in author response

**Optional:**
- W4: Clarify C0 vs C1 practical implications
- W5: Consider removing or demoting the scheduled-kp variant
- M1-M5: Minor text and citation improvements
