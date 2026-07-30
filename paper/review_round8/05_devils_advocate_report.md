# Devil's Advocate Report -- Round 8

**Paper:** "Anchored Conditionally-Gated Control for Wheeled Bipedal Balance and Disturbance Recovery"
**Author:** Van Thuong Nguyen
**Review Date:** 2026-07-28
**Review Mode:** Full stress-test across 9 challenge dimensions with Round 7 P2 fix verification

---

## Round 7 P2 Fix Verification

Before presenting new findings, I verify the three P2 fixes specifically called out for this round:

### P2-R9: Non-monotonic noise explained -- ADDRESSED, but explanation reveals deeper fragility

**What was requested:** Explain why high noise (3.00mm idle RMS) outperforms medium noise (34.00mm idle RMS) at 0ms delay.

**What the paper now says** (Table VI footnote, lines 641--642): "The apparent improvement from medium to high noise at 0ms delay (34mm->3mm RMS) occurs because high noise holds the EMA permanently above threshold, disengaging the anchor -- the robot reverts to P-only behavior, avoiding the partial-engagement limit cycle of medium noise. The non-monotonic low-noise idle RMS (12.82mm at 10ms vs. 7.31mm at 30ms) reflects partial-engagement resonance: 10ms phase-shifts the anchor into intermittent engage/release; at 30ms the EMA stays above threshold, avoiding resonance."

**Devil's Advocate assessment:** The explanation is physically plausible and acknowledges the phenomenon rather than hiding it. However, "partial-engagement resonance" is a **post-hoc label, not a modeled or predicted phenomenon**. The paper provides no quantitative model of the engage/release duty cycle, no transfer function from noise amplitude to gate state, and no prediction of where the worst-case resonance occurs. This is a description, not an explanation. More importantly, the phenomenon exposes a **fundamental structural fragility** -- ACC has three qualitatively distinct operating regimes (anchor-engaged, partially-engaged, anchor-disengaged/P-only), and the middle regime is *worse than having no anchor at all*. The controller has no mechanism to detect which regime it is in or to avoid the pathological middle regime. This is not a tuning issue -- it is inherent to any binary-gated architecture applied to continuous physical systems with noise.

**Verdict:** The text-level fix is adequate for disclosure. But the underlying fragility is now *more visible*, not less concerning. This becomes **M1** in the new issue list below.

### P2-R12: Zamani comparison deepened -- ADDRESSED, but creates internal contradiction

**What was requested:** Deepen the Zamani et al. (2020) comparison.

**What the paper now says** (lines 87--88): "Zamani et al. demonstrated coupled pitch-roll LQR on a rigid two-wheeled platform without articulated legs, where torque commands act directly on wheels. Our 6-state LQR replicates their coupled structure on a legged morphology and finds the PID servo path prevents stabilization despite ~7.7x stronger hip-roll gains: the ~0.25s servo lag -- absent in Zamani's direct-drive platform -- introduces phase margin loss that the coupled LQR cannot overcome, confirming the servo path as the dominant bottleneck."

**Devil's Advocate assessment:** The comparison is substantively deeper and now cites specific metrics from Zamani. However, the fix has created a **new internal contradiction** that was not present in the Round 7 version:

1. The paper's argument: Zamani's LQR succeeds because their platform is direct-drive (no PID servo lag).
2. The paper's own data: ACC's Direct Torque LQR (which also has NO PID servo lag) still fails 100%, with only 35% survival improvement.
3. Therefore: if removing the servo lag doesn't prevent failure on ACC's platform, Zamani's success cannot be attributed solely to the absence of servo lag. Zamani's platform must differ in additional ways (rigid two-wheeled body, no articulated legs, different mass distribution, different actuator authority) that the paper does not discuss.

The paper is now using Zamani's results to argue that servo lag IS the bottleneck, while its own direct-torque data shows that removing servo lag DOES NOT solve the problem. This is a logical inconsistency that a sharp reviewer will notice. Either:
- (a) Zamani's success is due to platform differences beyond servo architecture, in which case the comparison does not support the bottleneck claim; OR
- (b) ACC's Direct Torque LQR is not a fair test of direct-drive LQR (e.g., gains were not re-optimized for the direct-torque plant), in which case the direct-torque result is not probative.

The paper doesn't resolve this tension. This becomes **M2** below.

### P2-R13: 46-config footnote arithmetic -- FIXED

**What was requested:** Clarify or correct the 46-configuration sweep count.

**What the paper now says** (line 595 footnote): "Swept 35 roll PD combinations (kp_roll in {0.4,0.8,1.2,1.6,2.0,3.0,5.0}, kd_roll in {0.08,0.2,0.5,1.0,2.0}), 5 wheel velocity limits ({20,30,40,60,100} rad/s), and 6 roll--wheel cross-coupling gains ({0.0,0.5,1.0,2.0,5.0,10.0})."

**Devil's Advocate assessment:** 7 x 5 = 35 + 5 + 6 = 46. Arithmetic now consistent. **FIXED.**

However, a separate concern (not about arithmetic): the sweep tests roll *PD* gains (kp_roll, kd_roll) and wheel velocity limits -- parameters of the *decoupled* PD roll controller, not parameters of the coupled 6-state LQR. The coupled LQR derives its roll gains from the Riccati solution, not from a PD parameter sweep. Sweeping PD parameters on the decoupled controller does not probe whether the coupled LQR's Riccati-derived gains could work. The sweep validates that no amount of roll PD tuning rescues the *decoupled* baseline, but the paper uses this to claim the *coupled* 6-state LQR failure is "structural, not parametric." The sweep scope doesn't match the claim. See **M3** below.

---

## Strongest Counter-Argument

The paper's Round 7 revisions have created a self-undermining architecture. Three data points -- each added or modified in response to Round 7 critiques -- now form an internal contradiction that was absent in the original submission:

1. **S1 (fixed k_p=50) outperforms S0 (scheduled k_p) on push survival** -- acknowledged by the paper, which now "recommends" fixed k_p=50 (line 188). Yet the scheduled variant is still presented in the architecture diagram, equation (1), and the "Full ACC" configuration. A component that the paper's own data shows is either superfluous or detrimental is still part of the architectural specification. "Retained for completeness" (line 189) is not a scientific justification -- it means the paper describes an architecture it does not recommend.

2. **The Zamani comparison now argues servo lag is the bottleneck** (lines 87--88: Zamani succeeds with direct-drive because no servo lag), but the paper's own Direct Torque LQR (which also has no servo lag) fails 100% with only 35% improvement. The comparison simultaneously cites Zamani to prove the servo-lag hypothesis AND presents data that contradicts the hypothesis. The reader is left with: Zamani's LQR works on Zamani's platform; ACC's LQR doesn't work on ACC's platform regardless of action path. The platform difference, not the action path, appears to be the dominant factor.

3. **The non-monotonic noise response explanation** (P2-R9 fix) reveals that ACC has a pathological "partial-engagement" regime where performance is WORSE than the P-only baseline. The controller has no mechanism to detect or avoid this regime. This is not a corner case -- medium sensor noise (5x consumer IMU levels) with zero delay produces 34mm idle RMS, compared to 0.42mm clean and 39mm P-only. The anchor mechanism creates a performance discontinuity such that with moderate noise, the user would be better off disabling it entirely.

The tension: the paper presents ACC as a "unified architecture" with "five components gated by physical conditions," but the data increasingly suggests the architecture should be simplified to: fixed k_p + anchor integral with asymmetric EMA gate + damping boost + pitch safety gate, where the k_p schedule is removed, the "conditionally-gated architecture" framing is reduced to "a gated integral with envelope-based dead-zone detection," and the hardware path must confront the fact that the precision mechanism collapses under the noise levels of real IMUs before it even reaches the latency bottleneck.

This is not to say ACC doesn't work in simulation -- it clearly does, and the 0.73mm idle precision with 80N omnidirectional push survival is a genuine engineering achievement. But the paper's architectural narrative has become less coherent, not more, after the Round 7 fixes. The data now tells a simpler, more honest story than the framing: a well-tuned PD + proximity-gated integral + asymmetric envelope detector works well on this specific simulated platform under clean conditions. That story is publishable but needs to be told as engineering, not as architectural innovation.

---

## Issue List

### CRITICAL

#### C1 -- Architecture-description mismatch: k_p schedule "retained for completeness" despite contradicting data
- **Dimension:** Core novelty (D1), Logic chain validation (D4)
- **Location:** Section IV-B, lines 187--193; Table IV S0 vs S1 rows; architecture Figure 2
- **Problem:** S1 (fixed k_p=50) achieves F_min=83.1N and F_med=118.8N vs S0 (scheduled k_p) at 80.4N and 110.1N -- a 3--8% improvement at identical precision. The paper now acknowledges this (line 188: "Fixed k_p=50 is recommended") but states the schedule is "retained for completeness" (line 189). The scheduled variant remains in the architecture diagram, equation (1), and defines the "Full ACC" (S0) reference configuration.
- **Why CRITICAL:** A paper cannot simultaneously recommend configuration A (fixed k_p) while presenting configuration B (scheduled k_p) as its canonical architecture. "Retained for completeness" means the paper's own architectural specification includes a component that its own data shows should not be used. This is not a minor framing issue -- it means the paper's description of "what ACC is" differs from "what ACC should be." If the schedule is purely incidental (as the data shows), removing it from the architecture would (a) simplify the controller, (b) make the paper's recommendation match its description, and (c) remove one bidirectional coupling pathway (the k_p schedule creates an implicit algebraic loop with the anchor integral, acknowledged in Limitation 7, line 676). Keeping it creates unnecessary complexity and a documented algebraic loop for no empirical benefit. A reviewer will ask: if it doesn't help and has a known stability concern (algebraic loop), why is it still there?
- **Evidence:** Table IV, S0 vs S1 comparison. Text line 188--189. Text line 676 (Limitation 7: bidirectional coupling via algebraic loop).
- **Required fix:** Either (a) remove the k_p schedule from the architecture entirely, re-label S1 as "Full ACC," and delete equation (1) -- the cleanest fix; OR (b) provide a principled reason to retain it despite the data (e.g., it enables future extensions, it provides a safety margin under conditions not yet tested). "Retained for completeness" is not sufficient.

#### C2 -- Zamani comparison now contradicts the bottleneck claim it was added to support
- **Dimension:** Logic chain validation (D4), Confirmation bias detection (D3)
- **Location:** Section II-A, lines 87--88; Section V-D, Table V Direct Torque row
- **Problem:** The deepened Zamani comparison (P2-R12 fix) argues that Zamani's coupled LQR succeeds because their platform is direct-drive (no PID servo lag), and that ACC's 6-state LQR fails because of servo lag. But ACC's Direct Torque LQR variant -- which also has no PID servo lag -- still fails 100%, with only 35% survival improvement (0.52s to 0.70s). The paper is using Zamani to prove that servo lag IS the bottleneck, while its own direct-torque data shows that removing servo lag DOES NOT prevent failure. The two data points (Zamani's success, ACC Direct Torque's failure) are both attributed to the same variable (servo lag), but point in opposite directions. This is a logical inconsistency visible to any careful reader.
- **Why CRITICAL:** The Zamani comparison was added in Round 7 to strengthen the bottleneck argument. Instead, it has created a contradiction: if removing servo lag was sufficient for Zamani, why is it insufficient for ACC? The obvious answer -- platform differences beyond servo architecture (rigid two-wheeled body, no articulated legs, different mass/inertia, different actuator authority) -- means that Zamani's success says little about ACC's bottleneck. The paper cannot cite Zamani as supporting evidence for a claim that its own data rejects. The bottleneck narrative now has three mutually inconsistent pieces: (1) Zamani works without servo lag; (2) ACC's 6-state LQR fails WITH servo lag; (3) ACC's Direct Torque LQR fails WITHOUT servo lag. These three facts cannot all be explained by "servo lag is the dominant bottleneck."
- **Evidence:** Lines 87--88 (Zamani comparison text claiming servo lag as explanation). Table V, LQR DT row: survival 0.70s, 100% fall rate (direct-torque, no servo lag).
- **Required fix:** Resolve the contradiction explicitly. Options: (a) acknowledge that platform differences (mass distribution, leg inertia, actuator limits) are likely the primary factor in Zamani's success, and the servo-lag comparison is therefore inconclusive; (b) demonstrate that ACC's Direct Torque LQR was not fairly tuned for the direct-torque plant (gains remained optimized for the servo plant), making the direct-torque result conservative; (c) run the coupled 6-state LQR in direct-torque mode -- if it succeeds where the 4-state direct-torque fails, that would genuinely demonstrate that controller structure matters once the servo bottleneck is removed. Option (c) is deferred to "future work" (line 596), which means the paper currently has no resolution for this contradiction.

### MAJOR

#### M1 -- Gate-state bifurcation is a systematic fragility, not a tuning corner case
- **Dimension:** Overgeneralization check (D5), "So what?" test (D8)
- **Location:** Table VI, Section V-E robustness analysis
- **Problem:** The non-monotonic noise response (P2-R9 fix) reveals three qualitatively distinct regimes: (a) low noise: anchor engaged, excellent precision (0.42mm); (b) medium noise: partial engagement, WORSE than P-only (34mm); (c) high noise: anchor disengaged, P-only behavior (3mm). The controller has no mechanism to detect which regime it is in or to avoid regime (b). Regime (b) is a *performance cliff* -- add noise gradually and performance degrades severely, then suddenly IMPROVES as the anchor disengages entirely. This means ACC's precision advantage is not robust to gradual sensor degradation: as an IMU ages or as vibration increases, the controller transitions through a worst-case regime where the user would be better off disabling the anchor entirely. The paper describes this phenomenon but does not treat it as the fundamental architectural limitation that it is. Any controller using binary thresholding on noisy signals will exhibit similar bifurcation. The paper should discuss whether this is fixable (hysteresis? adaptive thresholds? probabilistic gating?) or inherent to the approach.
- **Evidence:** Table VI: Clean 0ms = 0.42mm, Med noise 0ms = 34.00mm, High noise 0ms = 3.00mm. The medium-noise regime is 80x worse than clean and 11x worse than high-noise (anchor-disabled).
- **Why MAJOR rather than CRITICAL:** The paper does disclose the data and provides explanation. The issue is the failure to discuss this as a fundamental limitation of threshold-based gating, rather than as an anomaly to be explained away.

#### M2 -- "ACC as precondition for RL" remains unevidenced (carried from Round 7 M3)
- **Dimension:** Logic chain validation (D4)
- **Location:** Section I, line 66: "ACC is thus a precondition for RL on this platform: a height-general structured prior that a residual policy can later augment."
- **Problem:** This claim was identified as M3 in Round 7 and listed as P3-R18 ("Remove or down-scope 'ACC as precondition for RL' claim -- zero RL+ACC evidence"). The claim remains in the paper unchanged. There is still zero RL+ACC experimental evidence. The CLAUDE.md (project instructions, not part of the paper) confirms that residual RL is planned but not trained. A claim that requires joint evidence (ACC+RL succeeds where pure RL fails) is supported by neither arm. The paper reports one failed PPO run as the sole basis for concluding that ACC is necessary.
- **Evidence:** Line 66. Absence of any RL+ACC experiment in the Results section.
- **Why MAJOR rather than CRITICAL:** The claim is peripheral to the paper's core contribution (the ACC controller itself). But it remains a substantive unsupported claim that a reviewer can flag.

#### M3 -- LQR gain sweep validates the wrong controller
- **Dimension:** Confirmation bias detection (D3), Cherry-picking (D2)
- **Location:** Section V-D, line 595 footnote
- **Problem:** The 46-configuration sweep (now with corrected arithmetic) tests roll PD gains (kp_roll, kd_roll), wheel velocity limits, and roll--wheel cross-coupling gains. These are parameters of the *decoupled PD roll controller* used in the 4-state LQR baseline. They are NOT parameters of the coupled 6-state LQR, whose roll gains come from the Riccati solution (Q and R matrices). The paper claims the sweep shows that the coupled LQR failure is "structural, not parametric" (line 595), but the sweep doesn't probe the coupled LQR's parameter space at all. The 6-state LQR's performance depends on its Q/R matrices -- sweeping PD gains on a different controller cannot validate that the 6-state LQR is optimally tuned. Furthermore, the sweep is of individual parameters swept independently, not a full factorial grid. A parameter could interact with another in ways not captured by independent sweeps.
- **Evidence:** Line 595 footnote: sweep parameters are kp_roll, kd_roll, wheel velocity limits, cross-coupling gains -- all PD controller parameters. Line 566: 6-state LQR uses Q=diag(10,2,3,0.5,3,0.3), R=diag(0.8,1.0), i.e., Riccati-derived gains.
- **Why MAJOR:** The sweep is presented as validating a claim about the coupled LQR but tests a different controller's parameter space. This is a methodological mismatch. The claim "structural, not parametric" requires sweeping the 6-state LQR's Q/R matrices, not the decoupled PD's gains.

#### M4 -- 0.25s servo lag figure remains unjustified (carried from Round 7 m5)
- **Dimension:** Logic chain validation (D4)
- **Location:** Section V-D, line 563: "outputting targets through a PID servo layer (~0.25s lag)"
- **Problem:** This was raised as m5 in Round 7. The paper still provides no derivation, measurement, or justification for the ~0.25s figure. At 100Hz control, a reasonably tuned position servo can achieve settling times well below 250ms. If the 0.25s figure is due to deliberately conservative PID tuning, that tuning choice -- not the PID servo architecture -- is the bottleneck. The paper needs to either (a) provide the PID gains and a step-response measurement showing ~0.25s settling, or (b) acknowledge that the PID was conservatively tuned and that a more aggressive tuning could reduce the lag. Without this, the "servo lag is the bottleneck" claim rests on an unverified assumption about PID tuning.
- **Evidence:** Line 563. Absence of PID gain values, step response data, or settling time measurement in the paper or supplementary material.
- **Why MAJOR:** This figure is load-bearing for the central bottleneck claim (C2 above). If the PID can be tuned faster, the servo-lag bottleneck may be a tuning artifact, not an architectural constraint.

#### M5 -- Latency sensitivity and hardware pathway are in fundamental tension
- **Dimension:** "So what?" test (D8), Overgeneralization (D5)
- **Location:** Section V-E (robustness), Section VI (Path to Hardware)
- **Problem:** The paper acknowledges ACC is "fundamentally latency-sensitive with a threshold below 10ms" (Limitation 4, line 673). The hardware pathway estimates end-to-end latency at ~9.5ms (line 682) -- within the 10ms window but with zero margin. Any of the following would push latency past the threshold: a slower IMU readout, additional filtering, a heavier control computation, bus contention, or OS scheduling jitter. The paper's own robustness data shows that ANY non-zero delay bifurcates F_max from ~95N to ~45N (Table VI). So on hardware, ACC would operate with at best ~45N push tolerance -- less than half the simulated performance -- and at worst could fail entirely if latency exceeds 10ms. The paper presents the ~9.5ms estimate as reassuring, but a 0.5ms margin on a real-time embedded system with multiple subsystems is not credible. A hardware practitioner would expect 2--5ms of jitter alone.
- **Evidence:** Table VI: F_max 96N at 0ms vs 47N at 10ms. Line 682: ~9.5ms hardware latency estimate. Line 673: "<10ms threshold."
- **Why MAJOR:** The paper's strongest results (80N omnidirectional push, 0.73mm precision) are conditional on <10ms latency that may not be achievable on hardware. The claimed "hardware-aligned" design (line 661) is in tension with the latency sensitivity data.

#### M6 -- 50+ hand-tuned gains without methodology (carried from Round 7 M4, not addressed)
- **Dimension:** Alternative paths analysis (D6), Core thesis (D1)
- **Location:** Section IV-F, line 355: "the full set (50+ gains) is available in the supplementary material"
- **Problem:** This was M4 in Round 7. The paper has not changed. The 5-step porting heuristic (lines 646--647) provides guidance for the dominant parameter (alpha_r) but does not explain how the remaining 49+ gains were chosen. With 50+ hand-tuned parameters, the space of possible controllers is enormous. Without documenting the tuning process, the paper cannot distinguish between "this architecture works" and "this specific parameterization of this architecture works on this specific platform." The gate parameter sensitivity analysis identifying alpha_r as dominant (20x the next parameter) is a step toward addressing this, but (a) the analysis is presented as a single finding without methodology, and (b) a parameter being insensitive does not mean it was chosen correctly -- only that small perturbations around the chosen value don't change behavior much.
- **Evidence:** Line 355. Absence of systematic tuning documentation for the full parameter set.
- **Why MAJOR:** Reproducibility. A reader attempting to implement ACC on a different platform cannot replicate the tuning process from the information provided.

### MINOR

#### m1 -- High noise + delay anomaly unexplained
- **Dimension:** Logic chain validation (D4)
- **Location:** Table VI, High noise rows
- **Problem:** At high noise level: 0ms delay gives 3.00mm idle RMS with 95N F_max, 10ms delay gives 3.00mm idle RMS with 95N F_max, but 30ms delay causes immediate fall. If high noise holds the EMA permanently above threshold (anchor disengaged, P-only mode), then push performance of 95N is being delivered by the P-only baseline alone. This is notable because the P-only baseline (L0 in Table IV) achieves "only" F_med=121.8N with omnidirectional F_min=69.7N. The high-noise P-only mode achieving 95N forward push raises the question: is the P-only baseline's omnidirectional F_min (69.7N) limited by a worst-case direction rather than sagittal forward push? If so, the forward-push comparison in Table VI (F_max, forward only) is not directly comparable to the omnidirectional F_min in Table IV. This is a potential apples-to-oranges comparison that the paper should flag.
- **Evidence:** Table VI: High noise, 0ms: F_max=95N. Table IV: L0 (P-only, no I, no pos): F_min=69.7N, F_med=121.8N. The high-noise forward push result (95N) is between P-only's F_min and F_med, consistent with forward being an easier direction than the worst omnidirectional case.
- **Why MINOR:** The paper is transparent about the metrics used (F_max for robustness, F_min for ablation). The issue is that a casual reader might compare the 95N figure in the robustness table to the 69.7N figure in the ablation table and conclude ACC's P-only mode outperforms ACC's P-only baseline, which would be a comparison across different metrics.

#### m2 -- Abstract still omits push degradation under delay
- **Dimension:** Overgeneralization (D5)
- **Location:** Abstract, line 49: "degrades to 2.30mm under 10ms delay"
- **Problem:** Carried from Round 7 m1. The abstract mentions precision degradation under delay but still omits the F_max collapse from ~95N to ~45N. Given that "omnidirectional push recovery (F_min=80N, F_med=110N)" is a headline abstract claim, the omission of delay sensitivity on push performance is a material selective disclosure.
- **Evidence:** Abstract line 49--50 vs Table VI F_max data.
- **Why MINOR:** The robustness table is in the paper body. But the abstract shapes reader expectations, and omitting the most severe limitation of a headline result is a disclosure gap.

#### m3 -- Compound-disturbance 84% degradation contradicts "two independent channels" framing
- **Dimension:** Core thesis challenge (D1)
- **Location:** Section V-E, line 653: "push survival degrades 84% (94N to 15N) due to shared torque budget"
- **Problem:** The architecture is presented as having "two independent torque channels" (Section IV-A, equation 1). But during compound disturbances (push during height transition), push survival degrades 84%. The paper attributes this to "shared torque budget" -- which is exactly the point: the channels are NOT independent when torque is constrained. The 84% figure demonstrates that the decomposition into balance and posture channels does not actually decouple them under torque saturation. The paper acknowledges this (Limitation 5) but the architectural claim of "independent channels" is misleading when the independence fails exactly when it matters most (under disturbance).
- **Evidence:** Line 653: 84% degradation. Equation (1): separate wheel and leg-joint torque channels. The leg joints and wheels share the same torque budget per actuator.
- **Why MINOR:** The paper is honest about the degradation in Limitations. The point is about the architectural framing, not hidden data.

#### m4 -- "Dominant bottleneck" softened but still logically incomplete
- **Dimension:** Logic chain validation (D4)
- **Location:** Abstract line 49--50, Section V-D lines 588, 593
- **Problem:** Round 7 C2 identified that "THE bottleneck" was logically invalid because removing the servo lag doesn't prevent failure. The paper now uses "the dominant bottleneck" (lines 49, 593), which is a partial fix. But the Direct Torque result (35% improvement, 100% fall rate) still doesn't support "dominant" -- if removing the dominant bottleneck improves survival only 35% and doesn't prevent failure, either (a) the removed factor wasn't dominant, or (b) multiple bottlenecks of similar magnitude exist. The paper should say "a significant bottleneck" or "the primary identifiable bottleneck" rather than "the dominant bottleneck," and should explicitly note that Direct Torque's persistent 100% fall rate indicates additional unaddressed bottlenecks remain.
- **Evidence:** Direct Torque LQR: 0.52s -> 0.70s (35% improvement), still 100% fall rate.
- **Why MINOR:** The softening from "THE" to "dominant" is in the right direction. The remaining imprecision is a wording issue, not a data issue.

#### m5 -- Torque rate-limit invariance claim is underspecified
- **Dimension:** Overgeneralization (D5)
- **Location:** Section V-E, line 650: "ACC is invariant across 200--800 Nm/s (4x)"
- **Problem:** The claim appears without a supporting table, figure, or data. What metrics are invariant? Survival time? Idle precision? Push threshold? All of them? "Invariant" is a strong claim -- if idle RMS goes from 0.73mm to 0.80mm, that's 10% degradation, not invariance. The paper should present the data or qualify the claim with the specific metrics and tolerance band.
- **Evidence:** Line 650. Absence of rate-limit sweep data in any table.
- **Why MINOR:** Easily fixed by adding a supplementary table or qualifying "invariant" with specific metrics.

---

## Ignored Alternative Explanations/Paths

1. **ACC is best characterized as a specific anti-windup parameterization, not a new architecture.** The proximity-gated anchor (spatial gating of integral) falls within the conditional-integration AW taxonomy. The asymmetric envelope follower plus damping boost is the genuinely novel element, but it is an *implementation detail* of the integral channel, not a new architectural paradigm. The paper's own data increasingly supports this: the k_p schedule (presented as part of the "architecture") doesn't help; the pitch gate is a safety interlock; the flight and terrain modules are simple PD controllers added to independent torque channels. What remains that is both novel and effective is: (a) the asymmetric EMA for quiet-stance detection, and (b) the specific tuning of a proximity-gated integral with envelope-gated damping for this wheeled-biped platform. Both could be presented as contributions without the "unified conditionally-gated architecture" framing.

2. **The LQR baselines may be unfairly handicapped by the PID tuning, not the PID architecture.** The 0.25s servo lag figure, if due to conservative PID gains, means the LQR baselines were tested with an artificially slow inner loop. A fair comparison would tune the PID as aggressively as stability permits, then test LQR through that PID. The paper doesn't report the PID gains used or justify the 0.25s figure.

3. **PPO RL may have failed due to reward design, not because ACC is a necessary precondition.** The paper reports one PPO attempt with one reward function. Standard remedies for height-generalization failure in RL include: height-conditioned policies, curriculum learning over heights, domain randomization over dynamics parameters, asymmetric actor-critic architectures, and better reward shaping. None were attempted. The claim that ACC is a "precondition for RL" requires excluding these alternatives, which the paper does not do.

4. **The 0.73mm idle precision may be an artifact of idealized MuJoCo contact dynamics.** The paper acknowledges this in Limitations (stiff solref) and projects 1--3mm on hardware. But a well-tuned PID+I controller (not the P-only baseline) might achieve comparable hardware precision. The paper doesn't compare against a properly anti-windup-protected PI controller for idle precision -- the only integral comparisons are global I (which windups) and the conditional-integration AW (mentioned but not measured). This missing comparison means the precision advantage over *practical* alternatives is unknown.

5. **The flight and terrain modules are independent controllers, not architectural extensions.** The paper claims these demonstrate "extensibility" of the architecture. But adding a PD controller to an unused torque channel is not an architectural property -- any controller can add terms to independent torque channels. The gates "isolate each extension to its physical regime" (line 549), which is genuine, but the gates (contact detection, terrain disparity) are simple threshold checks, not novel mechanisms. The paper could present these as what they are: additional capabilities enabled by the platform's torque-control interface, not architectural innovations.

---

## Missing Stakeholder Perspectives

- **Embedded systems engineer:** The ~9.5ms latency budget with 0.5ms margin would be considered unrealistic for a first-generation prototype. Real-time Linux on an embedded ARM board (common for small robots) has scheduling jitter of 1--5ms. A realistic assessment would put hardware latency at 8--20ms, with jitter, which -- per Table VI -- would mean 45N or worse push performance and 2--40mm idle precision. The paper's "path to hardware" is optimistic.

- **Experimental roboticist with hardware experience:** Would question whether the asymmetric envelope follower can function on real velocity estimates. The envelope attacks in ~30ms -- faster than most complementary filters can converge after an impact. A hardware push creates high-frequency vibration that saturates the IMU accelerometer, producing velocity estimate spikes that would falsely trigger the fast-attack envelope. The paper acknowledges IMU noise in Table VI but models it as additive Gaussian noise, not as transient saturation from impacts.

- **Control theorist specializing in hybrid systems:** The smoothstep gates create a system that is continuous but not linear. The paper provides no stability analysis, no guarantee that the gate transitions don't create limit cycles (the partial-engagement regime IS a limit cycle), and no proof that the composite system (anchor + boost + flight hysteresis + terrain adaptation) is stable. For a paper claiming a novel control architecture, the absence of any analytical treatment of the gate interactions is a gap.

- **RL practitioner:** Would object to the characterization of PPO as having "failed." One algorithm, one hyperparameter setting, one seed, one reward function. The RL literature standard for claiming an approach "fails" is substantially higher. The paper's RL claims would not survive review by an RL-aware reviewer.

- **Reviewer from a control journal (T-RO, Automatica):** Would note that the paper neither provides stability guarantees nor benchmarks against optimization-based methods (MPC). The comparison against only classical LQR variants -- on a single simulated platform -- would be considered insufficient for establishing a general control contribution.

---

## Observations (Non-Defects)

- **Disclosure quality remains exceptionally high.** The Limitations section now has 10 items and the statistical power discussion is unusually thorough for a robotics submission. The paper's honesty about its own weaknesses is its strongest asset and should be preserved in any revision.

- **The factorial ablation design (L0--L3 additive + S0--S5 subtractive) is methodologically strong.** The two-direction approach provides internal validation: if a component added in the L* path improves metrics and its removal in the S* path degrades them, the evidence is convergent. This design is more rigorous than typical robotics ablations.

- **The gate activation figure (Figure 4) is pedagogically excellent.** Showing raw velocity, EMA envelope, individual gates, composite boost, and effective stiffness in a single aligned time series is an effective way to communicate the controller's behavior during a push-recovery cycle.

- **The EMA initialization discussion** (including the hardware recommendation of EMA_0=1.0 to prevent premature anchor engagement) shows attention to practical deployment details that many simulation-only papers neglect.

- **The gate parameter sensitivity hierarchy** (alpha_r > 20x the next parameter) is a genuine contribution. Identifying which of 50+ parameters actually matters for porting to a new platform transforms the tuning problem from unmanageable to tractable. This finding should be elevated in prominence -- it partially addresses the "50+ hand-tuned gains" concern.

- **The distinction between free-fall drop (stress test) and ledge drive-off (use case)** in the flight recovery section shows careful thinking about what constitutes a meaningful capability demonstration. This nuance is often missing.

- **The direct-torque LQR variant is the right controlled experiment** for isolating the servo path's contribution. Even though the result creates logical tension with the Zamani comparison (C2), the experiment itself is well-designed and the data is valuable. The paper should keep the experiment and resolve the tension by adjusting the interpretation, not by weakening the experiment.
