# Phase 1: Independent Reviewer Reports — ACC Paper

**Paper:** "Anchored Conditionally-Gated Control for Wheeled Bipedal Balance and Disturbance Recovery"
**Author:** Van Thuong Nguyen
**Review Date:** 2026-07-28
**Mode:** Full (7-agent panel)

---

## 1. EIC Review Report

### Reviewer Identity
Senior Editor, *IEEE Robotics and Automation Letters* — 15+ years in legged/wheeled robot control. Known for demanding clear baselines and honest limitation disclosure.

### Overall Recommendation
**Minor Revision**

### Confidence Score
**4** — High confidence. The paper falls squarely within my editorial scope (wheeled bipedal control, simulation validation, classical control architectures). I have handled similar submissions at RA-L and ICRA.

### Summary Assessment
This paper presents ACC (Anchored Conditionally-Gated Controller), a classical control architecture for wheeled bipedal balance. The core contribution is a proximity-gated integral anchor with asymmetric envelope follower that achieves sub-millimeter idle precision (0.73±0.07 mm CoM RMS) while preserving omnidirectional push survival (F_min=80N, F_med=110N). Four LQR baselines all fail at 100% fall rate. The paper is well-structured, quantitatively rigorous, and refreshingly honest about limitations. As a 6-page RA-L letter, this is a strong submission. The contribution is a genuine engineering advance — not theoretical novelty, but a clever integration of conditional gating, asymmetric filtering, and scheduled stiffness that solves a real problem (the precision-vs-recovery trade-off). The paper's greatest strength is its intellectual honesty: statistical power limitations are disclosed (N=5 yields ~4.5N MDE), simulation-only status is explicit, and 10 specific limitations are enumerated. This transparency builds reviewer trust.

### Strengths
1. **S1 — Honest limitation disclosure**: The paper enumerates 10 specific limitations (Section VI) including latency sensitivity, statistical power caveats, simulation-only status, and backward-push weakness. text: "All results are simulation-only (MuJoCo); hardware validation remains future work" — §VI, Limitation (3). This is exemplary.

2. **S2 — Comprehensive ablation design**: The factorial ablation (Table III, now Table IV in manuscript — push ablation) with additive (L0→L3) and subtractive (S0→S5) paths isolates each component's contribution. The S5 result (pitch gate removal collapses F_min to 9N) is a striking demonstration of architectural necessity.

3. **S3 — LQR baseline thoroughness**: Four variants (TWIP, AW, Direct Torque, Coupled 6-State 3D) with a 46-configuration gain sweep. The finding that the PID servo action path — not controller structure — is the bottleneck is well-supported. figure: Table VIII (LQR baselines) showing identical 0.52s survival across three variants.

4. **S4 — Clear architectural narrative**: The paper tells a coherent story: P-only limitation → anchor motivation → asymmetric EMA necessity → gate interactions → ablation validation. Each component is justified by a measured failure mode.

### Weaknesses
1. **W1 — No learning-based or MPC baseline**: The paper compares only against classical LQR. For RA-L readership in 2026, a PPO/RL baseline (even a basic one) and an MPC baseline would substantially strengthen the claim that ACC is the right architecture for this platform. The CLAUDE.md in the repository indicates RL infrastructure exists (BalanceEnv, PPO training pipeline). Even a preliminary comparison would help.
   - **Severity**: Major | **Evidence Anchor**: `absence: §V — expected RL or MPC baseline comparison; checked entire manuscript and Table VIII (LQR baselines)` | **Confidence**: 4 — familiarity with RA-L reviewer expectations

2. **W2 — "Conditionally-gated" vs. "gain-scheduled" distinction needs sharper articulation**: The paper claims conditional gating differs from gain scheduling because gates respond to "qualitatively distinct physical regimes" rather than continuous variables. But the gates themselves are continuous smoothstep functions of continuous variables (Δx, EMA(v)). The distinction is real but the argument in §IV-A needs one paragraph of sharper theoretical framing to preempt the "this is just gain scheduling" criticism.
   - **Severity**: Major | **Evidence Anchor**: `text: §IV-A "Conditionally-gated denotes continuous smoothstep activation by physical conditions...distinct from classical cascade control where nested loops run at fixed time scales"` | **Confidence**: 4 — editorial judgment on framing

3. **W3 — Feasibility demonstrations (N=3–10) in abstract risk overclaiming**: The abstract states "Flight recovery (drop 100cm, ledge 50cm) and per-leg terrain adaptation (curb 20cm) are demonstrated" without sample size. N=3 for terrain and N=4 for ledge — disclosed in the body but not the abstract. An RA-L abstract should signal the qualitative/feasibility nature of these results.
   - **Severity**: Minor | **Evidence Anchor**: `text: Abstract "Flight recovery (drop 100cm, ledge 50cm) and per-leg terrain adaptation (curb 20cm) are demonstrated as qualitative feasibility proofs"` — actually the current abstract does say "qualitative feasibility proofs." I retract this concern upon re-reading. | **Confidence**: 2

*Revised W3*: The abstract appropriately labels these as "qualitative feasibility proofs" — no issue.

### Detailed Comments

**Journal Fit**: Excellent for RA-L. The paper is concise (~6 pages IEEEtran), presents a clear single contribution (proximity-gated anchor), validates with rigorous ablations, and honestly discloses limitations. The classical-control focus may surprise some RA-L readers accustomed to learning-based methods, but the journal explicitly welcomes "novel control architectures" and the wheeled-biped topic is timely.

**Originality**: The combination of (a) proximity-gated integral, (b) asymmetric envelope follower for quiet-stance detection, (c) scheduled pitch stiffness, and (d) gated damping boost is genuinely novel as an integrated architecture. Individual components (conditional integration, envelope detection) have precedents, but the specific synthesis and the empirical demonstration that standard AW is insufficient are original contributions.

**Significance**: Moderate-to-high. The precision-vs-recovery trade-off is a real problem for wheeled bipeds. The finding that four LQR variants all fail structurally is valuable negative evidence for the community. The porting guide (gate parameter sensitivity, 5-step tuning procedure) increases practical impact.

**Structural Coherence**: Strong. Title → Abstract → Introduction → Method → Results → Discussion flows logically. The architecture figure (Fig. 2) is clear.

**Title & Abstract**: Title is descriptive and specific. Abstract contains quantitative claims (0.73mm RMS, 80N F_min) which is excellent. Minor note: "degrades to 2.30mm under 10ms delay" in the abstract could be misinterpreted as a weakness rather than the remarkable finding that it still works.

### Questions for Authors
1. Why no RL or MPC baseline? The repository appears to have PPO infrastructure. Even a modest comparison (e.g., 1M-step PPO at fixed height) would address the most likely reviewer objection.
2. Have you considered submitting the LQR-baseline-comparison as a separate short paper (e.g., "Why Classical LQR Fails on Wheeled Bipeds: A Structural Analysis")? The finding that the PID servo path is the bottleneck is independently publishable.

### Minor Issues
- L91: "blind stair climbing" → "blind stair climbing" (typo — should be "stair")
- Actually re-reading: it says "stair climbing" which is correct. No issue.
- L189: $k_p$ schedule described as gating on "proximity and quietness (envelope), not on height" — good, but this distinction could be made more prominent as it's key to the vs-gain-scheduling argument.

---

## 2. Methodology Review Report (Peer Reviewer 1)

### Reviewer Identity
Control theorist — expertise in nonlinear control, anti-windup design, disturbance rejection. 10+ years in *Automatica*, *IEEE TAC*, *Control Engineering Practice*. Deep knowledge of the AW taxonomy.

### Overall Recommendation
**Minor Revision**

### Confidence Score
**4** — High confidence. The experimental design, statistical methods, and control-theoretic claims fall within my core expertise. I have published on anti-windup taxonomies and can evaluate the AW-related claims with authority.

### Summary Assessment
This paper uses a factorial ablation design with binary search for threshold estimation to evaluate a classical control architecture. The experimental methodology is generally sound for a simulation study: deterministic RNG seeding, controlled ablation paths (additive + subtractive), and explicit statistical power disclosure (MDE ~4.5N at N=5). The paper's strongest methodological contribution is the honest statistical reporting — Clopper-Pearson CIs for binary outcomes, between-trial CV reporting, and explicit power caveats. However, there are methodological concerns: (1) the claim that "standard AW is insufficient" is evaluated against only ONE anti-windup variant (conditional integration) while the taxonomy includes tracking back-calculation and observer-based AW; (2) the LQR baseline tuning procedure is not fully described — the 46-config sweep addresses this partially but the Q/R matrix selection process needs documentation; (3) N=3 for terrain and N=4 for ledge are below any meaningful statistical threshold. These are addressable in revision.

### Strengths
1. **S1 — Factorial ablation with two-direction validation**: The additive (L0→L3) and subtractive (S0→S5) paths provide internal replication. The S5 finding (pitch gate removal → F_min=9N) is a compelling single-component ablation. table: Table IV (push ablation) showing L0–L3 additive and S0–S5 subtractive rows.

2. **S2 — Explicit statistical power disclosure**: The paper states "minimum detectable effect ~4.5N at N=5" and notes sub-2N differences are non-significant. This is rare and commendable in robotics simulation papers. text: §VI "With N=5, the minimum detectable effect is ~1.5× the within-cell std (~4.5N). Differences <5N cannot be distinguished from noise."

3. **S3 — Binary search for threshold estimation**: Using binary search over [10N, 160N] with 8 iterations and 5N tolerance for F_min/F_med is methodologically appropriate. text: §V-B "measured via binary search over [10N, 160N] (8 iterations per direction, 5N tolerance)"

4. **S4 — Reproducibility commitment**: Code, configs, seeds, and parameters are version-controlled and referenced by commit hash. text: Acknowledgment "Code and parameters: https://github.com/... (commit f92640c)"

### Weaknesses
1. **W1 — AW baseline tests only ONE variant from the taxonomy**: The paper claims "standard AW is insufficient" (Appendix A) but evaluates only conditional integration with saturation-based freezing. The AW taxonomy (Åström & Hägglund, 2006; Hippe, 2006; Tarbouriech et al., 2011) includes tracking back-calculation (feeding saturation error back to the integrator) and observer-based AW (estimating unsaturated plant state). Appendix A acknowledges this gap but the abstract and introduction's blanket statements about "standard AW" are therefore overbroad. The claim should be narrowed to "saturation-triggered conditional integration AW" or a second AW variant should be tested.
   - **Severity**: Major | **Evidence Anchor**: `text: Appendix A "Beyond saturation-triggered schemes, the AW taxonomy includes conditional integration, tracking back-calculation, and observer-based AW...A systematic comparison across the full AW taxonomy is deferred to future work"` | **Confidence**: 5 — this is my primary research area

2. **W2 — LQR Q/R matrix selection rationale not documented**: The 4-state LQR and 6-state LQR use specific Q/R matrices. The 46-config gain sweep addresses whether the failure is parametric, but does not replace a clear description of how the nominal Q/R was selected (Bryson's rule? Pole placement? Iterative tuning?). Without this, a reviewer cannot assess whether the LQR baseline is fairly tuned.
   - **Severity**: Major | **Evidence Anchor**: `absence: §V-D — expected Q/R selection methodology for LQR baselines; checked §V-D and Table VIII caption` | **Confidence**: 4 — control design methodology expertise

3. **W3 — Robustness sweep methodology needs clarification**: Table VI (robustness) shows a puzzling pattern: "High noise, 0ms delay" shows RMS=3.00mm and F_max=95N — better than "Med noise, 0ms delay" (34.00mm, 94N). This non-monotonicity is not explained. If the high-noise condition causes the envelope follower to stay disengaged (anchor off → P-only behavior), this should be explicitly analyzed, not left for the reader to infer.
   - **Severity**: Minor | **Evidence Anchor**: `table: Table VI rows "Med noise, 0ms delay" (34.00mm idle) vs "High noise, 0ms delay" (3.00mm idle)` | **Confidence**: 3 — statistical anomaly detection

4. **W4 — No correction for multiple comparisons in ablation**: The ablation reports p-values (p<0.001 for L0→L1, p<0.05 for L2→L3) across multiple pairwise tests. With 7 additive+subtractive comparisons, a Bonferroni or Holm correction should at minimum be discussed. At α=0.05/7≈0.007, the L2→L3 comparison (p<0.05) would not survive correction.
   - **Severity**: Minor | **Evidence Anchor**: `text: Table IV notes "L0→L1 +5.5N (p<0.001); L1→L2 +1.1N (n.s.); L2→L3 +4.1N (p<0.05)"` | **Confidence**: 4 — statistical methodology expertise

### Detailed Comments

**Research Design**: The factorial ablation with within-cell replication (N=5) is appropriate for a simulation study. The two-direction design (additive + subtractive) provides internal replication. The decision to report both F_min (worst-case across 8 directions) and F_med (median) is sound.

**Sampling Strategy**: Deterministic seeding with documented seed formulas enables exact reproduction. The 3-seed cycling for LQR baselines is minimal but acceptable given the consistent 100% fall rate.

**Analysis Methods**: Binary search for threshold estimation is correct. Between-trial CV (14.3%) is appropriately reported. The Clopper-Pearson CI for binary outcomes (N=3 terrain: [0.29,1.0]) is correctly computed and honestly wide.

**Reproducibility**: Excellent. Commit hash, config files, and scripts are referenced. The AW validation script is explicitly mentioned. This exceeds typical robotics simulation paper standards.

### Questions for Authors
1. Can you test ONE additional AW variant — specifically tracking back-calculation (the most common industrial AW scheme)? This would substantially strengthen the "standard AW is insufficient" claim. If not feasible, please narrow the AW-related claims in the abstract and introduction to "saturation-triggered conditional integration AW."
2. How were the nominal LQR Q/R matrices selected? Please add a sentence or footnote describing the design rationale (e.g., Bryson's rule, LQR pole placement, or iterative tuning).
3. Can you explain the non-monotonic noise response (high noise outperforming medium noise at 0ms delay)? This seems important for understanding the envelope follower's failure mode.

---

## 3. Domain Review Report (Peer Reviewer 2)

### Reviewer Identity
Wheeled bipedal robotics researcher — 8+ years building and controlling wheeled-legged platforms. Published on Ascento, TWIP, and related platforms at ICRA/IROS/RA-L. Hands-on MuJoCo experience.

### Overall Recommendation
**Minor Revision**

### Confidence Score
**5** — Completely within my area of expertise. I have built and simulated wheeled bipeds with similar morphology, worked with MuJoCo contact models, and dealt with the exact precision-vs-recovery trade-off this paper addresses.

### Summary Assessment
This is a practically valuable paper from someone who clearly understands the real engineering challenges of wheeled bipedal balancing. The precision-vs-recovery trade-off (kp=50 gives precision but kills push survival; kp=35 enables recovery but permits 55mm limit cycle) is a genuine, well-characterized problem that anyone who has tuned a TWIP controller will recognize. The ACC architecture is a clever integration of known ideas (conditional integration, envelope detection, gain scheduling) into a specific configuration that works remarkably well. The paper's greatest value is as an engineering contribution: a documented, reproducible controller architecture with clear tuning guidelines. The robot model (8.1kg, 0.26m thigh, 0.28m shin, 0.06m wheel radius) is physically plausible. The MuJoCo parameters (Newton solver, pyramidal cone, armature, solref) are described in unusual detail — this is commendable for reproducibility. I have several concerns about literature coverage and the physical realism of some claims, but these are addressable.

### Strengths
1. **S1 — Detailed robot and simulation specification**: The paper provides mass, link lengths, wheel radius, torque limits, MuJoCo solver settings, contact parameters, and armature values. This is unusually thorough and enables reproduction. text: §III "Mass: 8.1kg; thigh: 0.26m; shin: 0.28m; wheel radius: 0.06m; hip width: 0.23m. Torque limits: 60Nm (HR, HY, WH), 150Nm (HP, KN)."

2. **S2 — Platform comparison table is informative and honest**: Table I (platform comparison) uses "—" for missing capabilities rather than claiming superiority. The comparison against Ascento, DIABLO, Whleaper, and SKATER is fair and appropriately scoped. table: Table I — Capability comparison across wheeled biped platforms.

3. **S3 — Practical tuning guide for porting ACC**: The 5-step procedure (identify ringdown period → set τ_release → set quiet band → set pitch gates → tune kp) in §V-E shows the author has thought about transferability. This is rare and valuable. text: §V-E "For porting ACC to a new morphology: (1) identify the dominant ringdown period (Tr)...(5) tune kp (stiff) for desired idle precision, then soften by ~30%"

4. **S4 — Gate parameter sensitivity analysis**: The finite-difference gradient analysis revealing α_r dominance (220.4 sensitivity, 20× the next parameter) is genuinely useful for practitioners. text: §V-E "release coefficient α_r dominates (sensitivity 220.4, 20× the next, 130× pitch gates)"

### Weaknesses
1. **W1 — Missing key reference: Zamani et al. (2020) coupled 3D LQR context**: The paper cites Zamani et al. [zamani2020coupled] for "A Coupled 3D LQR for Two-Wheeled Balancing Robots" but this reference appears to be cited primarily to establish that coupled LQR has been studied, not deeply engaged with. The 6-state coupled LQR baseline in this paper is a direct extension of that work — the relationship should be explicitly discussed. What did Zamani et al. find? Did their coupled LQR work on their platform? If so, why does it fail here?
   - **Severity**: Major | **Evidence Anchor**: `text: §II-A "A coupled 3D LQR for similar morphology was studied by Zamani et al. [34]" and §V-D 6-State 3D LQR baseline` | **Confidence**: 4 — domain expertise in wheeled biped literature

2. **W2 — Ground contact model realism**: The paper uses MuJoCo with pyramidal cone friction for body contacts and elliptic for wheels. For a paper making sub-millimeter precision claims (0.73mm CoM RMS), the contact model stiffness (solref={0.002,1}) should be discussed. In real hardware, tire compliance alone contributes 0.1–0.5mm (as the paper acknowledges in §VI). But the simulation contact stiffness may be unrealistically high, making the 0.73mm figure optimistic even for simulation. A brief sensitivity analysis (varying solref) would strengthen credibility.
   - **Severity**: Minor | **Evidence Anchor**: `text: §III "contact stiffness/damping: solref={0.002,1} (body), {0.005,0.7} (wheels)"` | **Confidence**: 3 — MuJoCo contact modeling experience

3. **W3 — Literature on envelope/peak-detection for control mode switching is thin**: The asymmetric EMA (fast attack, slow release) is presented as novel for quiet-stance detection, but the signal processing literature on envelope followers for control mode switching (particularly in audio compressor/expander design and in fault detection) is not cited. The concept is not new — the application to wheeled biped balance is. This should be acknowledged.
   - **Severity**: Minor | **Evidence Anchor**: `absence: §IV-C — expected citations to envelope detection / signal processing literature for control applications; checked references [1–35]` | **Confidence**: 3 — cross-domain awareness [FIELD-NORM UNVERIFIED]

4. **W4 — Handle/Swiss-Mile comparison could be deeper**: The related work mentions Handle and Swiss-Mile but doesn't engage with their control architectures. Handle used a different morphology (two wheels, no legs) but its balancing controller likely addressed similar precision-vs-disturbance trade-offs. Swiss-Mile's whole-body controller for wheeled quadrupeds might offer relevant comparisons.
   - **Severity**: Minor | **Evidence Anchor**: `text: §II-A "Handle [2] and handleDRIVE [21] pioneered wheeled-leg coordination; Swiss-Mile [22] combined driving with quadrupedal walking"` | **Confidence**: 3 — literature breadth

### Detailed Comments

**Literature Coverage**: The paper cites 35 references spanning classical control (Åström & Hägglund, Rugh & Shamma), wheeled bipeds (Ascento, DIABLO, Whleaper, SKATER, BITeno), disturbance rejection (DOB, ADRC), and flight/terrain control. The coverage of wheeled biped platforms is comprehensive. Missing: deeper engagement with Handle's control architecture, and the envelope-detection signal processing literature.

**Theoretical Framework**: ACC is presented as an architectural contribution, not a theoretical one. This is appropriate — the paper does not claim new control theory. The framework is "conditional gating of control components by physical conditions." This is coherent and consistently applied.

**Contribution to the Field**: The primary contribution is a well-characterized, reproducible controller architecture that solves a real engineering problem (precision-vs-recovery trade-off). The secondary contribution is the negative result that four LQR variants fail structurally on this platform — this is valuable for the community. The contribution is incremental (not breakthrough) but genuine and well-documented.

**Physical Realism**: The 8.1kg mass, link lengths, and torque limits are plausible for a research-platform wheeled biped. The 400 Nm/s torque rate limit is realistic for commercial actuators. The 100Hz control rate is standard. The MuJoCo parameters are described in unusual detail — this is a strength. The one concern is contact stiffness (see W2).

**"Path to Hardware" Assessment**: The hardware degradation analysis (0.73mm → 1–3mm) is honest and well-reasoned. The staged commissioning plan (static torque → fixed-support → free standing → push recovery → flight → terrain) is sensible. The control computation budget (~4.5ms of 10ms window) leaves adequate margin. I would like to see one additional detail: what IMU is assumed for the 0.01 rad/s noise floor in the "low noise" condition? Tying noise levels to specific sensor models would strengthen the realism argument.

### Missing Key References
- [UNVERIFIED] Literature on envelope detection for control mode switching in audio dynamics processing (compressor/expander design) — relevant to the asymmetric EMA concept
- [UNVERIFIED] Handle's balancing controller publications (if any exist beyond the YouTube video) — for deeper architectural comparison

---

## 4. Perspective Review Report (Peer Reviewer 3)

### Reviewer Identity
Researcher bridging classical control and modern robotics — publications in *IEEE Control Systems Magazine* and *Science Robotics*. Skeptical of "new architecture" claims without clear theoretical grounding. Expertise in translating control-theoretic concepts to physical robots.

### Overall Recommendation
**Major Revision**

### Confidence Score
**4** — High confidence. My career bridges the exact gap this paper occupies (classical control → physical robots). I can evaluate both the control-theoretic framing and the robotics validation.

### Summary Assessment
This paper sits at an uncomfortable intersection. As a robotics contribution, it is strong: a well-tuned controller that demonstrably outperforms LQR baselines on a specific platform. As a control-theoretic contribution, it is weak: the individual components (conditional integration, envelope detection, gain scheduling) are known techniques, and the paper does not provide theoretical guarantees (stability proofs, convergence rates, robustness certificates). The question is whether the integration itself constitutes a sufficient contribution. I lean toward "yes, but the paper overclaims the novelty of the architecture." The term "conditionally-gated control" implies a new control paradigm, but what's presented is a specific configuration of known techniques for a specific platform. The paper would be stronger if it were more modest in its architectural claims and stronger in its empirical contribution. I also see missed connections to process control (split-range control, override control, selector-based control) and aerospace (flight mode switching, gain scheduling with hysteresis) that would enrich the related work and strengthen the paper's intellectual lineage.

### Strengths
1. **S1 — The asymmetric EMA is genuinely elegant**: Using different attack/release time constants to distinguish quiet stance from ringdown is a clever application of a known signal processing concept to a control problem. The bistable-failure demonstration (symmetric EMA ablation) proves this is not a trivial choice. text: §IV-C "a symmetric EMA is bistable — post-push oscillation holds the EMA in a middle band where the gate is half-open"

2. **S2 — The LQR comparison is the paper's strongest empirical contribution**: Four variants, all fail, structural diagnosis (PID servo path is the bottleneck) confirmed by gain sweep. This negative result is more convincing than the positive ACC results because it demonstrates the problem is real. table: Table VIII — Four LQR variants vs ACC, all LQR rows showing 0.52–0.70s survival.

3. **S3 — Practical transferability**: The 5-step porting guide, the parameter sensitivity analysis (α_r dominates), and the hardware degradation analysis all demonstrate genuine engineering maturity. This is what distinguishes a paper from someone who actually built and tested vs. someone who ran simulations.

### Weaknesses
1. **W1 — "Conditionally-gated control" is not a new paradigm — it's a specific application of override/selector-based control**: The process control literature has used selector-based control (where multiple controllers compete and a selector chooses the active one based on process conditions) since the 1970s. Override control, split-range control, and valve position control are standard techniques in chemical process control (Shinskey, 1996; Åström & Hägglund, 2005, Chapter 10). The ACC gates are continuous (smoothstep) rather than discrete selectors, but the conceptual framework is the same: condition → activate/deactivate control component. The paper should cite this lineage and position ACC as a continuous (smoothstep) extension of selector-based control for robotics, rather than implying a new paradigm.
   - **Severity**: Major | **Evidence Anchor**: `absence: §II and §IV — expected citations to process control literature on override/selector-based/split-range control; checked references [1–35]` | **Confidence**: 4 — control systems breadth

2. **W2 — No stability analysis or theoretical guarantees**: The paper provides zero theoretical analysis — no Lyapunov stability proof, no passivity-based argument, no describing function analysis of the limit cycle, no contraction analysis. For a paper claiming a novel control architecture, some theoretical characterization is expected. Even a simple analysis: (a) with all gates=0, the system reduces to PD balance → can we prove stability of the PD-only system? (b) with gates=1, the integral+boost is active → can we prove the equilibrium is asymptotically stable? Without this, the paper is purely empirical.
   - **Severity**: Major | **Evidence Anchor**: `absence: §IV — expected stability analysis or theoretical guarantees; checked entire manuscript` | **Confidence**: 3 — I acknowledge this may be a disciplinary expectation mismatch (control theory vs. robotics)[FIELD-NORM UNVERIFIED]

3. **W3 — Missed connection to aerospace flight mode switching**: The flight controller (§IV-E) uses discrete hysteresis (2-step engage, 5-step release) for contact-loss detection. This is essentially a finite-state machine with hysteresis — a technique extensively studied in aerospace for flight control mode transitions (e.g., transitioning from air-to-air refueling to normal flight). The aerospace literature on "gain scheduling with hysteresis," "bumpless transfer," and "mode transition logic" is directly relevant and uncited.
   - **Severity**: Minor | **Evidence Anchor**: `absence: §II and §IV-E — expected citations to aerospace mode-switching and bumpless transfer literature; checked references [1–35]` | **Confidence**: 3 — cross-domain awareness

4. **W4 — The paper's title promises more than the paper delivers**: "Anchored Conditionally-Gated Control for Wheeled Bipedal Balance and Disturbance Recovery" implies a general control framework. The paper delivers a specific controller for a specific robot. The contribution is real but more modest than the title suggests. Consider "Proximity-Gated Integral Control with Asymmetric Envelope Detection for Precision Standing of a Wheeled Biped."
   - **Severity**: Minor | **Evidence Anchor**: `text: Title "Anchored Conditionally-Gated Control for Wheeled Bipedal Balance and Disturbance Recovery"` | **Confidence**: 4 — framing and positioning judgment

### Detailed Comments

**Assumption Audit**:
- *Explicit assumption*: The PID servo layer (~0.25s lag) is the bottleneck for LQR. This is well-supported by the Direct Torque comparison.
- *Implicit assumption*: MuJoCo simulation fidelity is sufficient to draw conclusions about real-world controller performance. This is partially addressed in the hardware degradation analysis but remains an assumption.
- *Paradigmatic assumption*: The paper operates in the classical control paradigm (PID + LQR + gain scheduling). This is appropriate for the contribution but limits the analysis toolkit (no learning, no optimization, no formal verification).

**Cross-Disciplinary Connections**:
- *Process control*: Override control, split-range control, selector-based control — see W1.
- *Aerospace*: Flight mode switching, bumpless transfer — see W3.
- *Audio signal processing*: Envelope followers for dynamics processing — not cited but the concept is directly analogous.
- *Power electronics*: Hysteretic control, burst-mode control for light-load efficiency — conceptually similar to gated control activation.

**Practical Impact**: The paper's practical impact is high for experimentalists building wheeled bipeds. The controller is simple, the tuning procedure is documented, and the hardware path is clearly laid out. For the broader robotics community, the paper's value is the demonstration that careful classical control engineering can outperform naive LQR — a useful corrective to the current "RL/MPC everywhere" trend.

### Cross-Disciplinary Reading Recommendations
- [UNVERIFIED] Shinskey, F.G. "Process Control Systems" — for override/selector-based control
- [UNVERIFIED] Åström & Hägglund, "Advanced PID Control" Chapter 10 — for selector and split-range control
- [UNVERIFIED] Aerospace literature on bumpless transfer and mode transition logic — search for "gain scheduling hysteresis aerospace"

---

## 5. Devil's Advocate Review

### Strongest Counter-Argument

The paper's central claim is that ACC — a combination of proximity-gated integral, asymmetric envelope follower, scheduled stiffness, and gated boost — constitutes a novel control architecture that solves the precision-vs-recovery trade-off. The strongest counter-argument is: **this is a well-engineered but conceptually incremental integration of known techniques, and the paper does not demonstrate that ACC outperforms a competent alternative that already exists in the literature.**

Specifically: a disturbance observer (DOB) combined with an LQR balancing controller could theoretically achieve the same outcome — estimate the equilibrium bias online (replacing the anchor integral), compensate for it in the torque command (enabling precision), and attenuate disturbances through the DOB's low-pass filter (preserving recovery). The paper dismisses DOB in §II-B with the claim that DOB "cannot distinguish a push from post-push ringdown — both occupy the same frequency band." But this argument is incomplete: a DOB with a sufficiently low cutoff frequency would not respond to the 2–3 Hz ringdown oscillation, while still compensating the DC equilibrium bias. The paper does not implement or test a DOB baseline. Without this comparison, the claim that ACC's asymmetric envelope is the uniquely correct solution is unsubstantiated.

Similarly, a model-predictive controller (MPC) with the same 10ms control period could optimize the precision-vs-recovery trade-off explicitly through its cost function, without needing five hand-tuned gates. The paper's silence on MPC — despite SKATER [skater2024] demonstrating convex MPC for the same morphology class — is a significant omission.

The four LQR baselines that all "fail" are all variants of the same underlying controller structure (PD servo with different integral handling). Demonstrating that four variants of the same approach fail does not prove that the approach class is insufficient — it proves that those four specific configurations fail. A competent DOB+LQR or MPC baseline might succeed, and without testing one, the paper's architectural claims remain unvalidated against the most obvious alternatives.

### Issue List

#### CRITICAL

| # | Dimension | Issue Description | Evidence Anchor | Confidence | Field-Norm Boundary | Evidence-Crossing Rationale |
|---|-----------|-------------------|-----------------|------------|---------------------|-----------------------------|
| C1 | Alternative Paths (D6) | No DOB, ADRC, or MPC baseline tested despite these being the most obvious alternatives to gated integral control. The DOB dismissal in §II-B is theoretical, not empirical. The SKATER reference demonstrates MPC for the same morphology class. | absence: §V — expected DOB/ADRC/MPC baseline comparison; checked Tables IV, VIII and all of §V | 4 — control architecture breadth | A robotics control paper at RA-L level is expected to compare against at least ONE learning-based or optimization-based baseline; see RA-L author guidelines and recent accepted papers (e.g., SKATER [skater2024] includes MPC comparisons). | This paper claims architectural novelty of a classical controller but only compares against other classical controllers. A DOB+LQR or MPC baseline would test whether the gated-integral architecture is genuinely necessary or merely one valid approach among several. |

| C2 | Overgeneralization (D5) | The paper claims "four classical LQR baselines all fail (100% fall rate)" as evidence that ACC's architecture is necessary. But the LQR baselines all share the same PID servo action path. The paper's own Direct Torque variant improves survival by 35%, suggesting the PID servo — not LQR — is the bottleneck. The correct conclusion is "LQR through a PID servo fails," not "LQR fails." The paper acknowledges this in §V-D but the abstract and conclusion generalize to "LQR baselines." | text: Abstract "Four classical LQR baselines — PID-servo, integral-anti-windup, direct-torque, and coupled 6-state 3D — all fail (100% fall rate)" vs §V-D "the PID servo action path — not controller structure — is the bottleneck" | 5 — logical consistency check | | |

| C3 | "So What?" Test (D8) | The paper demonstrates a controller that works in simulation on one robot. The impact beyond this specific platform is unclear. The 5-step porting guide suggests generality, but without testing on a second platform or simulator, the claim of architectural generality is unsupported. The paper's own limitation (8) acknowledges this: "Architectural generality across morphologies and simulators remains to be demonstrated." | text: Limitation (8) "Architectural generality across morphologies and simulators remains to be demonstrated" | 4 — contribution scope assessment | | |

#### MAJOR

| # | Dimension | Issue Description | Evidence Anchor | Confidence | Field-Norm Boundary | Evidence-Crossing Rationale |
|---|-----------|-------------------|-----------------|------------|---------------------|-----------------------------|
| M1 | Core Thesis (D1) | The paper claims ACC "achieves all demonstrated behaviors without neural network training" (Conclusion). But the behaviors demonstrated are: standing still, recovering from pushes, falling and landing, and driving over a curb. A PPO policy trained on the same platform could likely achieve similar results. The repository CLAUDE.md indicates BalanceEnv PPO infrastructure exists. The paper does not justify why classical control is preferable to RL for this application, beyond stating a hypothesis about sim-to-real gap. | absence: §I and §VI — expected justification for classical vs learned approach; checked Introduction and Discussion/Conclusion | 3 — RL expertise is outside my core competence | | |

| M2 | Logic Chain (D4) | The asymmetric EMA is presented as essential because a symmetric EMA is "bistable" (§IV-C). But the bistability claim depends on the specific symmetric EMA parameters. A symmetric EMA with a lower cutoff frequency would not exhibit bistability — it would simply be slower to respond to the ringdown decay. The paper demonstrates that ONE symmetric EMA configuration fails, not that ALL symmetric EMAs fail. This is a single-point ablation, not a proof of necessity. | text: §IV-C "a symmetric EMA is bistable — post-push oscillation holds the EMA in a middle band where the gate is half-open and the boost cannot collapse the cycle" | 4 — signal processing logic | | |

| M3 | Confirmation Bias (D3) | The ablation design is structured to confirm ACC's value: additive path starts from P-only and builds toward ACC; subtractive path starts from ACC and removes components. Both paths lead to the same conclusion (ACC is best). While this is good experimental design, the choice of ablation ORDER may introduce confirmation bias: in the subtractive path, removing the pitch gate LAST (S5) produces the most dramatic failure (F_min=9N), creating a narrative climax. A different removal order might produce different insights. | table: Table IV subtractive rows S1→S5 ordered as: -kp schedule, -boost, -asym EMA, -prox gate, -pitch gate | 3 — experimental design assessment | | |

| M4 | Cherry-Picking (D2) | The platform comparison table (Table I) selectively chooses metrics where ACC excels: idle precision and push recovery are where ACC is designed to win. It omits metrics where other platforms might lead: locomotion speed, energy efficiency, stair climbing success rate, maximum terrain height. This creates an impression of comprehensive superiority that the evidence does not support. | table: Table I — Capability comparison, metrics include: Balance, Idle precision, Push recovery, Height, Flight, Terrain | 4 — comparison fairness assessment | | |

### Ignored Alternative Explanations/Paths
1. **DOB + LQR baseline**: A disturbance observer with low-pass filter (cutoff ~0.1 Hz) could estimate and cancel the equilibrium bias without responding to 2-3 Hz ringdown, potentially achieving both precision and recovery with a simpler architecture.
2. **MPC with the same 10ms control period**: SKATER demonstrated convex MPC for wheeled bipedal balancing. An MPC with state constraints could explicitly enforce precision near equilibrium while allowing larger excursions during pushes.
3. **Simple dead-zone + leaky integral**: A dead-zone on position error (|Δx| > 2cm → integral frozen, < 2cm → integral active) with a faster leak (×10 current) might achieve similar results with far fewer parameters. The paper claims dead-zones "over-restrict or under-restrict" but does not test a tuned dead-zone variant.

### Missing Stakeholder Perspectives
- Hardware experimentalists who need to implement this on real robots — the paper's hardware path is detailed but untested
- RL practitioners who might argue the same results are achievable with 1M steps of PPO
- Industrial users who need controllers that work reliably without per-platform tuning — ACC requires 50+ hand-tuned parameters

### Observations (Non-Defects)
- The paper's intellectual honesty (10 explicit limitations, statistical power disclosure) is genuinely exceptional and should be recognized as a strength
- The gate parameter sensitivity analysis showing α_r dominance is a valuable practical insight that many papers would omit
- The paper would benefit from a clearer statement that its contribution is *empirical* (a well-characterized solution to a specific problem) rather than *theoretical* (a new control paradigm)
