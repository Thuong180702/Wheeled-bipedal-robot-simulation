Now I have all three source documents. Let me analyze the paper and produce the complete Reviewer Configuration Card.

---

# Field Analysis Report

## Paper Basic Information
- **Title**: Anchored Conditionally-Gated Control for Wheeled Bipedal Balance and Disturbance Recovery
- **Author**: Van Thuong Nguyen (Independent Researcher)
- **Abstract length**: ~180 words
- **Full text length**: ~7,500 words (IEEEtran conference format, 2-column, ~12 pages equivalent)
- **Number of references**: 40
- **Document class**: IEEEtran (conference)

## Field Analysis

| Dimension | Analysis Result |
|-----------|----------------|
| Primary Discipline | Robotics and Control Systems Engineering |
| Secondary Disciplines | Classical Control Theory (anti-windup, gating, LQR), Mechatronics (actuator modeling, torque control), Reinforcement Learning (motivation only -- PPO baseline) |
| Research Paradigm | Quantitative Research |
| Methodology Type | Experimental / Quasi-experimental (simulation-based comparative evaluation with factorial ablation, binary-search threshold measurement, controlled baseline comparisons) |
| Target Journal Tier | Q2 -- IEEE Robotics and Automation Letters (IF ~5-7) or IEEE/ASME Trans. Mechatronics (IF ~5-7). The paper is well-structured with thorough ablation and honest limitations, but is simulation-only (no hardware), has low N for some tests (N=3-5), and explicitly defers hardware validation. The IEEEtran conference format suggests ICRA/IROS as a conference target. |
| Paper Maturity | Pre-submission -- Complete structure (Introduction through Conclusion + Appendix), well-formatted references, exhaustive ablation tables, figures in place, limitations explicitly stated, statistical power analysis included. The paper self-identifies remaining gaps (N<10 for terrain/flight, recommends increased sample sizes before submission, hardware validation deferred). Quality is near submission-ready for a conference; for a journal would need expanded N and ideally hardware results. |

## Recommended Target Journals (Top 3)
1. **IEEE Robotics and Automation Letters (RA-L)** -- Primary target. Publishes rapid-turnaround letters on novel control methods with experimental validation. Accepts simulation-only with strong comparative baselines. The paper's length and format are RA-L appropriate. The SKATER paper and the Klemm LQR+WBC paper were both published in RA-L.
2. **IEEE/ASME Transactions on Mechatronics** -- Published the SKATER paper (2025) and foundational mechatronics control papers. Values integrated mechanical+control contributions. The paper's detailed actuator modeling (armature, rate limits, contact parameters) and hardware degradation analysis fit the mechatronics scope.
3. **IEEE International Conference on Robotics and Automation (ICRA)** -- The IEEEtran format directly targets this venue. ICRA accepts simulation-only with strong comparative evaluation. The paper's ablation depth exceeds typical ICRA standards; conference page limits may require compression.

---

## Reviewer Configuration Cards

### Reviewer Configuration Card #1 -- EIC

**Role**: Editor-in-Chief / Senior Editor
**Identity Description**: Senior Editor of *IEEE Robotics and Automation Letters*, specializing in novel control architectures for legged and wheeled mobile robots. Formerly led the Dynamic Locomotion group at a European robotics institute; has handled review cycles for Ascento, ANYmal, and SKATER submissions. Values experimental rigor, reproducible benchmarks, and architectural contributions that generalize beyond a single platform.

**Review Focus**:
1. Does this paper fit RA-L's scope of "reporting results available for immediate use by the robotics community," or is the method too tightly coupled to one specific morphology?
2. Is the claimed novelty -- the proximity-gated anchor with asymmetric envelope follower -- sufficiently distinct from existing conditional-integration and override-control techniques to warrant publication in a top robotics venue?
3. Are the comparative baselines fair and representative? The 100% LQR failure rate is a strong claim -- does the evidence support attributing this to the PID servo action path rather than to inadequate gain tuning of the LQR baselines?

**Will particularly care about**: Whether the paper demonstrates generality or merely shows careful parameter tuning on one platform. The architectural claims ("two-channel torque assembly," "conditionally-gated" as a framework) need to be supported by evidence that the gates *compose* without destructive interference, not just that they each work in isolation.

**Possible blind spots**: May undervalue the classical control theory lineage (anti-windup taxonomy, override control) in favor of robotics-specific novelty. The paper's connection to the process control literature may be underappreciated. Also, as an editor accustomed to hardware-validated submissions, may discount the thoroughness of the simulation study because of the hardware-deferred status.

---

### Reviewer Configuration Card #2 -- Peer Reviewer 1 (Methodology)

**Role**: Peer Reviewer 1 -- Methodology
**Identity Description**: Researcher in experimental robotics evaluation methodology, specializing in systematic ablation design and statistical validation for classical and learned controllers. Has published on benchmark design for legged robot evaluation and on statistical power analysis for robotics experiments. Background in control theory with a focus on reproducible experimental protocols.

**Review Focus**:
1. Factorial ablation design: Does the additive (L0-L3) and subtractive (S0-S5) ablation correctly isolate each component's causal contribution? The marginal effects are small (L1 adds +5.5N) -- are the measurement protocols (binary search at 5N tolerance, N=5) sufficiently powered to distinguish real effects from noise?
2. Statistical validity: The paper honestly reports N=5 power limitations (minimum detectable effect ~4.5N). But are the reported p-values (L0-L1 p<0.001) computed correctly given the binary-search measurement process? Does the Bonferroni correction discussion adequately address the 7-comparison problem?
3. LQR baseline fairness: The 46-configuration gain sweep is cited as evidence that LQR failure is structural, not parametric. Were the sweeps systematic and documented? Is the "direct torque" variant a fair comparison given it uses the same LQR gains derived for a PID-servo architecture?

**Will particularly care about**: Whether the experimental protocol is reproducible from the paper alone. The paper references scripts (`scripts/sweep_lqr_roll_gains.py`, `scripts/validate_lqr_aw.py`) and a GitHub repository -- is the evaluation pipeline documented sufficiently for independent reproduction? Does the binary-search protocol for push thresholds have well-defined convergence criteria?

**Possible blind spots**: May focus heavily on statistical rigor at the expense of appreciating the architectural insight. The 70x idle precision improvement is visually unambiguous regardless of p-values, and the S5 pitch-gate collapse (80N to 9N) is a >8x effect that does not require subtle statistics. May also undervalue the qualitative flight/terrain demonstrations because N<10.

---

### Reviewer Configuration Card #3 -- Peer Reviewer 2 (Domain)

**Role**: Peer Reviewer 2 -- Domain Expertise
**Identity Description**: Senior researcher in wheeled bipedal and wheeled-legged robot control, with direct experience on platforms similar to Ascento, SKATER, and Whleaper. Has published on coupled pitch-roll-roll LQR, whole-body control for kinematic loops, and proprioceptive state estimation for wheeled platforms. Deeply familiar with the wheeled biped literature from 2017-present and the practical challenges of hardware deployment.

**Review Focus**:
1. Literature positioning: Is the coverage of related wheeled biped platforms (Ascento, DIABLO, Whleaper, SKATER, BITeno, Swiss-Mile) complete and accurate? The platform comparison table (Table I) makes qualitative claims about other platforms -- are these characterizations fair and well-sourced?
2. Technical correctness of the LQR bottleneck claim: The paper asserts that the "PID servo action path" is the binding constraint, proven by the coupled 6-state LQR performing identically to the 4-state baseline. Is this causal chain valid? Could the direct-torque LQR variant (which survives 0.70s vs 0.52s) be further improved with direct-torque-optimized gains rather than servo-derived gains?
3. Practical significance for the wheeled biped community: Does ACC represent a meaningful advance over the state of the art, or is it primarily a careful integration of known techniques (scheduled gains, conditional integration, envelope detection) on a specific morphology?

**Will particularly care about**: Whether the paper's claims about *why* LQR baselines fail are technically correct and whether the PID servo path is genuinely the bottleneck or a convenient explanation. Having likely debugged PID servo lag on real platforms, this reviewer will scrutinize whether the ~0.25s lag estimate is accurate for the specific actuator model and whether faster servo tuning could close the gap.

**Possible blind spots**: May be biased toward methods that have proven successful on hardware (LQR, WBC, MPC) and skeptical of a method that is simulation-only and latency-sensitive below 10ms. May dismiss the proximity-gated anchor as "just a conditional integrator" without fully appreciating the asymmetric envelope follower's role in distinguishing push ringdown from quiet stance. Also may not appreciate the process control lineage (override control, selector-based architectures) that ACC extends.

---

### Reviewer Configuration Card #4 -- Peer Reviewer 3 (Cross-Disciplinary Perspective)

**Role**: Peer Reviewer 3 -- Cross-Disciplinary / Practical Perspective
**Identity Description**: Control theorist and industrial automation researcher specializing in anti-windup compensation, override control, and split-range architectures for chemical process control. Background in nonlinear control systems with applied work on conditional integration, tracking back-calculation, and observer-based anti-windup. Brings a perspective from a field where conditional activation of control components has been standard practice for decades (e.g., override control for turbine protection, split-range for heating/cooling).

**Review Focus**:
1. Contextualization within classical control theory: ACC's smoothstep gating is positioned as extending selector-based and override control "from discrete selection to continuous blending." Is this a genuine extension or a relabeling of well-established techniques? The process control literature has used continuous blending (bumpless transfer, gain scheduling with fuzzy transition regions) since at least the 1990s.
2. Anti-windup taxonomy completeness: Appendix A positions ACC relative to conditional integration, tracking back-calculation, and observer-based AW. Is the taxonomy correctly applied? Conditional integration schemes that gate on state (not just error) exist in the process control literature -- is ACC's claim of distinctiveness accurate?
3. Generality of the gate design methodology: The paper provides a 5-step tuning procedure for porting ACC to new morphologies. Is this procedure well-justified by the sensitivity analysis, or does it assume the specific dynamics of this wheeled biped? The dominance of the release coefficient (sensitivity 220.4, 20x the next parameter) suggests the method may reduce to tuning one parameter -- is that a strength (simplicity) or a weakness (overfitting to one failure mode)?

**Will particularly care about**: Whether ACC's contribution is properly scoped relative to the classical control literature. The paper cites Astrom & Hagglund (2006) for advanced PID but does not engage deeply with the anti-windup survey literature (e.g., Tarbouriech & Turner, 2009; Galeani et al., 2009). The reviewer will ask: what did ACC invent that was not already possible with a well-designed conditional integration scheme and a bumpless transfer mechanism?

**Possible blind spots**: May undervalue the robotics-specific contributions (the asymmetric envelope follower as a solution to the push-ringdown vs. quiet-stance discrimination problem) because they are framed as control theory contributions rather than domain-specific engineering solutions. May also hold ACC to an unrealistically high standard of theoretical novelty that is atypical for robotics venues, where practical effectiveness on real platforms is the primary criterion.

---

### Reviewer Configuration Card #5 -- Devil's Advocate

**Role**: Devil's Advocate
**Identity Description**: A skeptical, methodologically rigorous reviewer who actively searches for logical gaps, overclaimed contributions, and unexamined assumptions. This reviewer treats every claim as a hypothesis to be falsified and applies the principle that "extraordinary claims require extraordinary evidence." They are not hostile -- they genuinely want the paper to be correct -- but they will probe every weak point until it either holds or breaks.

**Review Focus**:
1. **The core novelty question**: Is the "proximity-gated anchor" genuinely novel or just a well-tuned conditional integrator with an envelope detector? The individual components (conditional integration, asymmetric EMA, gain scheduling, smoothstep blending, damping boost) are all standard techniques. The paper claims the *combination* plus the *physical-condition gating* is novel -- but at what level of granularity does combining known techniques constitute a contribution? Where exactly is the line between "novel architecture" and "careful engineering"?

2. **The 70x baseline comparison**: The paper's headline result is a ~70x idle precision improvement over P-only (0.73mm vs. 55mm RMS). But the P-only baseline was tuned with k_p=35 (to maximize push survival). A fair precision comparison would be against P-only with k_p=50 (the same stiff value ACC uses), which the paper acknowledges would improve precision at the cost of push survival. Is the 70x figure comparing ACC's best against P-only's worst on the precision dimension?

3. **The LQR failure attribution**: The paper claims LQR baselines fail because of the PID servo action path bottleneck. But the direct-torque LQR variant, which removes this bottleneck, still falls in 0.70s (100% fall rate). If removing the PID servo path doesn't solve the problem, then the bottleneck explanation is incomplete -- something else is failing. What is it? Is it that LQR gains derived for a 2D TWIP are fundamentally inadequate for this morphology regardless of the action path?

4. **The latency sensitivity**: ACC requires <10ms control latency for full push performance -- a delay bifurcation drops F_max from ~95N to ~45N. This is an extremely tight requirement for a real embedded system (IMU read + state estimation + control computation + torque command). If the method cannot tolerate realistic hardware latency, what is the path to hardware validation? The paper acknowledges this but does not propose a mitigation.

5. **Simulation-only with explicit hardware fragility**: The paper is entirely simulation-based and explicitly lists 5 hardware degradation mechanisms (encoder quantization, backlash, stiction, tire compliance, IMU noise) plus fundamental latency sensitivity. Given these acknowledged gaps, are the strong architectural claims ("unified architecture," "two-channel torque assembly") premature? Should the contribution be scoped more narrowly as a simulation study demonstrating the potential of conditional gating?

6. **The compound-disturbance limitation**: Push survival degrades 84% during height transitions (94N to 15N) due to "shared torque budget." This suggests ACC's two-channel decomposition may not truly decouple balance from posture -- the torque budget is shared across channels. If the architecture's primary claim is clean separation of concerns, this result directly contradicts it.

**Will particularly care about**: Whether the paper's conclusions are supported by its evidence, or whether the evidence supports weaker conclusions than those stated. The difference between "ACC achieves sub-millimeter precision and omnidirectional push recovery in simulation" (supported) and "ACC is a unified control architecture for wheeled bipeds" (less supported, given hardware gaps and compound-disturbance limitations).

**Possible blind spots**: May be overly skeptical of simulation-only work and undervalue the genuine engineering insight in the asymmetric envelope follower design. The Devil's Advocate role can become performative if it demands a standard of evidence that no simulation paper could satisfy. The key question is whether the paper's claims are proportional to its evidence, not whether the evidence meets a hardware-validation standard the paper never claimed to achieve.

---

## Review Strategy Recommendations

**Special characteristics requiring particular attention**:
- This is a single-author, independent-researcher submission. The paper should be judged on its technical merits, not institutional affiliation, but the lack of a lab's hardware infrastructure explains the simulation-only scope and should be considered when evaluating the hardware-deferred limitation.
- The paper sits at an unusual intersection: it is fundamentally a classical control paper submitted in a robotics venue, using robotics evaluation conventions (ablation tables, push thresholds, platform comparisons) but drawing on control theory concepts (anti-windup, override control, gain scheduling). Reviewers from pure robotics backgrounds may miss the control theory context; reviewers from pure control theory may expect more theoretical analysis (stability proofs, Lyapunov arguments).
- The paper is unusually honest about limitations -- it explicitly lists 10 limitations, discusses statistical power, and scopes qualitative demonstrations as "feasibility proofs." This transparency should be rewarded, not penalized, during review.

**Potential complementarity or tension between reviewers**:
- **Tension**: Reviewer 1 (Methodology) will push for larger N and more statistical rigor; Reviewer 2 (Domain) may accept N=5 for push ablation as standard in robotics venues. The EIC will need to adjudicate what standard of evidence is appropriate for the target venue.
- **Tension**: Reviewer 3 (Cross-Disciplinary) may argue the contribution is insufficiently novel relative to the process control literature; Reviewer 2 (Domain) may argue it is sufficiently novel for the robotics community which does not typically read process control journals. This is a genuine disciplinary gap that the EIC must bridge.
- **Complementarity**: Reviewer 1's methodology critique and Reviewer 3's control-theory framing both push toward the same question from different angles: is this a genuine contribution or careful engineering? Their independent convergence (or divergence) on this question will be informative.
- **Devil's Advocate amplification**: The Devil's Advocate's concerns about the 70x comparison and the LQR failure attribution are likely to be shared in milder form by Reviewer 1 (methodology) and Reviewer 2 (domain) respectively. If all three raise similar concerns, the authors will need to address them substantively.

---

## Reviewer Coverage Matrix

| Concern | EIC | R1 (Method) | R2 (Domain) | R3 (Cross-Disc) | D.A. |
|---------|-----|-------------|-------------|-----------------|------|
| Novelty vs. known techniques | Medium | Low | High | **Critical** | **Critical** |
| Experimental validity / N | Medium | **Critical** | Low | Low | High |
| LQR baseline fairness | Medium | High | **Critical** | Low | **Critical** |
| Hardware readiness / latency | High | Low | Medium | Low | **Critical** |
| Literature completeness | High | Low | **Critical** | High | Medium |
| Generality of architecture | **Critical** | Low | Medium | High | High |
| Control theory lineage | Low | Low | Low | **Critical** | Medium |

---

*End of Reviewer Configuration Card*