# Peer Review: "Anchored Conditionally-Gated Control for Wheeled Bipedal Balance and Disturbance Recovery"

**Reviewer:** Tenured Professor, Wheeled-Legged Locomotion and Whole-Body Control
**Confidentiality:** This review is for the authors and editors only.

---

## Dimension 1: LITERATURE COVERAGE — Score: 3.0/10

### Related Work: What is Present

The paper cites ~38 references spanning wheeled-biped platforms (Ascento, Handle, Swiss-Mile, BITeno, DIABLO, SKATER, Whleaper), classical control (PID, LQR, anti-windup, split-range control), and one RL baseline (PPO). Coverage breadth is adequate for a conference paper.

### Critical Gaps

**1. Whleaper (Zhu 2024) — Inadequate Characterization.** This is the single closest prior work to ACC. Whleaper proposes an LQR + PPO residual architecture for a wheeled biped — the same platform class, the same hybrid model-based/learning framing, the same push-recovery evaluation scenario. The paper's characterization of Whleaper as merely "LQR+PPO residual" is insufficient. Specific questions the paper must answer: (a) What is Whleaper's reported push-recovery envelope? How does ACC quantitatively compare? (b) Does Whleaper report idle precision? If not, that is a legitimate gap the paper can claim, but it must be explicit. (c) Whleaper uses RL to learn the residual — ACC uses classical control only. Is ACC a *replacement* for the RL component, a *precondition* for it, or an *alternative*? The relationship must be clarified. As written, the paper creates the impression that Whleaper is a distant neighbor rather than the most directly comparable prior result. **This is a major revision requirement.**

**2. Ascento (Klemm et al. 2019, 2020).** The characterization is broadly correct — Ascento uses MPC/WBC for balance and locomotion. However, one nuance is missing: Ascento's hopping and jump recovery (Klemm 2020) demonstrates a form of "flight recovery" that predates ACC's. The paper should explicitly compare ACC's load-gated flight hysteresis to Ascento's airborne state machine. Are they functionally equivalent? If ACC's version is simpler (it likely is — Ascento's involves trajectory optimization for landing), this is a selling point that should be stated. If they are solving different problems (jumping vs. drop recovery), that distinction needs to be drawn.

**3. Zamani et al. (2020) — Coupled Pitch-Roll LQR.** The paper claims that the LQR baselines' failure (100% fall rate) arises from "platform complexity, not servo lag" — distinguishing ACC's robot from Zamani's two-wheeled platform. This is a fair point but it is asserted, not demonstrated. The paper does not show what happens when Zamani's exact LQR gains are ported to ACC's robot (which would likely fail faster). More importantly, the paper does not investigate whether the *class* of LQR controller (decoupled vs. coupled, SISO vs. MIMO, continuous-time vs. discrete-time) is the issue, or whether *any* linear controller is insufficient. The logical chain is: "LQR fails on our platform → our platform is more complex than Zamani's → therefore our controller is needed." But this skips a step: "Would a properly-derived, discrete-time, fully-coupled 6-state or 10-state LQR also fail?" The paper acknowledges (Limitation 6) that the coupled LQR was not re-derived for the DT plant. This is a significant gap in the comparative evaluation, not a minor limitation.

**4. SKATER (Wang 2025).** Correctly positioned as MPC-based. No issues.

**5. Swiss-Mile / Handle / BITeno / DIABLO.** These are correctly identified as platform references rather than method references. The characterizations are accurate. No issues.

**6. Anti-Windup Literature.** The paper claims that saturation-triggered anti-windup (conditional integration, back-calculation, tracking) is "insufficient" because it is reactive rather than anticipatory. This is a potentially valid argument but it is not supported by citation or demonstration. The paper should either: (a) cite specific anti-windup schemes and explain why they fail for *this* problem (e.g., cite Åström & Hägglund 2006, Peng et al. 1996, Bohn & Atherton 1995 for specific schemes), or (b) demonstrate a saturation-triggered AW baseline failing on the same push-recovery task. As written, the claim is a strawman. I note that the anti-windup literature is vast; claiming it is "insufficient" without a specific comparison is not acceptable for a publication-quality paper.

**7. Hybrid/Switched Systems (Liberzon 2003).** The paper cites Liberzon to connect gated control to hybrid systems theory. This is a superficial connection. ACC's gates are continuous (smoothstep) — there is no discrete mode switching, no dwell-time analysis, no Lyapunov stability across modes. The paper does not use any analytical tools from hybrid systems theory (multiple Lyapunov functions, invariant sets, common Lyapunov). If the connection to hybrid systems is not analytical, it should either be developed (with formal analysis) or removed. A one-sentence citation of Liberzon does not constitute engagement with that literature.

### Missing Baselines and Comparisons

**8. DCM / Capture Point.** Push recovery for legged robots has a substantial literature using Divergent Component of Motion (DCM) and capture point (Pratt et al. 2006, Englsberger et al. 2011, 2015, Koolen et al. 2012). While these were developed for walking robots, the underlying concept — controlling the instantaneous capture point to remain within the support polygon — applies to wheeled bipeds as well. The paper should at minimum cite this literature and explain why it is not a relevant comparison (e.g., "DCM approaches require footstep adjustment, which wheeled bipeds cannot do"). Without this, the push-recovery framing appears to be developed in a vacuum.

**9. Whole-Body Control (WBC).** Several wheeled-biped papers use WBC formulations (e.g., Ascento's hierarchical QP, Xin et al. 2022 for BITeno). The paper's "two-channel torque assembly" (τ_w, τ_q) is essentially a simplified WBC. The relationship should be discussed — is ACC a special case of WBC with hand-tuned task weights? Is it an alternative that avoids the computational cost of QP solving? This would strengthen positioning.

**10. MPC Baselines.** The paper compares against LQR (four variants) and PPO. MPC — particularly linear MPC with the same linearized model used for LQR — is an obvious intermediate baseline. A simple finite-horizon LQR (which is what unconstrained linear MPC reduces to) could handle the integral windup problem through receding-horizon replanning without needing ACC's gate machinery. The paper's claim that "LQR fails" would be stronger if it also showed that a simple MPC baseline fails (or that ACC outperforms MPC on computational cost).

### Literature Score Justification

The paper covers the wheeled-biped platform landscape adequately but misses critical depth on its closest comparator (Whleaper), does not engage with the DCM/capture-point push-recovery literature, fails to provide an anti-windup baseline comparison, and makes a superficial citation to hybrid systems theory. These are not minor omissions — they affect how the contribution is framed.

---

## Dimension 2: THEORETICAL FRAMEWORK — Score: 4.0/10

### Is the "Conditionally-Gated" Concept Genuinely Novel?

The core idea — using a spatial proximity gate and a temporal envelope follower to enable/disable integral action — is an engineering combination of known techniques. Let me decompose:

**Gate 1: Proximity gate (g_prox, smoothstep from 5-15cm).** This is a spatial dead-zone with soft transition. The idea of disabling integral action far from equilibrium is present in the anti-windup literature (e.g., "conditional integration" in Åström & Hägglund 2006, Section 3.5). The smoothstep function makes it continuous, which is a practical improvement but not a conceptual one.

**Gate 2: Envelope gate (g_env, asymmetric EMA of |v_sag|).** The asymmetric envelope follower (τ_attack ≈ 23ms, τ_release ≈ 1.5s) is the most interesting component. It distinguishes a fast disturbance onset from slow recovery — enabling anchor re-engagement only after ringdown. Is this novel in robotics control? Asymmetric envelope tracking is standard in audio dynamics processing (compressor/expander circuits, RMS detectors with asymmetric time constants, dating to the 1930s), and has been used in robotics for contact detection and collision monitoring (Haddadin et al. 2008, 2017 use momentum observers with asymmetric filtering for collision detection). The application to integral gating in a wheeled biped is *new*, but the technique itself is not. The paper should cite prior uses of asymmetric filtering in robotics to properly position this contribution.

**Gate 3: Pitch safety interlock (g_θ, 2-12°).** Straightforward state-based gating. Not novel.

**Combined gating: g_anchor = g_prox · g_env · g_θ.** The multiplicative combination of three continuous gates creates a smooth transition surface. This is clean engineering but does not constitute a new control-theoretic construct.

### Split-Range / Override Control Distinction

The paper acknowledges split-range control (Shinskey 1996) and override/selector control (Åström 2006) and claims ACC is different because it uses "continuous multiplicative gating rather than if/else selection." This distinction is thin. Split-range control with continuous blending functions (e.g., using smooth interpolation rather than hard switching) is a standard extension. The paper's smoothstep gating is a specific choice of blending function. Calling this a fundamentally different *class* of controller overstates the novelty.

**Recommendation:** Reframe ACC as a specific *instance* of conditionally-structured control rather than a new *class*. The contribution is the specific gate design and parameterization for wheeled-biped balance, not the gating concept itself.

### Asymmetric Envelope Follower — Contribution or Engineering Detail?

This is the strongest candidate for genuine novelty, but the paper undersells it by burying the analysis. The gate sensitivity analysis (τ_release sensitivity = 220.4, 20× the next parameter) is excellent and suggests that the release time constant is the dominant design parameter. However, the paper does not explain *why* this asymmetry is necessary or what physical phenomenon it captures. A brief physical argument — e.g., "τ_attack must be faster than the push rise time (~20ms for impulsive pushes) to detect disturbance onset, while τ_release must be longer than the ringdown time constant (~1s for the underdamped pitch mode) to prevent premature anchor re-engagement" — would transform this from an empirical observation into a principled design.

### Gated Damping Boost (Ringdown)

The gated damping boost (g_boost = g_prox · g_env · g_θ) that activates additional damping during ringdown is a neat idea — essentially a state-dependent damping schedule. This is reminiscent of variable impedance control and sliding-mode reaching-law design, though the paper does not make these connections. The effectiveness of the boost is demonstrated (the ringdown time-series figure), but its necessity relative to a fixed higher damping is not established. Would a constant higher damping coefficient achieve similar ringdown suppression at the cost of higher torque during quiet stance? If so, the gate is buying efficiency, not capability. This should be quantified.

### Theoretical Framework Score Justification

The framework is a well-engineered combination of known techniques. The asymmetric envelope follower applied to integral gating is the most novel element, but it is not contextualized within prior signal-processing uses in robotics. The multiplicative gating architecture is elegant but does not constitute a new control-theoretic class. The connection to hybrid systems theory is decorative. The framework is sound and the design is principled, but the claimed novelty is overstated.

---

## Dimension 3: DOMAIN CONTRIBUTION — Score: 5.0/10

### What the Paper Actually Demonstrates

The paper demonstrates that a carefully tuned classical controller (ACC) achieves:
1. Sub-millimeter idle precision (0.73mm CoM RMS) on a simulated wheeled biped.
2. Omnidirectional push recovery up to ~119N median, ~83N minimum direction.
3. Survival under 100cm drops and 50cm ledges.
4. Straddling of a 20cm curb.
5. Outperformance of four under-tuned LQR baselines and one PPO baseline (85% fall rate).

### Contribution 1: Proximity-Gated Anchor

**Verdict: Genuine contribution, scoped too broadly.**

The proximity-gated anchor is the core mechanism. The asymmetric envelope follower is the key enabler. The sub-millimeter precision is a real result. However, the claim "without sacrificing omnidirectional push survival" needs qualification — the backward direction is weakest (the paper acknowledges this as Limitation 2), and push survival degrades 84% during height transitions (Limitation 4). "Omnidirectional" typically implies isotropic performance; ACC's polar envelope shows significant anisotropy. "Multidirectional" would be more accurate.

The use of "sub-millimeter" as the headline (0.73mm) also deserves scrutiny (see Claims Check below).

### Contribution 2: Flight + Terrain Extensions

**Verdict: Minor contributions, correctly scoped as "independent extensions."**

The flight recovery (load-gated hysteresis) is simple and effective. The per-leg terrain adaptation (splitting height commands from contact-point estimates) is also simple. Both are N=10 in simulation only. These are nice demonstrations but are not core contributions — the paper correctly labels them as "independent extensions." I appreciate this honesty.

However, the paper should clarify what these extensions demonstrate about the *architecture*. Is the point that ACC's two-channel torque assembly is *composable* — that flight and terrain can be added without retuning the balance core? If so, that is the contribution to articulate. If these are just bolt-on features that work, the contribution is weaker.

### Contribution 3: Factorial Ablation

**Verdict: Not a contribution. This is good experimental practice.**

A factorial ablation (L0→L3 additive, S0→S5 subtractive) is thorough evaluation, not a contribution. Claiming it as a contribution inflates the contribution count. The paper would be stronger with two contributions: (1) the proximity-gated anchor mechanism, and (2) the demonstration that this mechanism enables composable extensions (flight, terrain). The ablation then supports Contribution 1, rather than being a contribution itself.

### What the Paper Does NOT Demonstrate

- **Hardware validation.** Acknowledged as limitation. Acceptable for a simulation paper, but limits the contribution weight. Simulation-only results on a MuJoCo model of a custom robot carry less weight than hardware results on a standard platform.
- **Generality across morphologies.** Acknowledged. The controller is tuned to one specific 8.1kg, 10-DOF robot.
- **Comparison to state-of-the-art.** Whleaper is the closest comparator and is not quantitatively compared. DCM-based push recovery is not compared. MPC is not compared.
- **Guarantees.** No stability analysis, no region-of-attraction estimates, no formal robustness margins (though the empirical robustness sweeps are good).

### Domain Contribution Score Justification

The proximity-gated anchor is a genuine contribution to the wheeled-biped control literature, specifically addressing the precision-vs-recovery trade-off. The empirical results are thorough for a simulation study. However, the contribution weight is limited by: (a) simulation-only validation, (b) lack of comparison to the closest prior work (Whleaper), (c) overclaimed contribution count (ablation as contribution #3), and (d) no formal analysis. For RA-L, this level of contribution might be acceptable with revisions. For T-RO, it would need hardware validation or formal analysis.

---

## Dimension 4: POSITIONING — Score: 3.5/10

### ACC as "Structured Prior" for RL

The paper positions ACC as "hypothesized as a precondition for RL on this platform" — meaning ACC provides a structured action prior that residual RL can learn around. This is a coherent vision. However, the paper itself is purely classical control and contains zero RL results beyond the baseline comparison. The hypothesis that ACC is a *precondition* for RL is untested.

This creates a tension: the paper is submitted as a results paper (it has 7 pages of results), but its framing points toward a future RL paper. The question for the editor is: is ACC as a standalone classical controller sufficiently novel and well-validated, independent of its potential RL role?

If the answer is yes, the paper should be positioned as a classical control paper and the RL discussion should be moved to a "Future Work" paragraph. If the answer is no, the paper should be held until RL results are available. The current hybrid positioning — classical control results with RL-motivated framing — satisfies neither audience fully.

**Recommendation:** For RA-L, restructure as a classical control paper with a brief "Implications for Learning-Based Control" subsection. For a longer venue (T-RO), include initial RL results (even preliminary) to validate the precondition hypothesis.

### Positioning Relative to Whleaper

The paper positions ACC against LQR and PPO baselines, but not against Whleaper. This is a strategic gap. The reader familiar with wheeled-biped literature will ask: "How does ACC compare to Whleaper's LQR+PPO residual?" If the answer is "we don't know," that weakens the contribution claim. If the answer is "ACC achieves comparable push recovery without needing RL," that is a strong positioning statement that should be made explicitly.

### Positioning Score Justification

The RL-motivated framing is premature for a paper with no RL results beyond a baseline. The positioning against the field is incomplete without quantitative comparison to Whleaper. The internal logic is clear (precision-vs-recovery trade-off → gated anchor → ACC → results), but the external positioning is muddled.

---

## Dimension 5: COMPARATIVE POSITIONING — Score: 4.0/10

### Table I (Platform Comparison)

The use of "---" to indicate "not reported" is standard in survey tables, but it creates asymmetry: it implies ACC *has* the metric while others do not, without distinguishing "the platform cannot do this" from "the authors did not measure this." Recommendations:

- Add a note: "--- indicates value not reported in the cited work; this does not imply the platform is incapable of the behavior."
- If possible, mark whether the metric is "not reported" versus "not applicable" (e.g., flight recovery is N/A for a platform that does not attempt drops).
- Consider whether Whleaper's push-recovery force or idle precision can be estimated from their published figures even if not numerically reported. If so, add an "~estimated" entry.

### LQR as Baseline Class — Is This Fair?

The paper uses four LQR variants as the primary classical comparison. All fail with 100% fall rate. The question is whether *any* linear controller (properly tuned) would fail, or whether the specific LQR implementations are under-tuned.

Concerns:
1. The LQRs use a 3-state model (pitch, pitch rate, forward velocity). The robot has 10 DOF. A 3-state LQR is a deliberately weak baseline.
2. The paper acknowledges that the coupled 6-state LQR was not re-derived for the DT plant (Limitation 6). This means the "coupled" LQR is a CT design applied to a DT system — it may be destabilized by discretization, not by inherent limitations of LQR.
3. The state penalty matrices (Q) and control penalty (R) are not specified in the paper. Without knowing the tuning effort, the reader cannot distinguish "LQR inherently fails" from "we did not tune LQR well."
4. The PID servo action path is identified as a bottleneck. But this is a *simulation artifact* (the robot model uses PID position/velocity servos) — a hardware robot would have different actuator dynamics. Would LQR perform better on hardware with direct torque control? This is not discussed.

**Recommendation:** Either (a) re-derive a properly discretized LQR and tune it fairly, or (b) reframe the LQR comparison as "simple linear baselines" rather than "LQR" (which implies optimality), or (c) report the LQR tuning process and Q/R matrices transparently so readers can assess fairness.

### PPO Baseline

The PPO baseline (100M steps, 85% fall rate) is a useful reference but raises questions:
- Was PPO trained on the same reward function across all heights? Multi-height transitions are challenging for RL — did the training curriculum include them?
- 100M steps is substantial for a MuJoCo environment. Did the reward saturate? Was the policy architecture appropriate?
- The paper uses PPO's failure to motivate ACC, but a poorly-performing RL baseline does not make a classical controller look better — it makes the baseline look weak.

### Comparative Positioning Score Justification

The LQR baselines are under-specified and potentially under-tuned, weakening the comparative claims. The PPO baseline provides a useful reference but is not the right foil for a classical control paper. The absence of Whleaper comparison is the most significant gap.

---

## Dimension 6: CLAIMS CHECK — Score: 4.5/10

### "Sub-Millimeter Precision" (0.73 ± 0.07mm)

The measurement is: CoM RMS over N=10 60-second episodes with EMA₀=0 transient bias acknowledged.

**Issues:**
1. The EMA₀=0 transient is properly acknowledged, which is good. But 0.73mm is steady-state precision after the transient. What is the precision *including* the first 5 seconds? If it is 2-3mm, the "sub-millimeter" headline should be qualified as "steady-state sub-millimeter precision."
2. RMS is a quadratic metric sensitive to outliers. Is 0.73mm dominated by a few larger deviations or is it consistently sub-millimeter? A percentile distribution (p50, p95, p99) would be more informative than RMS alone.
3. The paper reports 95% CI [0.68, 0.78]mm, which is good statistical practice. However, N=10 episodes may not be enough for a tight CI if there is between-episode variance. Were the 10 episodes from a single simulation run with different random seeds, or repeated runs of the same seed?
4. How is "idle" defined? Is the robot commanded to a fixed height, or is there a height command active? External disturbances? Sensor noise? These should be specified.

**Verdict:** The 0.73mm claim is plausible and well-measured for steady-state, but "sub-millimeter" is a strong headline that should be qualified as "steady-state sub-millimeter idle precision."

### "Omnidirectional Push Recovery"

The paper reports push recovery in 8 directions, F_min=83N, F_med=119N, N=5 per direction.

**Issues:**
1. F_min=83N is the *minimum across directions* (likely backward or oblique-backward). "Omnidirectional" implies all directions survive pushes of at least F_min — this is technically true (8/8 directions survive at F_min) but the survival force varies by direction. The polar envelope figure shows this, which is honest.
2. N=5 per direction is small. The 95% CI for per-direction survival force with N=5 will be wide.
3. The push profile (force magnitude, duration, rise time, application point) matters. Is it an impulse, a step, or a ramp? Where on the body is the force applied? These parameters affect comparability with other push-recovery results.

**Verdict:** "Multidirectional" is more accurate than "omnidirectional." The paper should report the push profile in full detail.

### "Four Classical LQR Baselines All Fail"

As discussed above, this claim is weak because the LQR implementations are under-specified and potentially under-tuned. The paper should either strengthen or qualify this claim.

---

## Overall Assessment

### Summary

The paper presents a carefully engineered classical controller (ACC) for a simulated wheeled biped. The core mechanism — a proximity-gated position integral with asymmetric envelope detection — addresses a real trade-off between idle precision and push recovery. The empirical evaluation is thorough for a simulation study (factorial ablation, gate sensitivity analysis, robustness sweeps). The writing is clear and the figures are informative.

However, the paper has significant weaknesses in literature engagement, comparative evaluation, and contribution framing. The closest prior work (Whleaper) is not quantitatively compared. The LQR baselines are under-specified. The contribution count is inflated (ablation is not a contribution). The RL-motivated positioning is premature. The connection to hybrid systems theory is superficial. Several claims ("omnidirectional," "sub-millimeter" without qualification, "LQR fails") need tempering.

### Recommendation

**Major Revision.** The paper has a solid engineering core but needs substantial revision to be publishable at RA-L or T-RO. The required revisions are:

1. **Engage with Whleaper quantitatively.** Even if the comparison is approximate (estimating Whleaper's push force from published figures), the paper must position itself against the closest prior work.

2. **Strengthen or qualify the LQR comparison.** Either re-derive and fairly tune the LQR baselines, or reframe them as "simple linear baselines" rather than claiming they represent the LQR class.

3. **Redefine contributions.** Drop the factorial ablation as Contribution 3. Reframe as two contributions: (1) the gated anchor mechanism, (2) composable architecture demonstrated via flight/terrain extensions.

4. **Decide on positioning.** Either restructure as a pure classical control paper (moving RL to Future Work) or add preliminary RL results to validate the precondition hypothesis.

5. **Qualify key claims.** "Multidirectional" not "omnidirectional." "Steady-state sub-millimeter" not "sub-millimeter." Contextualize the asymmetric envelope follower within prior signal-processing uses in robotics.

6. **Remove or develop the hybrid systems connection.** If no formal analysis, remove the Liberzon citation and hybrid-systems framing.

### Dimension Scores

| Dimension | Score | Weight | Weighted |
|-----------|-------|--------|----------|
| Literature Coverage | 3.0 | 0.20 | 0.60 |
| Theoretical Framework | 4.0 | 0.20 | 0.80 |
| Domain Contribution | 5.0 | 0.30 | 1.50 |
| Positioning | 3.5 | 0.15 | 0.525 |
| Comparative Positioning | 4.0 | 0.10 | 0.40 |
| Claims Check | 4.5 | 0.05 | 0.225 |
| **OVERALL** | | | **4.05/10** |

### Strengths

- The asymmetric envelope follower (τ_attack ≈ 23ms, τ_release ≈ 1.5s) for integral gating is clever and well-motivated by the physics.
- The gate sensitivity analysis identifying τ_release as the dominant parameter (sensitivity 220.4) is excellent and rare in robotics papers.
- The factorial ablation (L0→L3, S0→S5) with Welch's t-test is thorough and allows readers to assess each component's contribution independently.
- The torque rate-limit invariance analysis (200-800 Nm/s) demonstrates robustness to a practically important parameter.
- Honest limitation disclosure (8 items) is commendable and builds credibility.

### Weaknesses

- Whleaper (Zhu 2024) — the closest prior work — is not quantitatively compared. This is the single most significant gap.
- The LQR baselines are underspecified and potentially under-tuned, weakening the central comparative claim.
- Contribution #3 (factorial ablation) is not a contribution; it is good experimental practice.
- The RL-motivated positioning is premature for a paper with zero RL results beyond a baseline.
- The connection to hybrid systems theory (Liberzon 2003) is superficial and should be removed or developed.
- "Omnidirectional" and "sub-millimeter" claims need qualification.
- The asymmetric envelope follower is not contextualized within prior uses of asymmetric filtering in robotics (e.g., momentum observers for collision detection).

### Questions for Authors

1. What is Whleaper's reported push-recovery force (even approximately from their published figures)? How does ACC's F_med=119N compare?
2. What Q and R matrices were used for the four LQR variants? Were the gains tuned systematically or taken from the cited papers directly?
3. Did you attempt a simple MPC baseline (e.g., finite-horizon LQR with the same linearized model)? If not, why not?
4. How does the PPO failure rate change if the policy is trained with a height curriculum rather than random height commands?
5. Have you tested ACC on a different wheeled-biped morphology (e.g., scaling the robot model by ±20% mass/length) to assess architectural generality?
6. Can the asymmetric envelope follower be derived from a physical model (e.g., the underdamped pitch mode's ringdown time constant), or is it purely empirically tuned?

---

**Confidential note to editor:** This paper has a publishable core but needs significant revision. The closest prior work (Whleaper) must be engaged, the LQR comparison must be strengthened, and the positioning must be clarified. I recommend major revision with the expectation that the authors can address these concerns in one round. The gate sensitivity analysis and honest limitation disclosure are strengths that suggest the authors are capable of the required revisions.