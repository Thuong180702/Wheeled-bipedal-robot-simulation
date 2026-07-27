# Supplementary Video Material

## ACC Paper: Anchored Cascade Control with Conditional Activation for Wheeled Bipedal Balance and Disturbance Recovery

### Video 1: Anchored Standing (anchor_standing.mp4)
- **Source:** `outputs/visual/anchor_standup_sitdown_5cm.gif`
- **Description:** ACC anchored standing with commanded height transitions (±5 cm at anchor).
- Demonstrates mm-level idle precision (0.3 mm CoM RMS) and smooth height tracking.

### Video 2: Omnidirectional Push Recovery (push_recovery.mp4)
- **Source:** `outputs/visual/push_90N.mp4`, `outputs/visual/push_back_90N.mp4`, `outputs/visual/push_lateral_90N.mp4`
- **Description:** ACC surviving 90 N pushes from forward, backward, and lateral directions. Shows the full recovery trajectory: ballistic catch → ringdown → re-anchor.
- Key metric: F_min=70 N, F_med=115 N (24-direction polar sweep).

### Video 3: Push Ablation — Ringdown Comparison (push_ringdown_comparison.mp4)
- **Source:** `outputs/visual/anchor_vs_homing_90Nfwd.gif`
- **Description:** Side-by-side: ACC (anchor, top) vs. no-anchor baseline (bottom) after 90 N forward push. ACC ringdown completes in ~9 s; baseline oscillates indefinitely.

### Video 4: Contact-Loss Recovery — Free-Fall Drop (drop_recovery.mp4)
- **Source:** `outputs/visual/drop_100cm.mp4`, `outputs/visual/drop_60cm.mp4`
- **Description:** Robot dropped from 60 cm and 100 cm onto flat ground. Flight attitude PD engages during contact loss, using wheels as reaction flywheels. Peak landing pitch: 23.5° (100 cm), 16.8° (60 cm). 10/10 survival.

### Video 5: Contact-Loss Recovery — Ledge Drive-Off (ledge_driveoff.mp4)
- **Source:** `outputs/visual/ramp_step_50cm.mp4`, `outputs/visual/ramp_step_20cm.mp4`
- **Description:** Robot drives up 12° ramp, across platform, off cliff edge at 1.0 m/s. Flight PD recovers attitude during descent; robot re-anchors after landing. Peak pitch: 27.4° (50 cm), 20.5° (20 cm). 4/4 survival all heights.

### Video 6: Per-Leg Terrain Adaptation — Curb (curb_terrain.mp4)
- **Source:** `outputs/visual/curb_20cm.mp4`, `outputs/visual/curb_10cm.mp4`
- **Description:** Robot drives one wheel onto 10 cm and 20 cm curbs. Per-leg ground adaptation splits height commands; torso remains approximately level (straddle roll 4.8-6.2°). Without adaptation, roll reaches 15.3° and robot falls.

### Video 7: Teleop Driving — Flat (teleop_driving.mp4)
- **Source:** `outputs/visual/v3_anchor_flat_drive_stop.mp4`
- **Description:** ACC teleop: drive forward, stop, re-anchor. Demonstrates cruise control (0.8 m/s), brake-to-stop, and instant re-anchoring at the stopped position (pos error <2 cm).

### Video 8: Ablation — No Flight PD (no_flight_pd.mp4)
- **Description:** Drop 100 cm with flight PD disabled (spin-down only). Unloaded wheels free-spin; reaction torque tumbles the base. Peak pitch >90° (flip), 0/10 survival. Contrast with Video 4.
