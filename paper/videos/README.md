# Video evidence for the ACC paper

Every clip here is rendered by `scripts/render_paper_videos.py`, which drives the
*same* harness that produced the corresponding table or figure. A clip and the
number it illustrates therefore cannot drift apart:

| group | harness | backs |
|---|---|---|
| idle, push | `scripts/push_sweep_paper.PushSim` | Table `tab:standing`, Table `tab:push_ablation`, Fig. `fig:push_polar` |
| drop | `scripts/viz_drop_recovery` → `scripts/drop_recovery_tests.DropSim` | §`sec:flight_terrain_results`, contact-loss recovery |
| curb, ledge | `scripts/ramp_step_tests` | §`sec:terrain`, §`sec:flight_terrain_results` |
| height | `scripts/viz_v3_homing_height` | §`sec:posture`, commanded height transitions |

Regenerate everything:

```bash
python scripts/render_paper_videos.py
```

or one group at a time: `--only push drop curb ledge height idle`.

All clips: MuJoCo, 500 Hz physics / 100 Hz control, 400 Nm/s torque rate limit,
V3_ANCHOR (= ACC) profile, 640×480, real time. The overlay shows elapsed time
and the live state the clip is about.

---

## Idle standing — `idle_standing.mp4`

Quiet stance, 12 s, zero disturbance, run through the push harness with force 0
so it is the identical code path as every push clip. This is the regime that
measures at 0.294±0.015 mm sagittal repeatability (`tab:standing`).

## Push recovery

Torso impulse, 7 control steps (70 ms), applied after a 3 s settle; 17 s of
recovery follow. Bearings are world-frame: 0°/180° lateral (track axis, world
x), ±90° sagittal (rolling axis, world y) — the convention of §`sec:height_convention`.

A red arrow strikes the torso along the applied force, its length scaled to the
magnitude, the same overlay the teleop viewer draws. The impulse itself lasts
70 ms and would be invisible, so the arrow is **held for 2 s** and drawn
translucent once the force is off — solid arrow = force applied, faded arrow =
the push that just happened. The overlay is drawn after the physics step and
changes nothing about the trial.

| clip | force | bearing | outcome |
|---|---|---|---|
| `push_90N_lateral.mp4` | 90 N | 0° | recovers — the push the ringdown figure plots |
| `push_75N_+000deg_lat_+x.mp4` | 75 N | 0° | recovers |
| `push_75N_+045deg_diag_+x+y.mp4` | 75 N | +45° | recovers — weakest bearing |
| `push_75N_+090deg_sag_+y.mp4` | 75 N | +90° | recovers |
| `push_75N_+135deg_diag_-x+y.mp4` | 75 N | +135° | recovers |
| `push_75N_+180deg_lat_-x.mp4` | 75 N | 180° | recovers |
| `push_75N_-135deg_diag_-x-y.mp4` | 75 N | −135° | recovers |
| `push_75N_-090deg_sag_-y.mp4` | 75 N | −90° | recovers — strongest bearing |
| `push_75N_-045deg_diag_+x-y.mp4` | 75 N | −45° | recovers |
| `push_85N_+045deg_over_threshold.mp4` | 85 N | +45° | **falls** — deliberate |

75 N is chosen to clear the weakest *single trial*, not the weakest mean. The
measured per-bearing thresholds (`outputs/paper_statistics/ablation_n10_results_freshctx_S1.json`,
N=10 each) are:

| bearing | mean (N) | sd | min of 10 |
|---|---|---|---|
| +45° | **82.3** | 2.3 | **77.1** |
| −135° | 91.9 | 2.1 | 88.6 |
| 0° | 95.6 | 0.3 | 95.3 |
| 180° | 100.7 | 0.6 | 100.1 |
| −45° | 107.3 | 8.5 | 94.1 |
| +90° | 117.8 | 11.1 | 103.2 |
| +135° | 132.2 | 1.8 | 128.0 |
| −90° | 144.8 | 5.2 | 140.7 |

`F_min = 82.3 N` is a mean whose reps span 77.1–83.8 N, so a single 80 N trial
at +45° can and does fall. The 75 N / 85 N pair at +45° brackets that boundary:
below every observed threshold, and above the mean.

## Drop recovery — `drop_{10,20,40,60,80,100}cm.mp4`

Released from standing posture at height h with zero velocity, autonomous
through touchdown. Verdicts from `scripts/drop_recovery_tests.py` at the same
seed as the renders, **6/6 PASS** under the strict settle criterion:

| h (cm) | touchdown v_z (m/s) | peak pitch (°) | settle (s) | tail height err (mm) |
|---|---|---|---|---|
| 100 | −4.32 | 24.0 | 1.10 | 6.1 |
| 80 | −3.82 | 22.3 | 4.75 | 5.8 |
| 60 | −3.33 | 17.6 | 4.50 | 5.8 |
| 40 | −2.65 | 11.5 | 3.94 | 5.8 |
| 20 | −1.76 | 7.7 | 3.49 | 5.8 |
| 10 | −1.17 | 6.1 | 3.35 | 5.9 |

The 100 cm peak pitch of 24.0° matches the paper's 24.0±1.2° over N=10.

## One-wheel curb — `curb_{10,15,20}cm.mp4`

Drive onto a curb with one wheel, straddle its full 2 m, drive off the end, stop
and anchor. Tests the per-leg ground estimate of §`sec:terrain`. **3/3 PASS**:

| h (cm) | max straddle roll (°) | commanded leg-length split (cm) | settle (s) |
|---|---|---|---|
| 10 | 2.5 | 10.3 | 6.06 |
| 15 | 3.0 | 15.4 | 5.67 |
| 20 | 4.0 | 21.3 | 5.55 |

Matches the paper's 2.4±0.1 / 2.9±0.2 / 3.9±0.1° over N=10, all 10/10 traversed.
The verdict here is *traversal*, not settling: a trial fails if the elevated
wheel leaves the 36 cm slab at any point, which a settle-only criterion cannot
see (the robot lands on flat ground and settles anyway).

At 20 cm the split commands 0.193 m of leg-length difference against the 0.20 m
step, so 7 mm goes unspanned — 1.4° of geometric roll — and the remaining ~2.5°
is servo error in the deeply flexed leg. The earlier 14.9° figure was not
geometry either: the stability envelope read the legitimate straddle roll as an
incipient fall and suppressed heading hold for the whole crossing, so the mount
transient went undamped and the wheel yawed off the slab 1.3 m in. The
split-proportional roll band and the settled-terrain relaxation in
`k2_jax_controller.py` fix that; see §`sec:terrain`.

## Ramp climb and ledge descent — `ramp_step_*.mp4`

12° ramp up to a platform, then off the ledge. **All PASS (seed 0)**:

| clip | h (cm) | course | peak landing pitch (°) | settle (s) |
|---|---|---|---|---|
| `ramp_step_20cm.mp4` | 20 | forward up, off the ledge | 19.3 | 6.06 |
| `ramp_step_30cm.mp4` | 30 | forward up, off the ledge | 21.4 | 6.03 |
| `ramp_step_40cm.mp4` | 40 | forward up, off the ledge | 24.4 | 6.37 |
| `ramp_step_50cm.mp4` | 50 | forward up, off the ledge | 28.2 | 6.51 |
| `ramp_step_30cm_up_down.mp4` | 30 | up, anchor 2.5 s, reverse back down | 12.4 | 5.03 |
| `ramp_step_30cm_back_off.mp4` | 30 | driven backward end to end, rear-first drop | 15.4 | 5.29 |
| `ramp_step_50cm_back_off.mp4` | 50 | driven backward end to end, rear-first drop | 25.6 | 5.34 |
| `ramp_step_30cm_diag_off45.mp4` | 30 | 45° oblique edge, wheels leave in sequence | 5.3 | 3.90 |

The 50 cm peak of 28.2° sits inside the paper's 27.1±2.0° over N=10.

**Read the oblique clip carefully.** It is one trial at seed 0, at 45°/30 cm —
inside the *deep* regime where both wheels release together and the flight path
takes over. The paper's 192-trial calibration puts that regime at 75% success at
its peak, and reports the shallow straddle window closing completely over
22–27 cm at 30°. A single passing clip is an illustration of the regime, not
evidence of a reliable capability.

## Commanded height transitions

| clip | motion |
|---|---|
| `height_transition.mp4` | continuous cosine squat/extend cycle between the calibrated low and high postures, 20 s |
| `height_standup_sitdown.mp4` | nominal → full extension → full squat, then hold |
| `height_sitdown_standup.mp4` | nominal → full squat → full extension, then hold (reverse order of the row above) |

All three update the CoM height command **and** the height-scheduled joint posture
`q_ref` each step. Commanding height alone does not move this robot — the
posture reference has to move with it. `height_sitdown_standup.mp4` uses the same
`standup_sitdown` mode with `--reverse`, which swaps which extreme is visited first;
both directions settle without falling (`fell=False`).

---

## Not covered by video

These paper results are numerical and have no meaningful single-clip rendering:
the robustness sweep over sensor noise and actuator delay (`tab:robustness`),
the classical-baseline and PPO comparison (`tab:lqr_baselines`), the factorial
component ablation (`tab:push_ablation`), and the closed-form parking-offset
model. Videos of individual trials from those tables would show a robot standing
still and would not distinguish the arms.
