# Missing terms analysis

Classification from current code and fresh diagnostics:

| Term/layer | Classification | Evidence |
|---|---|---|
| Pitch feedback | already present | `DualRateBalanceController.compute_action()` computes wheel LQR from pitch and pitch_rate. |
| Forward velocity feedback | already present | Wheel LQR uses wheel-derived `fwd_vel` and `base_lin_vel` proxy. |
| Roll angle feedback | present but disabled | Config `roll.kp=0`, `roll.kd=0`, `roll.max_correction=0`; controller therefore writes hip_roll actions as zero. |
| Roll rate feedback | present but disabled | Same disabled roll block; no roll damping reaches hip_roll actuators. |
| Lateral CoM position feedback | required for standing, missing | No lateral CoM state enters controller; `com_y` in code is forward channel naming, not lateral X. |
| Lateral CoM velocity feedback | required for standing, missing | No body-frame lateral velocity control path drives hip roll/contact-force balance. |
| Contact force difference feedback | optional but useful, missing | Contact forces are only diagnostics, not controller inputs. |
| Static gravity/contact preload | required and currently wrong | The saved balanced-root table stores root poses that produce invalid t=0 wheel clearance/contact states; Step 5 also ignores root pose and applies only leg joints. |
| Lateral balance layer | required after reset is fixed | The current controller has no active roll/lateral closed loop, but reset/static equilibrium fails before this can be isolated as the primary cause. |
| VMC/whole-body force distribution | likely required for robust standing | Hip_roll position targets alone do not map desired body roll/lateral wrench to wheel normal force distribution; diagnose after reset is physically valid. |
| Early high-rate stabilization | partially present for pitch only | Fast 50 Hz wheel loop exists; lateral path remains inactive. |
| Wheel-ground lateral stabilization | optional/limited | Wheel differential velocity probes do not directly create sustained roll torque in this morphology. |
