# Step 5.16 WBC/VMC Formulation

The controller computes a desired roll torque, lateral force, vertical support force, and left/right vertical force redistribution. Because the deployed interface is position-PID for legs and velocity-PID for wheels, this is not direct torque WBC. The desired wrench is mapped to bounded normalized offsets for hip roll, hip pitch, knee, and optionally differential wheel velocity.

`tau_roll_des = -k_roll * roll_error - k_roll_rate * roll_rate`

`Fy_des = -k_com_y * y_error - k_com_y_rate * y_rate`

`Fz_des = m*g - k_height * height_error - k_height_rate * height_rate`

`delta_Fz_des = clamp(tau_roll_des / support_width + force_balance, +/- max_delta_fz)`

`Fz_left_des = max(0, 0.5 * Fz_des + delta_Fz_des)` and `Fz_right_des = max(0, 0.5 * Fz_des - delta_Fz_des)`.
