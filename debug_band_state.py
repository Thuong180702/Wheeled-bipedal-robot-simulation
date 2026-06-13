from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
    APCR1ND_T5_BAND_LIMITED_BALANCED,
    SagittalVelocityDampedBalanceController,
)
import jax.numpy as jnp

ctrl = SagittalVelocityDampedBalanceController(
    authority_schedule=APCR1ND_T5_BAND_LIMITED_BALANCED
)

for error in [0.02, 0.049, 0.050, 0.051, 0.079, 0.080, 0.081]:
    tau, diag = ctrl.compute(
        pitch_x_rad=jnp.float32(0.05),
        pitch_rate_x_rad_s=jnp.float32(0.01),
        sagittal_velocity_m_s=jnp.float32(0.1),
        wheel_vel_left_rad_s=jnp.float32(1.0),
        wheel_vel_right_rad_s=jnp.float32(1.0),
        sagittal_position_error_m=jnp.float32(error),
        com_z_m=jnp.float32(0.35),
    )
    print(f'error={error:.3f}: band_state={diag["tuned_band_state"]}, band_id={diag["tuned_band_state_id"]}, abs_error={diag["tuned_abs_error"]:.6f}')
