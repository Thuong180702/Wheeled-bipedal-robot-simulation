"""Pitch rate consistency estimator with finite-difference validation.

Detects and corrects pitch rate measurement artifacts by comparing measured
angular velocity against finite-difference derivative of pitch angle. When
measured pitch_rate has inconsistent sign with finite-difference rate, uses
the FD rate instead to prevent damping sign flips in balance controllers.

Root cause: Step E transient at step 1236 showed pitch angle increasing while
measured pitch_rate flipped negative, causing sagittal damping term to flip
sign and trigger wheel acceleration spike.
"""

from dataclasses import dataclass


@dataclass
class PitchRateEstimate:
    """Output of pitch rate consistency estimator."""
    pitch_rate_corrected: float
    pitch_rate_measured: float
    pitch_rate_fd: float
    consistency_error: float
    sign_mismatch: bool
    source_used: str  # "measured" | "finite_difference" | "blended"
    filter_alpha: float


class PitchRateConsistencyEstimator:
    """Stateful pitch rate estimator with finite-difference consistency check.

    Maintains previous pitch angle for FD computation and applies low-pass
    filtering to corrected rate to avoid injecting noise into damping terms.
    """

    def __init__(
        self,
        dt: float,
        min_rate_for_sign_check: float = 0.01,
        filter_alpha: float = 0.3,
    ):
        """Initialize pitch rate consistency estimator.

        Args:
            dt: Control timestep (s).
            min_rate_for_sign_check: Minimum absolute rate (rad/s) to check sign.
                Below this threshold, rates are considered near-zero and sign
                mismatch is not flagged.
            filter_alpha: Low-pass filter coefficient for corrected rate.
                corrected[t] = alpha * corrected[t-1] + (1-alpha) * selected[t]
                Higher alpha = more filtering, more lag. Range [0, 1].
                Default 0.3 provides moderate smoothing without excessive lag.
        """
        if not 0.0 < filter_alpha <= 1.0:
            raise ValueError(f"filter_alpha must be in (0, 1], got {filter_alpha}")

        self.dt = dt
        self.min_rate_for_sign_check = min_rate_for_sign_check
        self.filter_alpha = filter_alpha

        # State
        self.prev_pitch_x: float | None = None
        self.prev_corrected_rate: float = 0.0

    def reset(self):
        """Reset estimator state (call at episode start)."""
        self.prev_pitch_x = None
        self.prev_corrected_rate = 0.0

    def estimate(
        self,
        pitch_x: float,
        pitch_rate_measured: float,
    ) -> PitchRateEstimate:
        """Estimate corrected pitch rate with consistency check.

        Args:
            pitch_x: Current pitch angle (rad).
            pitch_rate_measured: Measured pitch rate from qvel (rad/s).

        Returns:
            PitchRateEstimate with corrected rate and diagnostics.
        """
        # Compute finite-difference pitch rate
        if self.prev_pitch_x is not None:
            pitch_rate_fd = (pitch_x - self.prev_pitch_x) / self.dt
        else:
            # First step: no previous pitch, use measured rate
            pitch_rate_fd = pitch_rate_measured

        # Check sign consistency
        abs_measured = abs(pitch_rate_measured)
        abs_fd = abs(pitch_rate_fd)

        # Only check sign when both rates are above threshold
        if abs_measured > self.min_rate_for_sign_check and abs_fd > self.min_rate_for_sign_check:
            sign_measured = 1.0 if pitch_rate_measured > 0 else -1.0
            sign_fd = 1.0 if pitch_rate_fd > 0 else -1.0
            sign_mismatch = (sign_measured != sign_fd)
        else:
            sign_mismatch = False

        # Select rate source
        if sign_mismatch:
            # Use FD rate when signs are inconsistent
            selected_rate = pitch_rate_fd
            source_used = "finite_difference"
        else:
            # Use measured rate when consistent or near-zero
            selected_rate = pitch_rate_measured
            source_used = "measured"

        # Apply low-pass filter to corrected rate
        pitch_rate_corrected = (
            self.filter_alpha * self.prev_corrected_rate
            + (1.0 - self.filter_alpha) * selected_rate
        )

        # Compute consistency error (for diagnostics)
        consistency_error = pitch_rate_measured - pitch_rate_fd

        # Update state
        self.prev_pitch_x = pitch_x
        self.prev_corrected_rate = pitch_rate_corrected

        return PitchRateEstimate(
            pitch_rate_corrected=pitch_rate_corrected,
            pitch_rate_measured=pitch_rate_measured,
            pitch_rate_fd=pitch_rate_fd,
            consistency_error=consistency_error,
            sign_mismatch=sign_mismatch,
            source_used=source_used,
            filter_alpha=self.filter_alpha,
        )
