"""Stage 2D Sagittal LQR Controller.

Model-based LQR controller for sagittal (pitch) stabilization using identified linear dynamics.
Addresses unbounded wheel velocity and pitch divergence observed in Stage 2B/2C.

State vector: x = [pitch_x, pitch_rate_x, cp_error_y, com_vy, wheel_vel_mean]
Control: u = -K x (common wheel torque)
"""

import numpy as np
from scipy import linalg
from pathlib import Path


class Stage2DSagittalLQRController:
    """LQR controller for sagittal stabilization using identified dynamics."""

    # Predefined Q/R configurations
    CONFIGS = {
        'A': {
            'Q': np.diag([80.0, 10.0, 20.0, 5.0, 1.0]),
            'R': 1.0,
            'max_tau': 8.0,
            'description': 'Baseline LQR with moderate pitch/CP weighting',
        },
        'B': {
            'Q': np.diag([120.0, 20.0, 30.0, 8.0, 2.0]),
            'R': 1.0,
            'max_tau': 10.0,
            'description': 'Increased pitch/CP weighting, higher torque limit',
        },
        'C': {
            'Q': np.diag([160.0, 30.0, 40.0, 10.0, 4.0]),
            'R': 1.5,
            'max_tau': 12.0,
            'description': 'High pitch/CP weighting with control penalty',
        },
        'D': {
            'Q': np.diag([200.0, 40.0, 50.0, 15.0, 6.0]),
            'R': 2.0,
            'max_tau': 12.0,
            'description': 'Aggressive pitch/CP weighting with strong control penalty',
        },
    }

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        config: str = 'A',
        equilibrium_cp_y: float = 0.0,
    ):
        """Initialize LQR controller with identified dynamics.

        Args:
            A: (5, 5) discrete state transition matrix
            B: (5,) discrete input vector
            config: Configuration name ('A', 'B', 'C', 'D')
            equilibrium_cp_y: Equilibrium capture point Y position (m)
        """
        if config not in self.CONFIGS:
            raise ValueError(f"Unknown config '{config}'. Available: {list(self.CONFIGS.keys())}")

        self.A = A
        self.B = B.reshape(-1, 1)  # Ensure column vector
        self.config_name = config
        self.config = self.CONFIGS[config]
        self.equilibrium_cp_y = equilibrium_cp_y

        # Extract Q, R, max_tau from config
        self.Q = self.config['Q']
        self.R = np.array([[self.config['R']]])
        self.max_tau = self.config['max_tau']

        # Solve discrete-time algebraic Riccati equation
        self.P, self.K = self._solve_dare()

        print(f"[Stage2D LQR] Initialized with config '{config}':")
        print(f"  Description: {self.config['description']}")
        print(f"  Q diagonal: {np.diag(self.Q)}")
        print(f"  R: {self.R[0, 0]}")
        print(f"  Max tau: {self.max_tau} Nm")
        print(f"  LQR gain K: {self.K.ravel()}")

    @classmethod
    def from_identified_model(cls, model_path: str, config: str = 'A'):
        """Load identified model and create LQR controller.

        Args:
            model_path: Path to identified_model.npz from Phase 1
            config: Configuration name ('A', 'B', 'C', 'D')

        Returns:
            Stage2DSagittalLQRController instance
        """
        data = np.load(model_path)
        A = data['A']
        B = data['B']
        equilibrium_cp_y = float(data['equilibrium_cp_y'])

        return cls(A=A, B=B, config=config, equilibrium_cp_y=equilibrium_cp_y)

    def _solve_dare(self):
        """Solve discrete-time algebraic Riccati equation.

        Returns:
            P: Solution to DARE
            K: LQR gain matrix
        """
        # Solve DARE: A'PA - P - A'PB(R + B'PB)^{-1}B'PA + Q = 0
        P = linalg.solve_discrete_are(self.A, self.B, self.Q, self.R)

        # Compute LQR gain: K = (R + B'PB)^{-1}B'PA
        K = linalg.solve(
            self.R + self.B.T @ P @ self.B,
            self.B.T @ P @ self.A
        )

        return P, K

    def compute_wheel_torques(
        self,
        pitch_x: float,
        pitch_rate_x: float,
        cp_y: float,
        com_vy: float,
        wheel_vel_left: float,
        wheel_vel_right: float,
    ) -> tuple[np.ndarray, dict]:
        """Compute wheel torques using LQR control law.

        Args:
            pitch_x: Current pitch angle (rad)
            pitch_rate_x: Current pitch rate (rad/s)
            cp_y: Current capture point Y position (m)
            com_vy: Current CoM Y velocity (m/s)
            wheel_vel_left: Left wheel velocity (rad/s)
            wheel_vel_right: Right wheel velocity (rad/s)

        Returns:
            Tuple of (tau_wheel, diagnostics) where:
                - tau_wheel: (10,) torque vector, only [4,9] nonzero
                - diagnostics: dict with state, control, contributions, saturation
        """
        # Compute state vector
        cp_error_y = cp_y - self.equilibrium_cp_y
        wheel_vel_mean = 0.5 * (wheel_vel_left + wheel_vel_right)

        x = np.array([pitch_x, pitch_rate_x, cp_error_y, com_vy, wheel_vel_mean])

        # LQR control law: u = -K x
        u_raw = -self.K @ x
        u_raw_scalar = float(u_raw[0])

        # Clip to torque limit
        u_clipped = np.clip(u_raw_scalar, -self.max_tau, self.max_tau)

        # Build full torque vector (only wheel joints nonzero)
        tau_wheel = np.zeros(10)
        tau_wheel[4] = u_clipped  # l_wheel
        tau_wheel[9] = u_clipped  # r_wheel

        # Compute individual state contributions to control
        K_flat = self.K.ravel()
        contributions = {
            'pitch_x': -K_flat[0] * pitch_x,
            'pitch_rate_x': -K_flat[1] * pitch_rate_x,
            'cp_error_y': -K_flat[2] * cp_error_y,
            'com_vy': -K_flat[3] * com_vy,
            'wheel_vel_mean': -K_flat[4] * wheel_vel_mean,
        }

        # Diagnostics
        saturated = abs(u_raw_scalar) > self.max_tau

        diagnostics = {
            # State
            'pitch_x': float(pitch_x),
            'pitch_rate_x': float(pitch_rate_x),
            'cp_y': float(cp_y),
            'cp_error_y': float(cp_error_y),
            'com_vy': float(com_vy),
            'wheel_vel_left': float(wheel_vel_left),
            'wheel_vel_right': float(wheel_vel_right),
            'wheel_vel_mean': float(wheel_vel_mean),
            # Control
            'u_raw': float(u_raw_scalar),
            'u_clipped': float(u_clipped),
            'saturated': bool(saturated),
            # LQR gain
            'K': K_flat.tolist(),
            # State contributions
            'contrib_pitch_x': float(contributions['pitch_x']),
            'contrib_pitch_rate_x': float(contributions['pitch_rate_x']),
            'contrib_cp_error_y': float(contributions['cp_error_y']),
            'contrib_com_vy': float(contributions['com_vy']),
            'contrib_wheel_vel_mean': float(contributions['wheel_vel_mean']),
            # Config
            'config': self.config_name,
            'max_tau': self.max_tau,
        }

        return tau_wheel, diagnostics

    def get_closed_loop_eigenvalues(self):
        """Compute closed-loop eigenvalues: eig(A - B K).

        Returns:
            eigenvalues: Complex array of closed-loop eigenvalues
        """
        A_cl = self.A - self.B @ self.K
        eigenvalues = np.linalg.eigvals(A_cl)
        return eigenvalues

    def check_stability(self):
        """Check if closed-loop system is stable.

        Returns:
            is_stable: True if all eigenvalues are inside unit circle
            max_magnitude: Maximum eigenvalue magnitude
        """
        eigenvalues = self.get_closed_loop_eigenvalues()
        magnitudes = np.abs(eigenvalues)
        max_magnitude = np.max(magnitudes)
        is_stable = max_magnitude < 1.0

        return is_stable, max_magnitude

    def print_analysis(self):
        """Print detailed controller analysis."""
        print(f"\n{'='*60}")
        print(f"Stage2D LQR Controller Analysis - Config {self.config_name}")
        print(f"{'='*60}")

        print(f"\nConfiguration:")
        print(f"  {self.config['description']}")
        print(f"  Q diagonal: {np.diag(self.Q)}")
        print(f"  R: {self.R[0, 0]}")
        print(f"  Max tau: {self.max_tau} Nm")

        print(f"\nLQR Gain K:")
        K_flat = self.K.ravel()
        state_names = ['pitch_x', 'pitch_rate_x', 'cp_error_y', 'com_vy', 'wheel_vel_mean']
        for i, name in enumerate(state_names):
            print(f"  K[{i}] ({name:16s}): {K_flat[i]:+.4f}")

        print(f"\nClosed-loop eigenvalues:")
        eigenvalues = self.get_closed_loop_eigenvalues()
        for i, eig in enumerate(eigenvalues):
            mag = abs(eig)
            print(f"  λ{i+1}: {eig.real:+.4f} {eig.imag:+.4f}j  (|λ| = {mag:.4f})")

        is_stable, max_mag = self.check_stability()
        if is_stable:
            print(f"\n✓ System is STABLE (max |λ| = {max_mag:.4f} < 1.0)")
        else:
            print(f"\n✗ System is UNSTABLE (max |λ| = {max_mag:.4f} >= 1.0)")

        # Estimate settling time (rough approximation)
        if is_stable and max_mag > 0:
            # Discrete settling time: n ≈ -log(0.02) / log(max_mag)
            # Convert to continuous time: t ≈ n * dt
            dt = 0.002  # Control timestep
            n_steps = -np.log(0.02) / np.log(max_mag) if max_mag < 1.0 else np.inf
            settling_time = n_steps * dt
            print(f"  Estimated settling time (2%): {settling_time:.2f} s ({int(n_steps)} steps)")


def test_controller_from_model(model_path: str):
    """Test all LQR configurations with an identified model.

    Args:
        model_path: Path to identified_model.npz from Phase 1
    """
    print("="*60)
    print("Testing Stage2D LQR Controller Configurations")
    print("="*60)

    for config_name in ['A', 'B', 'C', 'D']:
        controller = Stage2DSagittalLQRController.from_identified_model(
            model_path=model_path,
            config=config_name,
        )
        controller.print_analysis()

        # Test zero state
        tau, diag = controller.compute_wheel_torques(
            pitch_x=0.0,
            pitch_rate_x=0.0,
            cp_y=controller.equilibrium_cp_y,
            com_vy=0.0,
            wheel_vel_left=0.0,
            wheel_vel_right=0.0,
        )
        assert np.allclose(tau, 0.0), f"Config {config_name}: Zero state should give zero torque"

        # Test positive pitch perturbation
        tau, diag = controller.compute_wheel_torques(
            pitch_x=0.1,  # ~5.7 deg forward tilt
            pitch_rate_x=0.0,
            cp_y=controller.equilibrium_cp_y,
            com_vy=0.0,
            wheel_vel_left=0.0,
            wheel_vel_right=0.0,
        )

        print(f"\n  Test: pitch_x = +0.1 rad (~5.7 deg)")
        print(f"    u_raw: {diag['u_raw']:+.3f} Nm")
        print(f"    u_clipped: {diag['u_clipped']:+.3f} Nm")
        print(f"    Saturated: {diag['saturated']}")

        # Positive pitch should give restoring torque
        # Sign depends on identified B[0], but should be consistent
        print(f"    → Wheel torque sign: {'positive (backward)' if diag['u_clipped'] > 0 else 'negative (forward)'}")

        print()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python stage2d_sagittal_lqr_controller.py <path_to_identified_model.npz>")
        sys.exit(1)

    model_path = sys.argv[1]
    test_controller_from_model(model_path)
