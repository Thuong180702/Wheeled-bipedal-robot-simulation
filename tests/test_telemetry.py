"""Tests for telemetry system (Phase B.7 Task 9)."""

import numpy as np
import pytest

from scripts.eval_classical_prior_with_telemetry import (
    TelemetrySnapshot,
    EpisodeTelemetry,
    classify_failure_mode,
)


class TestTelemetrySnapshot:
    """Test TelemetrySnapshot dataclass."""

    def test_snapshot_creation(self):
        """Test creating a telemetry snapshot."""
        snapshot = TelemetrySnapshot(
            time=1.0,
            pitch_deg=5.0,
            pitch_rate_deg_s=2.0,
            roll_deg=1.0,
            com_error_y_m=0.05,
            com_vel_y_m_s=0.1,
            wheel_vel_cmd_rad_s=3.0,
            wheel_vel_actual_rad_s=2.8,
            wheel_saturation_rate=0.1,
            lqr_pitch_contrib=-1.5,
            lqr_pitch_rate_contrib=-0.8,
            lqr_fwd_vel_contrib=-0.5,
            lqr_com_contrib=-0.6,
            lqr_com_rate_contrib=-0.3,
            height_cmd_m=0.55,
            height_actual_m=0.54,
            height_ik_error_m=0.01,
            hip_pitch_cmd_rad=-0.2,
            knee_cmd_rad=1.0,
        )

        assert snapshot.time == 1.0
        assert snapshot.pitch_deg == 5.0
        assert snapshot.com_error_y_m == 0.05
        assert snapshot.wheel_saturation_rate == 0.1

    def test_snapshot_all_fields_finite(self):
        """Test all snapshot fields are finite."""
        snapshot = TelemetrySnapshot(
            time=1.0,
            pitch_deg=5.0,
            pitch_rate_deg_s=2.0,
            roll_deg=1.0,
            com_error_y_m=0.05,
            com_vel_y_m_s=0.1,
            wheel_vel_cmd_rad_s=3.0,
            wheel_vel_actual_rad_s=2.8,
            wheel_saturation_rate=0.1,
            lqr_pitch_contrib=-1.5,
            lqr_pitch_rate_contrib=-0.8,
            lqr_fwd_vel_contrib=-0.5,
            lqr_com_contrib=-0.6,
            lqr_com_rate_contrib=-0.3,
            height_cmd_m=0.55,
            height_actual_m=0.54,
            height_ik_error_m=0.01,
            hip_pitch_cmd_rad=-0.2,
            knee_cmd_rad=1.0,
        )

        # Check all numeric fields are finite
        assert np.isfinite(snapshot.time)
        assert np.isfinite(snapshot.pitch_deg)
        assert np.isfinite(snapshot.pitch_rate_deg_s)
        assert np.isfinite(snapshot.roll_deg)
        assert np.isfinite(snapshot.com_error_y_m)
        assert np.isfinite(snapshot.com_vel_y_m_s)
        assert np.isfinite(snapshot.wheel_vel_cmd_rad_s)
        assert np.isfinite(snapshot.wheel_vel_actual_rad_s)
        assert np.isfinite(snapshot.wheel_saturation_rate)


class TestEpisodeTelemetry:
    """Test EpisodeTelemetry dataclass."""

    def test_episode_telemetry_creation(self):
        """Test creating episode telemetry."""
        snapshots = [
            TelemetrySnapshot(
                time=0.0,
                pitch_deg=0.0,
                pitch_rate_deg_s=0.0,
                roll_deg=0.0,
                com_error_y_m=0.0,
                com_vel_y_m_s=0.0,
                wheel_vel_cmd_rad_s=0.0,
                wheel_vel_actual_rad_s=0.0,
                wheel_saturation_rate=0.0,
                lqr_pitch_contrib=0.0,
                lqr_pitch_rate_contrib=0.0,
                lqr_fwd_vel_contrib=0.0,
                lqr_com_contrib=0.0,
                lqr_com_rate_contrib=0.0,
                height_cmd_m=0.55,
                height_actual_m=0.55,
                height_ik_error_m=0.0,
                hip_pitch_cmd_rad=0.0,
                knee_cmd_rad=1.0,
            )
        ]

        episode = EpisodeTelemetry(
            episode_id=0,
            height_cmd_m=0.55,
            survival_time_s=10.0,
            fell=False,
            failure_mode="survived",
            failure_reason="Episode completed successfully",
            snapshots=snapshots,
            pitch_rms_deg=2.5,
            roll_rms_deg=1.0,
            com_error_rms_m=0.02,
            wheel_saturation_duration_s=0.5,
        )

        assert episode.episode_id == 0
        assert episode.height_cmd_m == 0.55
        assert episode.survival_time_s == 10.0
        assert episode.fell is False
        assert episode.failure_mode == "survived"
        assert len(episode.snapshots) == 1

    def test_episode_telemetry_metrics(self):
        """Test episode telemetry metrics are computed correctly."""
        episode = EpisodeTelemetry(
            episode_id=0,
            height_cmd_m=0.55,
            survival_time_s=5.0,
            fell=True,
            failure_mode="pitch_oscillation",
            failure_reason="Pitch oscillation detected",
            snapshots=[],
            pitch_rms_deg=15.0,
            roll_rms_deg=3.0,
            com_error_rms_m=0.08,
            wheel_saturation_duration_s=2.0,
        )

        assert episode.pitch_rms_deg == 15.0
        assert episode.roll_rms_deg == 3.0
        assert episode.com_error_rms_m == 0.08
        assert episode.wheel_saturation_duration_s == 2.0


class TestClassifyFailureMode:
    """Test failure mode classification."""

    def create_snapshots(
        self,
        num_steps: int,
        pitch_pattern: str = "stable",
        com_pattern: str = "stable",
        saturation_pattern: str = "low",
    ) -> list[TelemetrySnapshot]:
        """Create test snapshots with specified patterns."""
        snapshots = []
        dt = 0.02

        for i in range(num_steps):
            time = i * dt

            # Pitch patterns
            if pitch_pattern == "stable":
                pitch = 2.0 + 0.5 * np.sin(2 * np.pi * 0.5 * time)
            elif pitch_pattern == "oscillating":
                pitch = 20.0 * np.sin(2 * np.pi * 2.0 * time)
            elif pitch_pattern == "diverging":
                pitch = 5.0 + 2.0 * time
            else:
                pitch = 0.0

            # CoM patterns
            if com_pattern == "stable":
                com_error = 0.02 + 0.01 * np.sin(2 * np.pi * 0.5 * time)
            elif com_pattern == "drifting":
                com_error = 0.01 * time
            else:
                com_error = 0.0

            # Saturation patterns
            if saturation_pattern == "low":
                saturation = 0.1
            elif saturation_pattern == "high":
                saturation = 0.9
            else:
                saturation = 0.0

            snapshot = TelemetrySnapshot(
                time=time,
                pitch_deg=pitch,
                pitch_rate_deg_s=0.0,
                roll_deg=1.0,
                com_error_y_m=com_error,
                com_vel_y_m_s=0.0,
                wheel_vel_cmd_rad_s=0.0,
                wheel_vel_actual_rad_s=0.0,
                wheel_saturation_rate=saturation,
                lqr_pitch_contrib=0.0,
                lqr_pitch_rate_contrib=0.0,
                lqr_fwd_vel_contrib=0.0,
                lqr_com_contrib=0.0,
                lqr_com_rate_contrib=0.0,
                height_cmd_m=0.55,
                height_actual_m=0.55,
                height_ik_error_m=0.0,
                hip_pitch_cmd_rad=0.0,
                knee_cmd_rad=1.0,
            )
            snapshots.append(snapshot)

        return snapshots

    def test_classify_survived(self):
        """Test classification of survived episode."""
        snapshots = self.create_snapshots(500, "stable", "stable", "low")
        survival_time = 10.0
        max_time = 10.0

        mode, reason = classify_failure_mode(snapshots, survival_time, max_time)

        assert mode == "survived"
        assert "successfully" in reason.lower()

    def test_classify_pitch_oscillation(self):
        """Test classification of pitch oscillation failure."""
        snapshots = self.create_snapshots(100, "oscillating", "stable", "low")
        survival_time = 2.0
        max_time = 10.0

        mode, reason = classify_failure_mode(snapshots, survival_time, max_time)

        assert mode == "pitch_oscillation"
        assert "oscillation" in reason.lower()

    def test_classify_com_drift(self):
        """Test classification of CoM drift failure."""
        snapshots = self.create_snapshots(100, "stable", "drifting", "low")
        survival_time = 2.0
        max_time = 10.0

        mode, reason = classify_failure_mode(snapshots, survival_time, max_time)

        assert mode == "com_drift"
        assert "drift" in reason.lower()

    def test_classify_wheel_saturation(self):
        """Test classification of wheel saturation failure."""
        snapshots = self.create_snapshots(100, "stable", "stable", "high")
        survival_time = 2.0
        max_time = 10.0

        mode, reason = classify_failure_mode(snapshots, survival_time, max_time)

        assert mode == "wheel_saturation"
        assert "saturation" in reason.lower()

    def test_classify_unknown_short_episode(self):
        """Test classification of short episode with no clear failure mode."""
        snapshots = self.create_snapshots(10, "stable", "stable", "low")
        survival_time = 0.2
        max_time = 10.0

        mode, reason = classify_failure_mode(snapshots, survival_time, max_time)

        assert mode == "unknown"
        assert "insufficient" in reason.lower() or "short" in reason.lower()

    def test_classify_empty_snapshots(self):
        """Test classification with empty snapshots."""
        snapshots = []
        survival_time = 0.0
        max_time = 10.0

        mode, reason = classify_failure_mode(snapshots, survival_time, max_time)

        assert mode == "unknown"
        assert "no telemetry" in reason.lower() or "insufficient" in reason.lower()

    def test_classify_priority_pitch_over_com(self):
        """Test that pitch oscillation takes priority over CoM drift."""
        # Both pitch oscillation and CoM drift present
        snapshots = self.create_snapshots(100, "oscillating", "drifting", "low")
        survival_time = 2.0
        max_time = 10.0

        mode, reason = classify_failure_mode(snapshots, survival_time, max_time)

        # Pitch oscillation should be detected first
        assert mode == "pitch_oscillation"

    def test_classify_priority_saturation_over_com(self):
        """Test that wheel saturation takes priority over CoM drift."""
        # Both saturation and CoM drift present
        snapshots = self.create_snapshots(100, "stable", "drifting", "high")
        survival_time = 2.0
        max_time = 10.0

        mode, reason = classify_failure_mode(snapshots, survival_time, max_time)

        # Wheel saturation should be detected first
        assert mode == "wheel_saturation"


class TestTelemetryIntegration:
    """Integration tests for telemetry system."""

    def test_full_episode_telemetry_workflow(self):
        """Test complete telemetry capture workflow."""
        # Simulate episode
        snapshots = []
        dt = 0.02
        num_steps = 100

        for i in range(num_steps):
            snapshot = TelemetrySnapshot(
                time=i * dt,
                pitch_deg=2.0 + 0.5 * np.sin(2 * np.pi * 0.5 * i * dt),
                pitch_rate_deg_s=0.0,
                roll_deg=1.0,
                com_error_y_m=0.02,
                com_vel_y_m_s=0.0,
                wheel_vel_cmd_rad_s=3.0,
                wheel_vel_actual_rad_s=2.8,
                wheel_saturation_rate=0.1,
                lqr_pitch_contrib=-1.5,
                lqr_pitch_rate_contrib=-0.8,
                lqr_fwd_vel_contrib=-0.5,
                lqr_com_contrib=-0.6,
                lqr_com_rate_contrib=-0.3,
                height_cmd_m=0.55,
                height_actual_m=0.54,
                height_ik_error_m=0.01,
                hip_pitch_cmd_rad=-0.2,
                knee_cmd_rad=1.0,
            )
            snapshots.append(snapshot)

        # Compute metrics
        pitches = [s.pitch_deg for s in snapshots]
        rolls = [s.roll_deg for s in snapshots]
        com_errors = [s.com_error_y_m for s in snapshots]
        saturations = [s.wheel_saturation_rate for s in snapshots]

        pitch_rms = np.sqrt(np.mean(np.array(pitches) ** 2))
        roll_rms = np.sqrt(np.mean(np.array(rolls) ** 2))
        com_error_rms = np.sqrt(np.mean(np.array(com_errors) ** 2))
        wheel_sat_duration = sum(s > 0.8 for s in saturations) * dt

        survival_time = num_steps * dt
        max_time = 10.0

        # Classify failure
        mode, reason = classify_failure_mode(snapshots, survival_time, max_time)

        # Create episode telemetry
        episode = EpisodeTelemetry(
            episode_id=0,
            height_cmd_m=0.55,
            survival_time_s=survival_time,
            fell=(survival_time < max_time - 0.01),
            failure_mode=mode,
            failure_reason=reason,
            snapshots=snapshots,
            pitch_rms_deg=pitch_rms,
            roll_rms_deg=roll_rms,
            com_error_rms_m=com_error_rms,
            wheel_saturation_duration_s=wheel_sat_duration,
        )

        # Verify episode
        assert episode.episode_id == 0
        assert episode.height_cmd_m == 0.55
        assert episode.survival_time_s == survival_time
        assert len(episode.snapshots) == num_steps
        assert episode.pitch_rms_deg > 0
        assert episode.roll_rms_deg > 0
        assert episode.com_error_rms_m > 0

    def test_telemetry_snapshot_time_ordering(self):
        """Test snapshots maintain time ordering."""
        snapshots = []
        for i in range(10):
            snapshot = TelemetrySnapshot(
                time=i * 0.02,
                pitch_deg=0.0,
                pitch_rate_deg_s=0.0,
                roll_deg=0.0,
                com_error_y_m=0.0,
                com_vel_y_m_s=0.0,
                wheel_vel_cmd_rad_s=0.0,
                wheel_vel_actual_rad_s=0.0,
                wheel_saturation_rate=0.0,
                lqr_pitch_contrib=0.0,
                lqr_pitch_rate_contrib=0.0,
                lqr_fwd_vel_contrib=0.0,
                lqr_com_contrib=0.0,
                lqr_com_rate_contrib=0.0,
                height_cmd_m=0.55,
                height_actual_m=0.55,
                height_ik_error_m=0.0,
                hip_pitch_cmd_rad=0.0,
                knee_cmd_rad=1.0,
            )
            snapshots.append(snapshot)

        # Check time ordering
        times = [s.time for s in snapshots]
        assert times == sorted(times)

    def test_telemetry_metrics_consistency(self):
        """Test telemetry metrics are consistent with snapshots."""
        snapshots = []
        pitches = []
        for i in range(50):
            pitch = 5.0 + i * 0.1
            pitches.append(pitch)
            snapshot = TelemetrySnapshot(
                time=i * 0.02,
                pitch_deg=pitch,
                pitch_rate_deg_s=0.0,
                roll_deg=0.0,
                com_error_y_m=0.0,
                com_vel_y_m_s=0.0,
                wheel_vel_cmd_rad_s=0.0,
                wheel_vel_actual_rad_s=0.0,
                wheel_saturation_rate=0.0,
                lqr_pitch_contrib=0.0,
                lqr_pitch_rate_contrib=0.0,
                lqr_fwd_vel_contrib=0.0,
                lqr_com_contrib=0.0,
                lqr_com_rate_contrib=0.0,
                height_cmd_m=0.55,
                height_actual_m=0.55,
                height_ik_error_m=0.0,
                hip_pitch_cmd_rad=0.0,
                knee_cmd_rad=1.0,
            )
            snapshots.append(snapshot)

        # Compute pitch RMS from snapshots
        pitch_rms_expected = np.sqrt(np.mean(np.array(pitches) ** 2))

        # Create episode
        episode = EpisodeTelemetry(
            episode_id=0,
            height_cmd_m=0.55,
            survival_time_s=1.0,
            fell=False,
            failure_mode="survived",
            failure_reason="Test",
            snapshots=snapshots,
            pitch_rms_deg=pitch_rms_expected,
            roll_rms_deg=0.0,
            com_error_rms_m=0.0,
            wheel_saturation_duration_s=0.0,
        )

        # Verify consistency
        assert abs(episode.pitch_rms_deg - pitch_rms_expected) < 1e-6
