"""Tests for capture point estimator."""

import jax.numpy as jnp
import pytest
from wheeled_biped.controllers.capture_point_estimator import (
    CapturePointEstimator,
    CapturePointEstimatorConfig,
)
from wheeled_biped.controllers.centroidal_state_estimator import CentroidalState


def test_capture_point_estimator_creation():
    """Test CapturePointEstimator can be created."""
    config = CapturePointEstimatorConfig(
        gravity=9.81,
    )
    estimator = CapturePointEstimator(config)

    assert estimator.config.gravity == 9.81
