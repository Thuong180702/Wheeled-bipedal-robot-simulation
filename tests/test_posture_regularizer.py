# tests/test_posture_regularizer.py
import jax.numpy as jnp
import pytest
from wheeled_biped.controllers.posture_regularizer import (
    PostureRegularizer,
    PostureRegularizerConfig,
)


def test_posture_regularizer_creation():
    """Test PostureRegularizer can be created with config."""
    config = PostureRegularizerConfig(
        k_posture=2.0,
        posture_authority_budget=0.2,
    )
    regularizer = PostureRegularizer(config)

    assert regularizer.config.k_posture == 2.0
    assert regularizer.config.posture_authority_budget == 0.2
