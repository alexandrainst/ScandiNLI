"""Session-wide fixtures for tests."""

from collections.abc import Generator

import pytest
from hydra import compose, initialize
from omegaconf import DictConfig

# Initialise Hydra
initialize(config_path="../config", version_base=None)


@pytest.fixture(scope="session")
def config() -> Generator[DictConfig, None, None]:
    """The Hydra config object.

    Returns:
        The Hydra config object.
    """
    yield compose(config_name="config")
