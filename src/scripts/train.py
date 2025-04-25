"""Script trains an NLI model."""

import hydra
from omegaconf import DictConfig

from scandinli.data import build_data
from scandinli.training import train


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    """Trains an NLI model.

    Args:
        config:
            The Hydra configuration.
    """
    build_data(config=config)
    train(config=config)


if __name__ == "__main__":
    main()
