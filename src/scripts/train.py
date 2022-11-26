"""Script trains an NLI model."""

import hydra
from omegaconf import DictConfig

from scandinli.train import train


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    """Trains an NLI model.

    Args:
        config (DictConfig):
            The Hydra configuration.
    """
    train(config)


if __name__ == "__main__":
    main()
