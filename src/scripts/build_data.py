"""Script that builds a Scandinavian NLI dataset."""

import hydra
from omegaconf import DictConfig

from scandinli.build_data import build_data


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    """Build an NLI dataset used for training.

    Args:
        config (DictConfig):
            The Hydra configuration.
    """
    build_data(config)


if __name__ == "__main__":
    main()
