"""Script evaluates an NLI model."""

import hydra
from omegaconf import DictConfig

from scandinli.evaluate import evaluate


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    """Trains an NLI model.

    Args:
        config (DictConfig):
            The Hydra configuration.
    """
    evaluate(config)


if __name__ == "__main__":
    main()
