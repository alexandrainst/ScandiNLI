"""Script evaluates an NLI model."""

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig

from scandinli.data import build_data
from scandinli.evaluation import evaluate_litellm

load_dotenv()


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    """Evaluates an NLI model.

    Args:
        config:
            The Hydra configuration.
    """
    build_data(config=config)
    evaluate_litellm(config=config)


if __name__ == "__main__":
    main()
