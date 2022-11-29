"""Build an NLI dataset used for training."""

import logging
from pathlib import Path
from typing import Dict, List

import hydra
from datasets import disable_progress_bar
from datasets.arrow_dataset import Dataset
from datasets.builder import DatasetGenerationError
from datasets.combine import concatenate_datasets, interleave_datasets
from datasets.dataset_dict import DatasetDict
from datasets.features import ClassLabel
from datasets.load import load_dataset
from omegaconf import DictConfig
from tqdm.auto import tqdm

# Ignore loggers from `datasets`
logging.getLogger("datasets").setLevel(logging.ERROR)


# Disable the `datasets` progress bar
disable_progress_bar()


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def build_data(config: DictConfig) -> None:
    """Build an NLI dataset with a training, validation and test split.

    Args:
        config (DictConfig):
            The Hydra configuration.
    """
    # Define data directory
    final_dir = Path(config.dirs.data) / config.dirs.final

    # Build the DanFever dataset
    build_danfever_with_splits(config)

    # Build the three splits
    train = build_training_data(config)
    val = build_validation_data(config)
    test = build_test_data(config)

    # Collect the splits into a DatasetDict
    dataset = DatasetDict(dict(train=train, val=val, test=test))

    # Store the dataset
    dataset.save_to_disk(final_dir / "ScandiNLI")


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def build_training_data(config: DictConfig) -> Dataset:
    """Build an NLI dataset used for training.

    Args:
        config (DictConfig):
            The Hydra configuration.

    Returns:
        Dataset:
            The training dataset.
    """

    # Define data directory
    raw_dir = Path(config.dirs.data) / config.dirs.raw

    # Initialise the lists of all datasets and their proportions
    all_datasets: List[Dataset] = list()
    all_proportions: List[float] = list()

    # Load the Danish dataset
    if config.dataset.train_datasets.da:
        da_dataset = build_dataset_for_single_language(
            dataset_configs=config.dataset.train_datasets.da,
            cache_dir=raw_dir,
            seed=config.seed,
        )
        all_datasets.append(da_dataset)
        all_proportions.append(config.dataset.dataset_proportions.da)

    # Load the Swedish dataset
    if config.dataset.train_datasets.sv:
        sv_dataset = build_dataset_for_single_language(
            dataset_configs=config.dataset.train_datasets.sv,
            cache_dir=raw_dir,
            seed=config.seed,
        )
        all_datasets.append(sv_dataset)
        all_proportions.append(config.dataset.dataset_proportions.sv)

    # Load the Norwegian dataset
    if config.dataset.train_datasets.nb:
        nb_dataset = build_dataset_for_single_language(
            dataset_configs=config.dataset.train_datasets.nb,
            cache_dir=raw_dir,
            seed=config.seed,
        )
        all_datasets.append(nb_dataset)
        all_proportions.append(config.dataset.dataset_proportions.nb)

    # Interleave the Danish, Swedish and Norwegian datasets with the given proportions
    dataset = interleave_datasets(datasets=all_datasets, probabilities=all_proportions)

    # Return the dataset
    return dataset


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def build_validation_data(config: DictConfig) -> Dataset:
    """Build a validation dataset for NLI evaluation.

    Args:
        config (DictConfig):
            The Hydra configuration.

    Returns:
        Dataset:
            The validation dataset.
    """
    # Define data directory
    raw_dir = Path(config.dirs.data) / config.dirs.raw

    # Load the datasets for each language
    da_dataset = build_dataset_for_single_language(
        dataset_configs=config.dataset.val_datasets.da,
        cache_dir=raw_dir,
        seed=config.seed,
    )
    sv_dataset = build_dataset_for_single_language(
        dataset_configs=config.dataset.val_datasets.sv,
        cache_dir=raw_dir,
        seed=config.seed,
    )
    nb_dataset = build_dataset_for_single_language(
        dataset_configs=config.dataset.val_datasets.nb,
        cache_dir=raw_dir,
        seed=config.seed,
    )

    # Get the proportions each of the languages should have
    proportions = [
        config.dataset.dataset_proportions.da,
        config.dataset.dataset_proportions.sv,
        config.dataset.dataset_proportions.nb,
    ]

    # Interleave the Danish, Swedish and Norwegian datasets with the given proportions
    dataset = interleave_datasets(
        datasets=[da_dataset, sv_dataset, nb_dataset],
        probabilities=proportions,
    )

    # Return the dataset
    return dataset


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def build_test_data(config: DictConfig) -> Dataset:
    """Build a test dataset for NLI evaluation.

    Args:
        config (DictConfig):
            The Hydra configuration.

    Returns:
        Dataset:
            The test dataset.
    """
    # Define data directory
    raw_dir = Path(config.dirs.data) / config.dirs.raw

    # Load the datasets for each language
    da_dataset = build_dataset_for_single_language(
        dataset_configs=config.dataset.test_datasets.da,
        cache_dir=raw_dir,
        seed=config.seed,
    )
    sv_dataset = build_dataset_for_single_language(
        dataset_configs=config.dataset.test_datasets.sv,
        cache_dir=raw_dir,
        seed=config.seed,
    )
    nb_dataset = build_dataset_for_single_language(
        dataset_configs=config.dataset.test_datasets.nb,
        cache_dir=raw_dir,
        seed=config.seed,
    )

    # Get the proportions each of the languages should have
    proportions = [
        config.dataset.dataset_proportions.da,
        config.dataset.dataset_proportions.sv,
        config.dataset.dataset_proportions.nb,
    ]

    # Interleave the Danish, Swedish and Norwegian datasets with the given proportions
    dataset = interleave_datasets(
        datasets=[da_dataset, sv_dataset, nb_dataset],
        probabilities=proportions,
    )

    # Return the dataset
    return dataset


def build_dataset_for_single_language(
    dataset_configs: List[Dict[str, str]],
    cache_dir: str,
    seed: int,
) -> Dataset:
    """Build a dataset for a single language.

    Args:
        dataset_configs (list of Dict[str, str]):
            The dataset configurations.
        cache_dir (str):
            The directory to cache the dataset in.
        seed (int):
            The seed to use for shuffling the dataset.

    Returns:
        Dataset:
            The dataset.
    """
    # Iterate over all the datasets in the configuration
    all_datasets: List[Dataset] = list()
    with tqdm(dataset_configs, desc="Building dataset") as pbar:
        for cfg in pbar:

            # Update the progress bar
            log_str = f"Building dataset: {cfg.id}"
            extras = [x for x in [cfg.subset, cfg.split] if x is not None]
            if extras:
                log_str += f' ({", ".join(extras)})'
            pbar.set_description(log_str)

            # Load the dataset
            try:
                dataset = load_dataset(
                    cfg.id,
                    cfg.subset,
                    split=cfg.split,
                    cache_dir=cache_dir,
                )
            except DatasetGenerationError:
                dataset = DatasetDict.load_from_disk(cfg.id)[cfg.split]

            # Rename the columns
            dataset = dataset.rename_columns(
                column_mapping={
                    cfg.premise_column: "premise",
                    cfg.hypothesis_column: "hypothesis",
                    cfg.label_column: "labels",
                }
            )

            # Remove unused columns
            dataset = dataset.remove_columns(
                column_names=[
                    col
                    for col in dataset.column_names
                    if col not in ["premise", "hypothesis", "labels"]
                ],
            )

            # Set up lists of old and new label name orders
            existing_label_names = list(cfg.label_names.keys())
            new_label_names = ["entailment", "neutral", "contradiction"]

            # Create a mapping, which converts the old label names to the new label names
            label_mapping = [
                new_label_names.index(label) for label in existing_label_names
            ]

            # Convert the labels to the new label names
            dataset = dataset.map(
                lambda example: {
                    "labels": label_mapping[example["labels"]],
                },
            )

            # Change the label names
            dataset.features["labels"] = ClassLabel(
                num_classes=3,
                names=new_label_names,
            )

            # Shuffle the dataset
            dataset = dataset.shuffle(seed=seed)

            # Add the dataset to the list of datasets
            all_datasets.append(dataset)

    # Concatenate the datasets
    dataset = concatenate_datasets(all_datasets).shuffle(seed=seed)

    # Return the dataset
    return dataset


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def build_danfever_with_splits(config: DictConfig) -> None:
    """Creates dataset splits for the DanFEVER dataset and stores them to disk.

    Args:
        config (DictConfig):
            Hydra configuration object.
    """
    # Load the DanFEVER dataset
    dataset = load_dataset("strombergnlp/danfever", split="train")

    # Convert the dataset to a Pandas DataFrame
    df = dataset.to_pandas()

    # Get list unique `evidence_extract` values, along with their counts
    evidence_extract_counts = df.evidence_extract.value_counts()

    # Pick the evidence extracts for the test split, being the ones that first sum up
    # above 1,000 samples
    test_evidence_extract = evidence_extract_counts[
        evidence_extract_counts.cumsum() < 1000
    ].index.tolist()

    # Pick the evidence extracts for the validation split, being the ones that first
    # sum up above 500 samples, but not in the test split
    val_evidence_extract = evidence_extract_counts[
        (evidence_extract_counts.cumsum() < 1500)
        & (~evidence_extract_counts.index.isin(test_evidence_extract))
    ].index.tolist()

    # Pick the evidence extracts for the train split, being the rest
    train_evidence_extract = evidence_extract_counts[
        ~evidence_extract_counts.index.isin(
            test_evidence_extract + val_evidence_extract
        )
    ].index.tolist()

    # Convert the dataframes back to datasets
    train_dataset = Dataset.from_pandas(
        df[df.evidence_extract.isin(train_evidence_extract)],
        preserve_index=False,
    )
    val_dataset = Dataset.from_pandas(
        df[df.evidence_extract.isin(val_evidence_extract)],
        preserve_index=False,
    )
    test_dataset = Dataset.from_pandas(
        df[df.evidence_extract.isin(test_evidence_extract)],
        preserve_index=False,
    )

    # Package the datasets into a DatasetDict
    dataset_dict = DatasetDict(
        dict(train=train_dataset, val=val_dataset, test=test_dataset)
    )

    # Save the dataset splits to disk
    dataset_path = Path(config.dirs.data) / config.dirs.processed / "danfever"
    dataset_dict.save_to_disk(dataset_path)
