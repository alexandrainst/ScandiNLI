"""Process the DanFEVER dataset and store it to disk."""

import os

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset


def main() -> None:
    """Process the DanFEVER dataset and store it to disk."""
    # Load the DanFEVER dataset
    dataset = load_dataset("strombergnlp/danfever", split="train")
    assert isinstance(dataset, Dataset)

    # Convert the dataset to a Pandas DataFrame
    df = dataset.to_pandas()
    assert isinstance(df, pd.DataFrame)

    # Get list unique `evidence_extract` values, along with their counts
    evidence_extract_counts = df.evidence_extract.value_counts()

    # Pick the evidence extracts for the test split, being the maximum amount that
    # sum up below 1,000 samples
    test_evidence_extract = evidence_extract_counts[
        evidence_extract_counts.cumsum() < 1000
    ].index.tolist()

    # Pick the evidence extracts for the validation split, being the maximum amount
    # that sum up below 500 samples, and which are not in the test split
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
        df[df.evidence_extract.isin(train_evidence_extract)], preserve_index=False
    )
    val_dataset = Dataset.from_pandas(
        df[df.evidence_extract.isin(val_evidence_extract)], preserve_index=False
    )
    test_dataset = Dataset.from_pandas(
        df[df.evidence_extract.isin(test_evidence_extract)], preserve_index=False
    )

    # Package the datasets into a DatasetDict
    dataset_dict = DatasetDict(
        dict(train=train_dataset, val=val_dataset, test=test_dataset)
    )

    dataset_dict.save_to_disk(os.path.join("data", "danfever"))


if __name__ == "__main__":
    main()
