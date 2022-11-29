"""Evaluate finetuned NLI models on Danish, Swedish, and Norwegian NLI datasets."""

import logging
from functools import partial
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from .build_data import build_dataset_for_single_language
from .train import compute_metrics, tokenize_function

# Set up logging
logger = logging.getLogger(__name__)


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def evaluate(config: DictConfig) -> None:
    """Evaluate finetuned NLI models on Danish, Swedish, and Norwegian NLI datasets.

    Args:
        config (DictConfig):
            Hydra config object.
    """
    # Define data and model directories
    raw_dir = Path(config.dirs.data) / config.dirs.raw
    model_dir = Path(config.dirs.models) / config.model.output_model_id

    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model.output_model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model.output_model_id
    )

    # Iterate over the languages
    for language in ["da", "sv", "nb"]:

        # Log
        logger.info(f"Evaluating on {language}.")

        # Build the test datasets
        test = build_dataset_for_single_language(
            dataset_configs=config.test_datasets[language],
            cache_dir=raw_dir,
            seed=config.seed,
        )

        # Tokenize the datasets
        tokenized_test = test.map(
            partial(tokenize_function, tokenizer=tokenizer),
            batched=True,
        )

        # Define the data collator
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        # Define the training arguments
        training_args = TrainingArguments(
            output_dir=model_dir,
            per_device_eval_batch_size=config.model.batch_size,
            use_mps_device=torch.backends.mps.is_available(),
            fp16=torch.cuda.is_available(),
        )

        # Define the trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            eval_dataset=tokenized_test,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        # Evaluate the model
        metrics = trainer.evaluate()

        # Log
        logger.info(f"Metrics: {metrics}")
