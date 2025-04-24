"""Evaluate finetuned NLI models on Danish, Swedish, and Norwegian NLI datasets."""

import logging
import os
from functools import partial
from pathlib import Path

import torch
import transformers.utils.logging as hf_logging
from datasets import disable_progress_bar
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

# Ignore loggers from `datasets`
logging.getLogger("datasets").setLevel(logging.ERROR)

# Disable the `datasets` progress bar
disable_progress_bar()

# Set up logging
logger = logging.getLogger(__name__)


def evaluate(config: DictConfig) -> None:
    """Evaluate finetuned NLI models on Danish, Swedish, and Norwegian NLI datasets.

    Args:
        config:
            Hydra config object.
    """
    # Get the model ID
    if config.evaluation.model_id is not None:
        model_id = config.evaluation.model_id
    else:
        model_id = config.model.output_model_id

    # Define data and model directories
    raw_dir = Path(config.dirs.data) / config.dirs.raw
    model_dir = Path(config.dirs.models) / model_id

    # Disable the `transformers` logging during model load
    hf_logging.set_verbosity_error()

    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)

    # Ensure that `model_max_length` is set
    if tokenizer.model_max_length > 100_000 or tokenizer.model_max_length is None:
        tokenizer.model_max_length = 512

    # Enable the `transformers` logging again
    hf_logging.set_verbosity_info()

    # Turn off tokenizer parallelism
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Iterate over the languages
    for language in config.evaluation.languages:
        # Build the test datasets
        test = build_dataset_for_single_language(
            dataset_configs=config.test_datasets[language],
            cache_dir=raw_dir,
            seed=config.seed,
            progress_bar=False,
            label_names=[model.config.id2label[idx] for idx in [0, 1, 2]],
        )

        # Tokenize the datasets
        tokenized_test = test.map(
            partial(tokenize_function, tokenizer=tokenizer), batched=True
        )

        # Define the data collator
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        # Define batch size
        if config.evaluation.batch_size is not None:
            batch_size = config.evaluation.batch_size
        else:
            batch_size = config.model.batch_size

        # Define the training arguments
        training_args = TrainingArguments(
            output_dir=model_dir,
            per_device_eval_batch_size=batch_size,
            use_mps_device=torch.backends.mps.is_available(),
            fp16=torch.cuda.is_available(),
            report_to="none",
        )

        # Define the trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            eval_dataset=tokenized_test,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        # Disable trainer logging
        trainer.log = lambda logs: None

        # Evaluate the model
        metrics = trainer.evaluate()
        mcc = metrics["eval_mcc"]
        accuracy = metrics["eval_accuracy"]
        macro_f1 = metrics["eval_macro_f1"]

        # Log
        logger.info(
            f"=== SCORES FOR {language.upper()} ===\n"
            f"MCC: {mcc}\n"
            f"Accuracy: {accuracy}\n"
            f"Macro F1: {macro_f1}"
        )
