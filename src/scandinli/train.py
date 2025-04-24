"""Train an NLI model on the built dataset."""

import logging
from functools import partial
from pathlib import Path

import hydra
import numpy as np
import torch
from datasets import disable_progress_bar
from datasets.dataset_dict import Datasetdict
from omegaconf import dictConfig
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BatchEncoding,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    EvalPrediction,
    IntervalStrategy,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.training_args import OptimizerNames

from scandinli.build_data import build_data

# Ignore loggers from `datasets`
logging.getLogger("datasets").setLevel(logging.ERROR)

# Disable the `datasets` progress bar
disable_progress_bar()


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def train(config: dictConfig) -> None:
    """Train an NLI model on the built dataset.

    Args:
        config:
            The Hydra configuration.
    """
    # Build the dataset
    build_data(config)

    # Define the path to the data and model
    dataset_dir = Path(config.dirs.data) / config.dirs.final / "ScandiNLI"
    model_dir = Path(config.dirs.models) / config.model.output_model_id

    # Load in the dataset dictionary
    dataset = Datasetdict.load_from_disk(dataset_dir)

    # Load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)

    # Tokenize the dataset
    tokenized_dataset = dataset.map(
        partial(tokenize_function, tokenizer=tokenizer), batched=True
    )

    # Define the data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir=model_dir,
        eval_strategy=IntervalStrategy.STEPS,
        eval_steps=config.eval_steps,
        logging_strategy=IntervalStrategy.STEPS,
        logging_steps=config.logging_steps,
        save_strategy=IntervalStrategy.STEPS,
        save_steps=config.save_steps,
        max_steps=config.max_steps,
        save_total_limit=config.save_total_limit,
        per_device_train_batch_size=config.model.batch_size,
        per_device_eval_batch_size=config.model.batch_size,
        gradient_accumulation_steps=32 // config.model.batch_size,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        load_best_model_at_end=True,
        optim=OptimizerNames.ADAMW_TORCH,
        seed=config.seed,
        use_mps_device=torch.backends.mps.is_available(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        report_to=["wandb"] if config.use_wandb else ["none"],
        run_name=config.model.wandb_run_name,
    )

    # Define the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["val"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.patience)],
    )

    # Train the model, continuing from the last checkpoint if it exists
    if config.checkpoint:
        checkpoint = model_dir / f"checkpoint-{config.checkpoint}"
    else:
        checkpoint = None
    trainer.train(resume_from_checkpoint=checkpoint)

    # Save the model
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    # Push the model to the Hugging Face Hub
    if config.push_to_hub:
        model.push_to_hub(config.model.output_model_id, private=True)
        tokenizer.push_to_hub(config.model.output_model_id, private=True)


def load_model_and_tokenizer(
    config: dictConfig,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load the model and tokenizer.

    Args:
        config:
            The Hydra configuration.

    Returns:
        The model and tokenizer.
    """
    # Define the model directory
    model_dir = Path(config.dirs.models) / config.model.output_model_id

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.input_model_id, cache_dir=model_dir
    )

    # Ensure that `model_max_length` is set
    if tokenizer.model_max_length > 100_000 or tokenizer.model_max_length is None:
        tokenizer.model_max_length = 512

    # Define the model
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model.input_model_id, num_labels=3, cache_dir=model_dir
    )

    # Set the label names
    model.config.id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
    model.config.label2id = {"entailment": 0, "neutral": 1, "contradiction": 2}

    # Return the model and tokenizer
    return model, tokenizer


def tokenize_function(
    examples: BatchEncoding, tokenizer: PreTrainedTokenizerBase
) -> BatchEncoding:
    """Tokenize the examples.

    Args:
        examples:
            The examples to tokenize.
        tokenizer:
            The tokenizer to use.

    Returns:
        The tokenized examples.
    """
    return tokenizer(
        examples["premise"],
        examples["hypothesis"],
        padding=False,
        truncation="only_first",
    )


def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    """Compute the metrics.

    Args:
        eval_pred:
            The predictions to evaluate.

    Returns:
        The metrics.
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return dict(
        mcc=matthews_corrcoef(labels, predictions),
        accuracy=accuracy_score(labels, predictions),
        macro_f1=f1_score(labels, predictions, average="macro"),
    )
