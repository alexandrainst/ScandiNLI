"""Train an NLI model on the built dataset."""

import logging
from functools import partial
from pathlib import Path
from typing import Dict

import hydra
import numpy as np
import torch
from datasets import disable_progress_bar
from datasets.dataset_dict import DatasetDict
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BatchEncoding,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    EvalPrediction,
    IntervalStrategy,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.training_args import OptimizerNames

# Ignore loggers from `datasets`
logging.getLogger("datasets").setLevel(logging.ERROR)

# Disable the `datasets` progress bar
disable_progress_bar()


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def train(config: DictConfig) -> None:
    """Train an NLI model on the built dataset.

    Args:
        config (DictConfig):
            The Hydra configuration.
    """
    # Define the path to the data and model
    dataset_dir = Path(config.dirs.data) / config.dirs.final / "ScandiNLI"
    model_dir = Path(config.dirs.models) / config.model.output_model_id

    # Load in the dataset dictionary
    dataset = DatasetDict.load_from_disk(dataset_dir)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.input_model_id, cache_dir=model_dir
    )

    # Ensure that `model_max_length` is set
    if tokenizer.model_max_length > 100_000:
        tokenizer.model_max_length = 512

    # Tokenize the dataset
    tokenized_dataset = dataset.map(
        partial(tokenize_function, tokenizer=tokenizer),
        batched=True,
    )

    # Define the data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Define the model
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model.input_model_id,
        num_labels=3,
        cache_dir=model_dir,
    )

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir=model_dir,
        evaluation_strategy=IntervalStrategy.STEPS,
        logging_strategy=IntervalStrategy.STEPS,
        save_strategy=IntervalStrategy.STEPS,
        eval_steps=config.eval_steps,
        logging_steps=config.logging_steps,
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
        fp16=torch.cuda.is_available(),
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

    # Evaluate the model
    trainer.evaluate(tokenized_dataset["test"])

    # Push the model to the Hugging Face Hub
    if config.push_to_hub:
        trainer.push_to_hub(private=True)
        tokenizer.push_to_hub(config.model.output_model_id, private=True)


def tokenize_function(
    examples: BatchEncoding, tokenizer: PreTrainedTokenizerBase
) -> BatchEncoding:
    """Tokenize the examples.

    Args:
        examples (BatchEncoding):
            The examples to tokenize.
        tokenizer (PreTrainedTokenizerBase):
            The tokenizer to use.

    Returns:
        BatchEncoding:
            The tokenized examples.
    """
    return tokenizer(
        examples["premise"],
        examples["hypothesis"],
        padding=False,
        truncation="only_first",
    )


def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    """Compute the metrics.

    Args:
        eval_pred (EvalPrediction):
            The predictions to evaluate.

    Returns:
        Dict[str, float]:
            The metrics.
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return dict(
        mcc=matthews_corrcoef(labels, predictions),
        accuracy=accuracy_score(labels, predictions),
        macro_f1=f1_score(labels, predictions, average="macro"),
    )
