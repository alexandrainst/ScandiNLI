"""Evaluate finetuned NLI models on Danish, Swedish, and Norwegian NLI datasets."""

import logging
import os
from functools import partial
from pathlib import Path

import litellm
import torch
import transformers.utils.logging as hf_logging
from datasets import disable_progress_bar
from Levenshtein import distance as levenshtein_distance
from litellm.types.utils import ModelResponse
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from tqdm.auto import tqdm
from transformers.data.data_collator import DataCollatorWithPadding
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

from .data import build_dataset_for_single_language
from .training import compute_metrics, tokenize_function

# Ignore loggers from `datasets`
logging.getLogger("datasets").setLevel(logging.ERROR)

# Disable the `datasets` progress bar
disable_progress_bar()

# Set up logging
logger = logging.getLogger(__name__)


PROMPT = """
    Here is a premise in {language}:

    <premise>
    {premise}
    </premise>

    Here is a hypothesis concerning the premise:

    <hypothesis>
    {hypothesis}
    </hypothesis>

    Determine whether the hypothesis is implied by the premise (entailment), contradicts
    the premise (contradiction), or is neither (neutral). Answer with "entailment",
    "contradiction", or "neutral", and nothing else.
"""


def evaluate_encoder(config: DictConfig) -> None:
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


def evaluate_litellm(config: DictConfig) -> None:
    """Evaluate LiteLLM API model on Danish, Swedish, and Norwegian NLI datasets.

    Args:
        config:
            Hydra config object.
    """
    # Disable `litellm` logging
    logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)
    logging.getLogger("httpx").setLevel(logging.CRITICAL)
    logging.getLogger("openai").setLevel(logging.CRITICAL)
    litellm.suppress_debug_info = True

    if config.evaluation.model_id is None:
        raise ValueError(
            "The `evaluation.model_id` must be set for evaluating LiteLLM models."
        )
    model_id = config.evaluation.model_id

    raw_dir = Path(config.dirs.data) / config.dirs.raw
    for language in config.evaluation.languages:
        # Build the prompts
        test = build_dataset_for_single_language(
            dataset_configs=config.test_datasets[language],
            cache_dir=raw_dir,
            seed=config.seed,
            progress_bar=False,
        )
        prompts = [
            PROMPT.format(language=language, premise=premise, hypothesis=hypothesis)
            for premise, hypothesis in zip(test["premise"], test["hypothesis"])
        ]

        # Get the generations from the model
        responses = [
            litellm.completion_with_retries(
                model=model_id,
                messages=[dict(role="user", content=prompt)],
                temperature=0.0,
                max_tokens=5,
            )
            for prompt in tqdm(
                iterable=prompts, desc=f"Evaluating {language.upper()}", unit="prompt"
            )
        ]
        responses = [
            response["choices"][0]["message"]["content"]
            for response in responses
            if isinstance(response, ModelResponse)
        ]

        # Extract the predictions
        label_names = test.features["labels"].names
        distances = [
            [levenshtein_distance(s1=response, s2=label) for label in label_names]
            for response in responses
        ]
        predictions = [
            label_names[min(range(len(distance_list)), key=distance_list.__getitem__)]
            for distance_list in distances
        ]

        # Extract the ground truth labels
        labels = [label_names[label_idx] for label_idx in test["labels"]]

        # Compute the metrics
        mcc = matthews_corrcoef(y_true=labels, y_pred=predictions)
        accuracy = accuracy_score(y_true=labels, y_pred=predictions)
        macro_f1 = f1_score(y_true=labels, y_pred=predictions, average="macro")

        logger.info(
            f"=== SCORES FOR {language.upper()} ===\n"
            f"MCC: {mcc}\n"
            f"Accuracy: {accuracy}\n"
            f"Macro F1: {macro_f1}"
        )
