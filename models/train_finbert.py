"""Train + evaluate FinBERT sentiment model with MLflow logging.

What it does:
- Loads Financial PhraseBank (sentences_allagree)
- Splits into train/test
- Evaluates the BASE ProsusAI/finbert checkpoint on test set
- Fine-tunes for 3 epochs (CPU-friendly defaults)
- Evaluates the fine-tuned model on the same test set
- Logs accuracy/F1/confusion matrix + loss curve artifacts to MLflow
- Saves fine-tuned model to `models/finbert-finetuned/`

Run:
    python -m models.train_finbert
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import mlflow
import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from config import MLFLOW_TRACKING_URI

BASE_MODEL = "ProsusAI/finbert"
OUT_DIR = Path(__file__).parent / "finbert-finetuned"

# ProsusAI/finbert label mapping: {0: positive, 1: negative, 2: neutral}
# Financial PhraseBank label mapping: {0: negative, 1: neutral, 2: positive}
# Need to remap base model predictions to FPB labels for fair comparison
BASE_TO_FPB = {0: 2, 1: 0, 2: 1}


def _set_mlflow_uri() -> None:
    if MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def _tokenize_fn(tokenizer, examples):
    return tokenizer(examples["sentence"], truncation=True, max_length=128)


def _compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro")),
    }


def _eval_base_against_fpb(base_preds, y_true) -> Tuple[Dict[str, float], np.ndarray]:
    """Score ProsusAI/finbert predictions against Financial PhraseBank labels."""
    y_pred = np.array([BASE_TO_FPB[int(p)] for p in base_preds])
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }
    return metrics, confusion_matrix(y_true, y_pred, labels=[0, 1, 2])


def _eval_model(
    model_name_or_path: str,
    tokenized_test,
    tokenizer,
    data_collator,
    is_base_model: bool = False,
) -> Tuple[Dict[str, float], np.ndarray]:
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
    args = TrainingArguments(
        output_dir=str(Path(tempfile.gettempdir()) / "finbert-eval"),
        per_device_eval_batch_size=32,
        dataloader_drop_last=False,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=_compute_metrics,
        eval_dataset=tokenized_test,
    )
    preds = trainer.predict(tokenized_test).predictions
    y_pred = np.argmax(preds, axis=-1)
    y_true = np.array(tokenized_test["label"])

    if is_base_model:
        return _eval_base_against_fpb(y_pred, y_true)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }
    return metrics, cm


def main() -> None:
    _set_mlflow_uri()
    mlflow.set_experiment("finbert-sentiment")

    # Financial PhraseBank is implemented as a dataset loading script on HF.
    # We pin `datasets==2.19.2` in requirements to support this dataset.
    dataset = load_dataset("takala/financial_phrasebank", "sentences_allagree", trust_remote_code=True)

    # The dataset provides a single split named "train"; we make our own stratified train/test split.
    # NOTE: datasets' built-in `train_test_split(stratify_by_column=...)` can fail under NumPy 2.x
    # due to a strict `copy=False` conversion path. We split indices via scikit-learn instead.
    full = dataset["train"]
    labels = full["label"]
    idx = np.arange(len(full))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )
    train_ds = full.select(train_idx.tolist())
    test_ds = full.select(test_idx.tolist())

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    tokenized_train = train_ds.map(lambda ex: _tokenize_fn(tokenizer, ex), batched=True, remove_columns=["sentence"])
    tokenized_test = test_ds.map(lambda ex: _tokenize_fn(tokenizer, ex), batched=True, remove_columns=["sentence"])

    with mlflow.start_run(run_name="finbert_base_vs_finetuned") as run:
        mlflow.log_param("base_model", BASE_MODEL)
        mlflow.log_param("dataset", "takala/financial_phrasebank:sentences_allagree")
        mlflow.log_param("train_size", len(tokenized_train))
        mlflow.log_param("test_size", len(tokenized_test))

        # -----------------------
        # Base model evaluation
        # -----------------------
        base_metrics, base_cm = _eval_model(BASE_MODEL, tokenized_test, tokenizer, data_collator, is_base_model=True)
        mlflow.log_metric("base_accuracy", base_metrics["accuracy"])
        mlflow.log_metric("base_f1_macro", base_metrics["f1_macro"])

        # -----------------------
        # Fine-tune
        # -----------------------
        model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=3)

        # CPU-friendly defaults; runs faster with GPU if available.
        args = TrainingArguments(
            output_dir=str(Path(tempfile.gettempdir()) / "finbert-train"),
            num_train_epochs=3,
            learning_rate=2e-5,
            weight_decay=0.01,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            evaluation_strategy="epoch",
            save_strategy="no",
            logging_strategy="steps",
            logging_steps=50,
            report_to=[],
            fp16=torch.cuda.is_available(),
            seed=42,
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_test,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=_compute_metrics,
        )

        trainer.train()

        # Persist fine-tuned model for local inference.
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(OUT_DIR)
        tokenizer.save_pretrained(OUT_DIR)
        mlflow.log_param("finetuned_dir", str(OUT_DIR))

        # -----------------------
        # Fine-tuned evaluation
        # -----------------------
        ft_metrics = trainer.evaluate()
        mlflow.log_metric("finetuned_accuracy", float(ft_metrics.get("eval_accuracy", 0.0)))
        mlflow.log_metric("finetuned_f1_macro", float(ft_metrics.get("eval_f1_macro", 0.0)))

        # Confusion matrices + loss curve artifacts
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)

            # Confusion matrices (JSON)
            (td_path / "base_confusion_matrix.json").write_text(json.dumps(base_cm.tolist(), indent=2), encoding="utf-8")

            preds = trainer.predict(tokenized_test).predictions
            y_pred = np.argmax(preds, axis=-1)
            y_true = np.array(tokenized_test["label"])
            ft_cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
            (td_path / "finetuned_confusion_matrix.json").write_text(
                json.dumps(ft_cm.tolist(), indent=2), encoding="utf-8"
            )

            # Loss curve (Trainer log history)
            history = trainer.state.log_history
            (td_path / "trainer_log_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

            # Base vs fine-tuned summary (for report)
            summary = {
                "base": base_metrics,
                "finetuned": {
                    "accuracy": float(ft_metrics.get("eval_accuracy", 0.0)),
                    "f1_macro": float(ft_metrics.get("eval_f1_macro", 0.0)),
                },
            }
            (td_path / "comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

            mlflow.log_artifacts(str(td_path), artifact_path="finbert_eval")

        # Helpful console print
        print("Base metrics:", base_metrics)
        print("Fine-tuned metrics:", {"accuracy": float(ft_metrics.get("eval_accuracy", 0.0)), "f1_macro": float(ft_metrics.get("eval_f1_macro", 0.0))})
        print("Saved fine-tuned model to:", OUT_DIR)
        print("MLflow run_id:", run.info.run_id)


if __name__ == "__main__":
    # Avoid HuggingFace tokenizers parallelism warning noise on Windows.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()

