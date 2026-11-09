
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import accuracy_score, f1_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from datasets import Dataset


# --------------------------------------------------
# 1. PROJECT PATHS AND CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_DIR = PROJECT_ROOT / "03-distilbert"

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = TASK_DIR / "models"
REPORT_DIR = TASK_DIR / "reports"

MODEL_NAME = "distilbert-base-uncased"

TEXT_COLUMN = "text"
TARGET_COLUMNS = ["category", "priority"]

MAX_LENGTH = 128
EPOCHS = 2
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16
LEARNING_RATE = 2e-5
SEED = 42

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)


# --------------------------------------------------
# 2. LOAD SHARED DATASETS
# --------------------------------------------------

train_df = pd.read_csv(DATA_DIR / "train.csv")
validation_df = pd.read_csv(DATA_DIR / "validation.csv")

required_columns = {TEXT_COLUMN, *TARGET_COLUMNS}

for name, dataframe in [
    ("Training", train_df),
    ("Validation", validation_df),
]:
    missing = required_columns - set(dataframe.columns)

    if missing:
        raise ValueError(
            f"{name} dataset is missing columns: {missing}"
        )

    if dataframe[TEXT_COLUMN].isna().any():
        raise ValueError(f"{name} dataset contains missing ticket text.")

print("Training shape:", train_df.shape)
print("Validation shape:", validation_df.shape)


# --------------------------------------------------
# 3. METRICS
# --------------------------------------------------

def compute_metrics(eval_prediction):
    logits, labels = eval_prediction

    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        ),
    }


# --------------------------------------------------
# 4. TRAIN ONE DISTILBERT MODEL
# --------------------------------------------------

def train_one_model(target_column):

    print("\n" + "=" * 60)
    print(f"TRAINING MODEL: {target_column.upper()}")
    print("=" * 60)

    output_dir = MODEL_DIR / f"{target_column}_model"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Encode labels using a stable mapping.
    classes = sorted(
        train_df[target_column].dropna().astype(str).unique()
    )

    if len(classes) < 2:
        raise ValueError(
            f"At least two classes are required for {target_column}."
        )

    label2id = {label: index for index, label in enumerate(classes)}
    id2label = {index: label for label, index in label2id.items()}

    train_data = train_df[
        [TEXT_COLUMN, target_column]
    ].dropna().copy()

    validation_data = validation_df[
        [TEXT_COLUMN, target_column]
    ].dropna().copy()

    train_data[target_column] = train_data[target_column].astype(str)
    validation_data[target_column] = validation_data[target_column].astype(str)

    unknown_labels = (
        set(validation_data[target_column]) - set(classes)
    )

    if unknown_labels:
        raise ValueError(
            f"Unknown validation labels for {target_column}: "
            f"{unknown_labels}"
        )

    train_data["labels"] = train_data[target_column].map(label2id)
    validation_data["labels"] = validation_data[target_column].map(label2id)

    # Load tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(batch):
        return tokenizer(
            batch[TEXT_COLUMN],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    # Convert to Hugging Face datasets.
    train_dataset = Dataset.from_pandas(
        train_data[[TEXT_COLUMN, "labels"]],
        preserve_index=False,
    )

    validation_dataset = Dataset.from_pandas(
        validation_data[[TEXT_COLUMN, "labels"]],
        preserve_index=False,
    )

    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=[TEXT_COLUMN],
    )

    validation_dataset = validation_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=[TEXT_COLUMN],
    )

    # Load pretrained classification model.
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(classes),
        label2id=label2id,
        id2label=id2label,
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir / "training_checkpoints"),
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="epoch",
        report_to="none",
        use_cpu=True,
        seed=SEED,
        data_seed=SEED,
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train and measure elapsed time.
    start_time = time.perf_counter()

    train_result = trainer.train()

    training_time = time.perf_counter() - start_time

    # Save model and tokenizer.
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    validation_metrics = trainer.evaluate()

    # Save label mapping.
    with open(output_dir / "labels.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "label2id": label2id,
                "id2label": {
                    str(key): value
                    for key, value in id2label.items()
                },
            },
            file,
            indent=4,
        )

    return {
        "target": target_column,
        "classes": classes,
        "training_time_seconds": round(training_time, 4),
        "training_loss": float(train_result.training_loss),
        "validation_accuracy": float(validation_metrics["eval_accuracy"]),
        "validation_macro_f1": float(validation_metrics["eval_macro_f1"]),
        "model_path": str(output_dir.relative_to(PROJECT_ROOT)),
    }


# --------------------------------------------------
# 5. TRAIN CATEGORY AND PRIORITY
# --------------------------------------------------

if __name__ == "__main__":

    results = {}

    for target in TARGET_COLUMNS:
        results[target] = train_one_model(target)

    report_path = REPORT_DIR / "training_metrics.json"

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print("\nTraining completed successfully.")

    print(json.dumps(results, indent=4))

    print("\nTraining report saved to:", report_path)
