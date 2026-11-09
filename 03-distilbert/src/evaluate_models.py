
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
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_DIR = PROJECT_ROOT / "03-distilbert"

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = TASK_DIR / "models"
REPORT_DIR = TASK_DIR / "reports"

TEXT_COLUMN = "text"
TARGET_COLUMNS = ["category", "priority"]

MAX_LENGTH = 128

REPORT_DIR.mkdir(parents=True, exist_ok=True)

test_df = pd.read_csv(DATA_DIR / "test.csv")

results = {}


def evaluate_one_model(target):

    print(f"\nEvaluating {target} model...")

    model_path = MODEL_DIR / f"{target}_model"

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path
    )

    model.to("cpu")
    model.eval()

    with open(model_path / "labels.json", encoding="utf-8") as file:
        label_data = json.load(file)

    label2id = label_data["label2id"]

    dataframe = test_df[
        [TEXT_COLUMN, target]
    ].dropna().copy()

    dataframe[target] = dataframe[target].astype(str)

    unknown_labels = set(dataframe[target]) - set(label2id)

    if unknown_labels:
        raise ValueError(
            f"Unknown test labels for {target}: {unknown_labels}"
        )

    true_labels = np.array(
        [label2id[label] for label in dataframe[target]]
    )

    predictions = []

    inference_times = []

    # Warm-up to avoid including initial model overhead.
    if len(dataframe) > 0:
        warmup = tokenizer(
            dataframe.iloc[0][TEXT_COLUMN],
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
        )

        with torch.inference_mode():
            model(**warmup)

    # Measure single-ticket inference latency.
    for text in dataframe[TEXT_COLUMN].astype(str):

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
        )

        start_time = time.perf_counter()

        with torch.inference_mode():
            outputs = model(**inputs)

        elapsed = time.perf_counter() - start_time

        inference_times.append(elapsed)

        prediction = torch.argmax(
            outputs.logits,
            dim=-1,
        ).item()

        predictions.append(prediction)

    predictions = np.array(predictions)

    accuracy = accuracy_score(
        true_labels,
        predictions,
    )

    macro_f1 = f1_score(
        true_labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    average_inference_time = (
        float(np.mean(inference_times))
        if inference_times else 0.0
    )

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "inference_time_per_ticket_ms": round(
            average_inference_time * 1000,
            4,
        ),
        "test_samples": len(dataframe),
    }


if __name__ == "__main__":

    for target in TARGET_COLUMNS:
        results[target] = evaluate_one_model(target)

    report_path = REPORT_DIR / "test_metrics.json"

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print("\nTest evaluation completed.")
    print(json.dumps(results, indent=4))

    print("\nReport saved to:", report_path)
