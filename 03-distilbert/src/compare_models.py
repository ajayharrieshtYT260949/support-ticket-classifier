
from pathlib import Path
import json
import time

import joblib
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

BASELINE_DIR = (
    PROJECT_ROOT / "02-baseline-model" / "models"
)

DISTILBERT_DIR = TASK_DIR / "models"
REPORT_DIR = TASK_DIR / "reports"

TEXT_COLUMN = "text"
TARGET_COLUMNS = ["category", "priority"]

MAX_LENGTH = 128

REPORT_DIR.mkdir(parents=True, exist_ok=True)

test_df = pd.read_csv(DATA_DIR / "test.csv")

with open(REPORT_DIR / "test_metrics.json", encoding="utf-8") as file:
    distilbert_metrics = json.load(file)


def measure_baseline(target):

    model_path = BASELINE_DIR / f"{target}_model.joblib"

    baseline_model = joblib.load(model_path)

    dataframe = test_df[
        [TEXT_COLUMN, target]
    ].dropna().copy()

    texts = dataframe[TEXT_COLUMN].astype(str).tolist()
    labels = dataframe[target].astype(str).tolist()

    predictions = baseline_model.predict(texts)

    accuracy = accuracy_score(labels, predictions)

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    # Warm-up before latency measurement.
    if texts:
        baseline_model.predict([texts[0]])

    inference_times = []

    for text in texts:

        start_time = time.perf_counter()

        baseline_model.predict([text])

        elapsed = time.perf_counter() - start_time

        inference_times.append(elapsed)

    average_inference_ms = (
        np.mean(inference_times) * 1000
        if inference_times else 0.0
    )

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "inference_time_per_ticket_ms": float(
            average_inference_ms
        ),
    }


def measure_distilbert(target):

    model_path = DISTILBERT_DIR / f"{target}_model"

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path
    )

    model.to("cpu")
    model.eval()

    dataframe = test_df[
        [TEXT_COLUMN, target]
    ].dropna().copy()

    with open(model_path / "labels.json", encoding="utf-8") as file:
        label_data = json.load(file)

    label2id = label_data["label2id"]

    texts = dataframe[TEXT_COLUMN].astype(str).tolist()

    true_labels = [
        label2id[str(label)]
        for label in dataframe[target]
    ]

    predictions = []
    inference_times = []

    if texts:

        warmup = tokenizer(
            texts[0],
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
        )

        with torch.inference_mode():
            model(**warmup)

    for text in texts:

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
        )

        start_time = time.perf_counter()

        with torch.inference_mode():
            outputs = model(**inputs)

        inference_times.append(
            time.perf_counter() - start_time
        )

        predictions.append(
            torch.argmax(outputs.logits, dim=-1).item()
        )

    average_inference_ms = (
        np.mean(inference_times) * 1000
        if inference_times else 0.0
    )

    return {
        "accuracy": float(
            accuracy_score(true_labels, predictions)
        ),
        "macro_f1": float(
            f1_score(
                true_labels,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "inference_time_per_ticket_ms": float(
            average_inference_ms
        ),
    }


if __name__ == "__main__":

    comparison_rows = []

    for target in TARGET_COLUMNS:

        print(f"\nComparing {target} models...")

        baseline = measure_baseline(target)
        distilbert = measure_distilbert(target)

        comparison_rows.append({
            "task": target,
            "model": "TF-IDF + Logistic Regression",
            **baseline,
        })

        comparison_rows.append({
            "task": target,
            "model": "DistilBERT",
            **distilbert,
        })

    comparison_df = pd.DataFrame(comparison_rows)

    comparison_path = REPORT_DIR / "model_comparison.csv"

    comparison_df.to_csv(
        comparison_path,
        index=False,
    )

    print("\nMODEL COMPARISON")
    print(comparison_df.to_string(index=False))

    print("\nComparison saved to:", comparison_path)
