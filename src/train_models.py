from pathlib import Path
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

PARAM_GRID = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "logreg__C": [0.1, 1.0, 3.0, 10.0],
}

def build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            max_features=30000,
            sublinear_tf=True,
        )),
        ("logreg", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        )),
    ])

def evaluate_and_save(name, search, X_val, y_val):
    predictions = search.predict(X_val)
    accuracy = accuracy_score(y_val, predictions)
    macro_f1 = f1_score(y_val, predictions, average="macro")
    labels = search.best_estimator_.named_steps["logreg"].classes_
    matrix = confusion_matrix(y_val, predictions, labels=labels)

    print(f"\n{'=' * 65}")
    print(f"{name.upper()} VALIDATION RESULTS")
    print(f"Best parameters: {search.best_params_}")
    print(f"CV macro F1: {search.best_score_:.4f}")
    print(f"Validation accuracy: {accuracy:.4f}")
    print(f"Validation macro F1: {macro_f1:.4f}")
    print("\nClassification report:\n")
    print(classification_report(y_val, predictions, zero_division=0))

    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(matrix, interpolation="nearest")
    ax.figure.colorbar(image, ax=ax)
    ax.set(
        xticks=range(len(labels)), yticks=range(len(labels)),
        xticklabels=labels, yticklabels=labels,
        xlabel="Predicted label", ylabel="True label",
        title=f"{name.title()} – Validation Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    threshold = matrix.max() / 2 if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > threshold else "black")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / f"confusion_matrix_{name}.png", dpi=160)
    plt.close(fig)

    joblib.dump(search.best_estimator_, MODEL_DIR / f"{name}_model.joblib")
    return predictions, {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "best_params": search.best_params_,
        "cv_macro_f1": float(search.best_score_),
    }

def main():
    train = pd.read_csv(DATA_DIR / "train.csv")
    validation = pd.read_csv(DATA_DIR / "validation.csv")

    X_train = train["text"].astype(str)
    X_val = validation["text"].astype(str)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    validation_output = validation[["ticket_id", "text", "category", "priority"]].copy()

    for target in ["category", "priority"]:
        print(f"\nTuning {target} model...")
        search = GridSearchCV(
            estimator=build_pipeline(),
            param_grid=PARAM_GRID,
            scoring="f1_macro",
            cv=cv,
            n_jobs=-1,
            verbose=1,
            refit=True,
        )
        search.fit(X_train, train[target])
        predictions, metrics = evaluate_and_save(
            target, search, X_val, validation[target]
        )
        validation_output[f"{target}_predicted"] = predictions
        # Confidence is the maximum predicted class probability. Lower
        # confidence gives us useful borderline examples for manual review.
        validation_output[f"{target}_confidence"] = search.predict_proba(X_val).max(axis=1)
        results[target] = metrics

    # Save validation predictions for transparent error analysis.
    validation_output.to_csv(REPORT_DIR / "validation_predictions.csv", index=False)

    # Rank validation examples for manual review. Misclassified rows come
    # first; when the synthetic baseline makes no errors, the least-confident
    # correct predictions are still useful borderline examples to inspect.
    errors = validation_output.copy()
    errors["category_wrong"] = errors["category"] != errors["category_predicted"]
    errors["priority_wrong"] = errors["priority"] != errors["priority_predicted"]
    errors["any_model_wrong"] = errors["category_wrong"] | errors["priority_wrong"]
    errors["review_confidence"] = errors[
        ["category_confidence", "priority_confidence"]
    ].min(axis=1)
    errors["error_reason_to_investigate"] = errors.apply(
        lambda row: (
            "Category prediction is incorrect; inspect overlapping issue vocabulary"
            if row["category_wrong"] and not row["priority_wrong"]
            else "Priority prediction is incorrect; inspect urgency wording"
            if row["priority_wrong"] and not row["category_wrong"]
            else "Both predictions are incorrect; inspect topic and urgency cues"
            if row["any_model_wrong"]
            else "Correct prediction with relatively low confidence; inspect borderline wording"
        ), axis=1
    )
    errors = errors.sort_values(
        ["any_model_wrong", "review_confidence"],
        ascending=[False, True]
    ).head(10)
    errors.to_csv(REPORT_DIR / "confusing_examples.csv", index=False)

    with open(REPORT_DIR / "validation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nSaved models:")
    print(MODEL_DIR / "category_model.joblib")
    print(MODEL_DIR / "priority_model.joblib")
    print("\nSaved validation metrics and error analysis in reports/")

if __name__ == "__main__":
    main()
