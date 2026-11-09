from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

ROOT = Path(__file__).resolve().parents[2]
TASK_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = TASK_DIR / "models"

def main():
    test = pd.read_csv(DATA_DIR / "test.csv")
    X_test = test["text"].astype(str)

    for target in ["category", "priority"]:
        model_path = MODEL_DIR / f"{target}_model.joblib"
        model = joblib.load(model_path)
        predictions = model.predict(X_test)
        print(f"\n{target.upper()} – HELD-OUT TEST RESULTS")
        print(f"Accuracy: {accuracy_score(test[target], predictions):.4f}")
        print(f"Macro F1: {f1_score(test[target], predictions, average='macro'):.4f}")
        print(classification_report(test[target], predictions, zero_division=0))

if __name__ == "__main__":
    main()
