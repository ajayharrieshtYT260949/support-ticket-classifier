from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "support_tickets.csv"
PROCESSED_DIR = ROOT / "data" / "processed"

def clean_text(value):
    """Lowercase and trim ticket text; collapse repeated whitespace."""
    return " ".join(str(value).lower().strip().split())

def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RAW_PATH)

    required = {"text", "category", "priority"}
    missing_columns = required - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    print("Original shape:", df.shape)
    print("\nMissing values:\n", df[["text", "category", "priority"]].isna().sum())

    df = df.dropna(subset=["text", "category", "priority"]).copy()
    df["text"] = df["text"].map(clean_text)
    df = df[df["text"].ne("")]
    before = len(df)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"\nDuplicate text rows removed: {before - len(df)}")
    print("Cleaned shape:", df.shape)

    if len(df) < 500:
        raise ValueError(f"At least 500 usable tickets are required; found {len(df)}.")

    # Stratifying on the combined label preserves the category-priority
    # combinations, and therefore the distribution of both target labels.
    df["stratify_key"] = df["category"].astype(str) + "___" + df["priority"].astype(str)

    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=SEED, stratify=df["stratify_key"]
    )
    validation_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=SEED,
        stratify=temp_df["stratify_key"]
    )

    for name, split in [
        ("train", train_df), ("validation", validation_df), ("test", test_df)
    ]:
        split = split.drop(columns=["stratify_key"]).reset_index(drop=True)
        split.to_csv(PROCESSED_DIR / f"{name}.csv", index=False)
        print(f"\n{name.title()} rows: {len(split)}")
        print("  Category counts:\n", split["category"].value_counts().sort_index().to_string())
        print("  Priority counts:\n", split["priority"].value_counts().sort_index().to_string())

    summary = {
        "total_rows": int(len(df)),
        "category_counts": {k: int(v) for k, v in df["category"].value_counts().sort_index().items()},
        "priority_counts": {k: int(v) for k, v in df["priority"].value_counts().sort_index().items()},
        "split_rows": {
            "train": int(len(train_df)),
            "validation": int(len(validation_df)),
            "test": int(len(test_df)),
        },
        "split_percentages": {"train": 70, "validation": 15, "test": 15},
        "duplicate_text_rows_removed": int(before - len(df)),
    }
    with open(PROCESSED_DIR / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nDataset summary saved to data/processed/dataset_summary.json")

if __name__ == "__main__":
    main()
