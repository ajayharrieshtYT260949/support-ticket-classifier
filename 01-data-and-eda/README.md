# Task 1 — Data Collection, Preparation and Exploratory Data Analysis

## 1. Overview

The objective of this task is to generate, load, inspect, clean, and prepare a support ticket dataset for machine learning.

The dataset contains customer support tickets with two prediction targets:

- **Category:** Identifies the type of customer support issue.
- **Priority:** Identifies the urgency of the ticket.

This task establishes the data preparation foundation for the Support Ticket Classifier project.

## 2. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

## 3. Dataset Information

The project uses a synthetically generated support ticket dataset.

**Dataset size:** 720 records  
**Total features:** 4

| Column | Description |
|---|---|
| ticket_id | Unique ticket identifier |
| text | Customer support ticket description |
| category | Ticket category |
| priority | Ticket priority |

### Ticket Categories

1. Account Access
2. Billing
3. Technical Issue
4. Shipping
5. Returns
6. Product Inquiry
7. Subscription
8. Security

### Priority Levels

- Low
- Medium
- High

The dataset is synthetically generated and does not contain personal customer information.

## 4. Project Structure

```text
01-data-and-eda/
├── src/
│   ├── generate_dataset.py
│   └── prepare_data.py
├── reports/
└── README.md

data/
├── raw/
│   └── support_tickets.csv
└── processed/
    ├── train.csv
    ├── validation.csv
    ├── test.csv
    └── dataset_summary.json
```

The dataset is shared at the project root so that subsequent machine learning tasks can reuse the same data.

## 5. Implementation

### Step 1: Dataset Generation

The `generate_dataset.py` script generates synthetic customer support tickets.

It creates ticket descriptions across eight categories and three priority levels.

A fixed random seed of 42 is used for reproducibility.

Generated data is saved to:

`data/raw/support_tickets.csv`

### Step 2: Data Loading and Inspection

The `prepare_data.py` script loads the raw dataset using Pandas.

The dataset is inspected for:

- Dataset dimensions
- Column data types
- Missing values
- Duplicate records
- Empty ticket descriptions
- Category distribution
- Priority distribution

### Step 3: Data Cleaning

The following cleaning operations are performed:

- Convert ticket text to lowercase.
- Remove leading and trailing whitespace.
- Normalize repeated whitespace.
- Remove rows with missing ticket descriptions or target labels.
- Remove empty ticket descriptions.
- Remove duplicate ticket descriptions.

### Step 4: Train, Validation and Test Split

The cleaned dataset is divided into three subsets:

| Dataset | Percentage |
|---|---:|
| Training | 70% |
| Validation | 15% |
| Testing | 15% |

Stratification is performed using a combined category-priority key to preserve the joint label distribution across the splits.

The random state is set to 42.

### Step 5: Save Processed Data

The processed datasets are saved as:

- `data/processed/train.csv`
- `data/processed/validation.csv`
- `data/processed/test.csv`

A dataset summary is also generated at:

`data/processed/dataset_summary.json`

## 6. Execution Instructions

Run the following commands from the project root directory.

Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Generate the dataset:

```powershell
python 01-data-and-eda/src/generate_dataset.py
```

Prepare and split the dataset:

```powershell
python 01-data-and-eda/src/prepare_data.py
```

## 7. Outcome

Task 1 successfully establishes the data preparation pipeline.

The resulting processed datasets are used by Task 2 for baseline model development and will also be reused for subsequent model experiments.

## 8. Important Note

This project uses synthetic data. Model performance on synthetic tickets may differ significantly from performance on real-world customer support tickets.

The same processed dataset is reused across subsequent tasks to support a consistent model comparison.