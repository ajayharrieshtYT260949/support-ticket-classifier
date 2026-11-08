# Task 2 — Baseline Model Development

## 1. Overview

The objective of this task is to develop baseline machine learning models for automatic customer support ticket classification.

The system predicts two outputs from a customer support ticket description:

1. **Category Classification:** Predicts the type of customer support issue.
2. **Priority Classification:** Predicts the urgency level of the ticket.

Two separate machine learning pipelines are developed using TF-IDF vectorization and Logistic Regression.

## 2. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression
- GridSearchCV
- Matplotlib
- Joblib

## 3. Project Structure

```text
02-baseline-model/
├── src/
│   ├── train_models.py
│   └── evaluate_models.py
├── models/
│   ├── category_model.joblib
│   └── priority_model.joblib
├── reports/
│   ├── confusing_examples.csv
│   ├── confusion_matrix_category.png
│   ├── confusion_matrix_priority.png
│   ├── validation_metrics.json
│   └── validation_predictions.csv
└── README.md

data/
└── processed/
    ├── train.csv
    ├── validation.csv
    └── test.csv
```

## 4. Machine Learning Workflow

### Step 1: Load Processed Dataset

The training and validation datasets generated in Task 1 are loaded from the shared `data/processed/` directory.

The ticket description is used as the input feature.

The category and priority columns are used as separate prediction targets.

### Step 2: Text Vectorization Using TF-IDF

TF-IDF (Term Frequency-Inverse Document Frequency) converts textual ticket descriptions into numerical feature vectors.

It assigns importance to words based on their frequency within a ticket and across the document collection.

The following configurations are evaluated:

- Unigrams: `(1, 1)`
- Unigrams and bigrams: `(1, 2)`

### Step 3: Logistic Regression

Two independent Logistic Regression models are developed:

**Category Model**

Predicts one of the eight support ticket categories.

**Priority Model**

Predicts one of the three priority levels: Low, Medium, and High.

Each model is implemented as a Scikit-learn pipeline combining TF-IDF vectorization and Logistic Regression.

### Step 4: Hyperparameter Tuning

GridSearchCV is used to identify suitable model configurations.

The hyperparameters evaluated are:

| Hyperparameter | Values |
|---|---|
| Logistic Regression C | 0.1, 1.0, 3.0, 10.0 |
| TF-IDF ngram_range | (1, 1), (1, 2) |
| Cross-validation | 5-fold Stratified CV |
| Scoring metric | Macro F1 |

The best configurations from the current experiment are:

**Category Model**

- C = 0.1
- ngram_range = (1, 1)

**Priority Model**

- C = 0.1
- ngram_range = (1, 2)

### Step 5: Model Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- Macro F1-score
- Classification Report
- Confusion Matrix

Validation metrics and predictions are saved for further analysis.

### Step 6: Error Analysis

The project generates a collection of confusing ticket examples from validation predictions.

These examples help identify classification errors and understand where the models may struggle.

### Step 7: Model Serialization

The trained pipelines are saved using Joblib.

Saved models:

- `models/category_model.joblib`
- `models/priority_model.joblib`

These serialized pipelines can be loaded later without retraining and will be reused in the project where appropriate.

## 5. Experimental Results

The following results were obtained in the current synthetic dataset experiment.

### Category Classification

| Metric | Validation | Test |
|---|---:|---:|
| Accuracy | 1.0000 | 1.0000 |
| Macro F1 | 1.0000 | 1.0000 |

### Priority Classification

| Metric | Validation | Test |
|---|---:|---:|
| Accuracy | 1.0000 | 1.0000 |
| Macro F1 | 1.0000 | 1.0000 |

Both models achieved perfect scores on the current validation and held-out test sets.

### Important Experimental Observation

The dataset is synthetically generated using structured ticket templates. Therefore, category-specific words and priority-related phrases may make the classification task relatively easy.

The reported results describe performance on this synthetic dataset and should not be interpreted as guaranteed real-world accuracy.

Further evaluation using more diverse, real-world support ticket data would be necessary to assess generalization.

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

Train both baseline models:

```powershell
python 02-baseline-model/src/train_models.py
```

Evaluate both models using the held-out test dataset:

```powershell
python 02-baseline-model/src/evaluate_models.py
```

## 7. Generated Outputs

After successful execution, the following artifacts are generated:

**Models**
- Category classification model
- Priority classification model

**Reports**
- Validation metrics
- Validation predictions
- Confusing ticket examples
- Category confusion matrix
- Priority confusion matrix

## 8. Outcome

Task 2 successfully establishes the baseline machine learning solution for the Support Ticket Classifier project.

The TF-IDF and Logistic Regression pipelines provide a reference point for evaluating more advanced approaches.

The baseline models and their evaluation results will be used for comparison with the DistilBERT fine-tuned models developed in Task 3.

## 9. Next Step

Task 3 will focus on fine-tuning DistilBERT using the Hugging Face Transformers Trainer API.

The transformer-based models will be evaluated using the same validation and test datasets to compare their performance against the baseline models.