
# Task 3 - DistilBERT Fine-Tuning and Model Comparison

## Objective

Fine-tune pretrained DistilBERT models to classify support tickets by category and priority.

Compare transformer-based models against the existing TF-IDF and Logistic Regression baseline.

## Dataset

The project uses the existing shared processed datasets:

- `data/processed/train.csv`
- `data/processed/validation.csv`
- `data/processed/test.csv`

The original 70/15/15 split and random seed 42 are preserved.

The dataset contains:

- `ticket_id`
- `text`
- `category`
- `priority`

The test set is reserved for final evaluation.

## Models

### Baseline

- TF-IDF vectorization
- Logistic Regression
- Existing Task 2 trained models

### Transformer

- DistilBERT
- Hugging Face Transformers
- Hugging Face Trainer API
- Separate category and priority classifiers

## Training Configuration

| Parameter | Value |
|---|---|
| Base model | distilbert-base-uncased |
| Epochs | 2 |
| Learning rate | 0.00002 |
| Training batch size | 8 |
| Evaluation batch size | 16 |
| Maximum sequence length | 128 |
| Random seed | 42 |
| Device | CPU |

## Evaluation Metrics

The models are evaluated using:

- Accuracy
- Macro F1-score
- Training time
- Average inference time per ticket

## Project Structure

```text
03-distilbert/
├── src/
│   ├── train_models.py
│   ├── evaluate_models.py
│   └── compare_models.py
├── models/
│   ├── category_model/
│   └── priority_model/
├── reports/
│   ├── training_metrics.json
│   ├── test_metrics.json
│   └── model_comparison.csv
└── README.md
```

## Execution

Run the following commands from the repository root with the Task 3 environment activated.

### 1. Train the models

```powershell
python 03-distilbert/src/train_models.py
```

### 2. Evaluate on the test dataset

```powershell
python 03-distilbert/src/evaluate_models.py
```

### 3. Compare against the baseline

```powershell
python 03-distilbert/src/compare_models.py
```

## Results

Generated reports:

- `reports/training_metrics.json`
- `reports/test_metrics.json`
- `reports/model_comparison.csv`

The comparison evaluates predictive performance and inference speed for both classification tasks.

## Deployment Recommendation

The final deployment model should be selected based on macro F1-score, accuracy, inference latency, computational requirements, and operational cost.

A TF-IDF baseline may be preferable when it offers competitive predictive performance with substantially lower latency and resource requirements. DistilBERT may be preferable when its improvement in classification quality justifies its additional computational cost.

The final choice will be based on the measured results.
