# Task 4 — FastAPI Endpoint and Demo

## Overview

This task exposes the Support Ticket Classifier through a REST API built using FastAPI.

The application uses two fine-tuned DistilBERT models from Task 3:

- Category classification
- Priority classification

The models are loaded once during application startup and reused for incoming requests.

## Project Structure

```text
04-fastapi-demo/
├── tests/
│   └── test_api.py
├── main.py
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10+
- FastAPI
- Uvicorn
- PyTorch
- Transformers
- Pydantic
- HTTPX
- Pytest

The trained models must be available in:

```text
03-distilbert/models/
├── category_model/
└── priority_model/
```

The model weight files are excluded from Git because of their size. Train the models using Task 3 or obtain the trained artifacts separately before starting the API.

## Installation

From the repository root, activate the Task 3 virtual environment:

```powershell
.\03-distilbert\.venv\Scripts\Activate.ps1
```

Install Task 4 dependencies:

```powershell
python -m pip install -r 04-fastapi-demo/requirements.txt
```

## Run the API

From the repository root:

```powershell
python -m uvicorn main:app --app-dir 04-fastapi-demo --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

## API Documentation

Interactive Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

Alternative API documentation:

```text
http://127.0.0.1:8000/redoc
```

## Health Endpoint

**GET /health**

Example:

```bash
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{
  "status": "healthy",
  "service": "Support Ticket Classifier API"
}
```

## Prediction Endpoint

**POST /predict**

Request:

```json
{
  "text": "I was charged twice"
}
```

Example curl command:

```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d "{\"text\":\"I was charged twice\"}"
```

Example response:

```json
{
  "category": "Billing",
  "priority": "High",
  "confidence": 0.93
}
```

The confidence value is the lower of the category and priority model's maximum softmax probabilities, rounded to two decimal places.

## Input Validation

Pydantic validates incoming requests.

The API returns HTTP 422 for:

- Empty ticket text
- Whitespace-only ticket text
- Missing text field
- Invalid request structure

## Run Automated Tests

From the repository root:

```powershell
python -m pytest 04-fastapi-demo/tests -v
```

Five automated tests cover:

1. Health endpoint.
2. Valid prediction.
3. Empty ticket text.
4. Whitespace-only ticket text.
5. Missing text field.

The tests use a fake predictor to avoid loading large model files.

## Model Loading

The category and priority models are loaded during application startup through FastAPI's lifespan mechanism.

They are kept in application state and reused for predictions instead of being loaded on every request.

## Model Architecture

```text
Client Request
      |
      v
FastAPI POST /predict
      |
      v
Pydantic Validation
      |
      v
Ticket Text
      |
      +----------------------+
      |                      |
      v                      v
Category DistilBERT    Priority DistilBERT
      |                      |
      v                      v
Predicted Category     Predicted Priority
      |                      |
      +-----------+----------+
                  |
                  v
             Confidence
                  |
                  v
             JSON Response
```

## Task 4 Deliverables

- FastAPI application.
- POST /predict endpoint.
- GET /health endpoint.
- Startup model loading.
- Pydantic input validation.
- Automated TestClient tests.
- API README and curl examples.

## Running the Demo

1. Start the FastAPI server.
2. Open the Swagger documentation at `/docs`.
3. Run the health endpoint.
4. Submit a support ticket to `/predict`.
5. Demonstrate validation using empty input.
6. Run the automated test suite.
7. Explain how the Task 3 models are reused by the API.