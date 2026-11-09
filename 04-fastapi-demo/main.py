
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable

import torch
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_DIR = BASE_DIR / "03-distilbert" / "models"

CATEGORY_MODEL_DIR = MODEL_DIR / "category_model"
PRIORITY_MODEL_DIR = MODEL_DIR / "priority_model"


# --------------------------------------------------
# MODEL PREDICTOR
# --------------------------------------------------

class TicketPredictor:

    def __init__(self):
        self.device = torch.device("cpu")

        self.category_tokenizer = None
        self.category_model = None

        self.priority_tokenizer = None
        self.priority_model = None

        self.category_labels = None
        self.priority_labels = None

    @staticmethod
    def load_labels(model_dir, model):
        labels_path = model_dir / "labels.json"

        with open(labels_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        # Support common label-file formats.
        if isinstance(data, list):
            return data

        if isinstance(data, dict):

            if isinstance(data.get("classes"), list):
                return data["classes"]

            if isinstance(data.get("id2label"), dict):
                mapping = data["id2label"]

                return [
                    mapping.get(str(index), mapping.get(index))
                    for index in range(len(mapping))
                ]

            if isinstance(data.get("label2id"), dict):
                mapping = data["label2id"]

                return [
                    label
                    for label, index in sorted(
                        mapping.items(),
                        key=lambda item: int(item[1])
                    )
                ]

        # Fall back to the model configuration.
        id2label = model.config.id2label

        return [
            id2label.get(index, id2label.get(str(index)))
            for index in range(model.config.num_labels)
        ]

    def load_models(self):

        print("Loading category model...")

        self.category_tokenizer = AutoTokenizer.from_pretrained(
            str(CATEGORY_MODEL_DIR)
        )

        self.category_model = (
            AutoModelForSequenceClassification.from_pretrained(
                str(CATEGORY_MODEL_DIR)
            )
        )

        print("Loading priority model...")

        self.priority_tokenizer = AutoTokenizer.from_pretrained(
            str(PRIORITY_MODEL_DIR)
        )

        self.priority_model = (
            AutoModelForSequenceClassification.from_pretrained(
                str(PRIORITY_MODEL_DIR)
            )
        )

        self.category_model.to(self.device)
        self.priority_model.to(self.device)

        self.category_model.eval()
        self.priority_model.eval()

        self.category_labels = self.load_labels(
            CATEGORY_MODEL_DIR,
            self.category_model
        )

        self.priority_labels = self.load_labels(
            PRIORITY_MODEL_DIR,
            self.priority_model
        )

        print("Both models loaded successfully!")

    def predict_one(
        self,
        text,
        tokenizer,
        model,
        labels
    ):

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():

            outputs = model(**inputs)

            probabilities = torch.softmax(
                outputs.logits,
                dim=-1
            )

            confidence, predicted_index = torch.max(
                probabilities,
                dim=-1
            )

        index = predicted_index.item()

        return {
            "label": labels[index],
            "confidence": confidence.item()
        }

    def predict(self, text):

        category_result = self.predict_one(
            text,
            self.category_tokenizer,
            self.category_model,
            self.category_labels
        )

        priority_result = self.predict_one(
            text,
            self.priority_tokenizer,
            self.priority_model,
            self.priority_labels
        )

        # Conservative combined confidence:
        # use the lower of the two prediction probabilities.
        confidence = min(
            category_result["confidence"],
            priority_result["confidence"]
        )

        return {
            "category": category_result["label"],
            "priority": priority_result["label"],
            "confidence": round(confidence, 2)
        }


# --------------------------------------------------
# MODEL LOADER
# --------------------------------------------------

def load_predictor():

    predictor = TicketPredictor()

    predictor.load_models()

    return predictor


# --------------------------------------------------
# PYDANTIC REQUEST MODEL
# --------------------------------------------------

class PredictionRequest(BaseModel):

    text: str = Field(
        ...,
        min_length=1,
        description="Support ticket text"
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, value):

        value = value.strip()

        if not value:
            raise ValueError("Ticket text cannot be empty.")

        return value


# --------------------------------------------------
# APPLICATION FACTORY AND LIFESPAN
# --------------------------------------------------

def create_app(
    model_loader: Callable = load_predictor
):

    @asynccontextmanager
    async def lifespan(app: FastAPI):

        print("Starting Support Ticket Classifier API...")

        # Models are loaded once at startup.
        app.state.predictor = model_loader()

        yield

        app.state.predictor = None

        print("API shutting down.")

    app = FastAPI(
        title="Support Ticket Classifier API",
        description=(
            "Classifies support tickets using "
            "fine-tuned DistilBERT models."
        ),
        version="1.0.0",
        lifespan=lifespan
    )

    # --------------------------------------------------
    # HEALTH ENDPOINT
    # --------------------------------------------------

    @app.get("/health")
    def health():

        return {
            "status": "healthy",
            "service": "Support Ticket Classifier API"
        }

    # --------------------------------------------------
    # PREDICTION ENDPOINT
    # --------------------------------------------------

    @app.post("/predict")
    def predict_ticket(
        request_data: PredictionRequest,
        request: Request
    ):

        predictor = getattr(
            request.app.state,
            "predictor",
            None
        )

        if predictor is None:

            raise HTTPException(
                status_code=503,
                detail="Prediction models are not available."
            )

        try:

            result = predictor.predict(
                request_data.text
            )

            return result

        except Exception as error:

            raise HTTPException(
                status_code=500,
                detail="Prediction failed."
            ) from error

    return app


# --------------------------------------------------
# APPLICATION INSTANCE
# --------------------------------------------------

app = create_app()
