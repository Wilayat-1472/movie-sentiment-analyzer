"""
main.py — FastAPI application entry point.

This file handles ONLY HTTP concerns: routes, request/response, startup.
All business logic (cleaning, tokenizing, model inference) lives in
preprocessing.py and model_loader.py.

Why separate?
    If you ever swap FastAPI for Flask, Django, or a CLI tool,
    you only rewrite this file — the model + preprocessing code stays.
    That's what "production-grade" actually means in practice.

Run with:
    conda activate Api_integ
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.model_loader import load_all_artifacts, get_tokenizer, get_label_encoder, get_model
from app.preprocessing import ensure_nltk_data, preprocess_text
from app.schemas import PredictionRequest, PredictionResponse, CompareResponse


# ──────────────────────────────────────────────────────────────────────
# Create the FastAPI app
# ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Movie Sentiment Analysis API",
    description=(
        "Predict movie review sentiment (positive/negative) using "
        "LSTM or SimpleRNN models trained on the IMDB 50K dataset."
    ),
    version="1.0.0",
)

# Allow Streamlit (or any frontend) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────────
# Startup event — runs once when the server starts
# ──────────────────────────────────────────────────────────────────────
@app.on_event("startup")
def startup_event():
    """
    1. Download NLTK stopwords (idempotent — safe to run every time)
    2. Load all model artifacts into memory

    This runs ONCE at startup, not per-request.
    If you loaded models per-request, each call would take 3-5 seconds.
    """
    print("\n🚀 Starting up Movie Sentiment Analysis API …\n")
    ensure_nltk_data()
    load_all_artifacts()
    print("✅ API is ready to serve predictions!\n")


# ──────────────────────────────────────────────────────────────────────
# Helper: run prediction on a single review with a given model
# ──────────────────────────────────────────────────────────────────────
def _predict(review_text: str, model_choice: str) -> PredictionResponse:
    """
    Core prediction logic, extracted so both /predict and /compare can use it.

    How the model output works:
        - The model has Dense(1, sigmoid) → outputs a single float in [0, 1]
        - Values close to 1.0 → "positive"
        - Values close to 0.0 → "negative"
        - The label_encoder maps index 0 → "negative", 1 → "positive"
        - We threshold at 0.5 (standard for binary sigmoid)
    """
    # Validate model choice
    if model_choice.lower() not in ("lstm", "rnn"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model_choice '{model_choice}'. Must be 'lstm' or 'rnn'."
        )

    # Get loaded artifacts
    tokenizer = get_tokenizer()
    label_encoder = get_label_encoder()
    model = get_model(model_choice)

    # Preprocess: clean → tokenize → pad (mirrors training exactly)
    padded_input = preprocess_text(review_text, tokenizer)

    # Predict: returns shape (1, 1) — one sample, one sigmoid output
    raw_prediction = model.predict(padded_input, verbose=0)
    probability = float(raw_prediction[0][0])

    # Decode: sigmoid output → class index → label string
    predicted_index = int(np.round(probability))
    sentiment_label = label_encoder.inverse_transform([predicted_index])[0]

    # Confidence: how sure the model is about its prediction
    # If positive (prob > 0.5), confidence = prob
    # If negative (prob < 0.5), confidence = 1 - prob
    confidence = probability if predicted_index == 1 else 1 - probability

    return PredictionResponse(
        sentiment=sentiment_label,
        confidence=round(confidence, 4),
        model_used=model_choice.lower(),
    )


# ──────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    """Root endpoint — API info."""
    return {
        "message": "Movie Sentiment Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST — Predict sentiment of a single review",
            "/compare": "POST — Compare LSTM vs RNN predictions side by side",
            "/health":  "GET  — Health check",
            "/docs":    "GET  — Swagger UI (interactive API docs)",
        }
    }


@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy", "models_loaded": True}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Predict sentiment of a movie review.

    - **review_text**: The raw review text to analyze
    - **model_choice**: Which model to use — "lstm" (default) or "rnn"
    """
    return _predict(request.review_text, request.model_choice)


@app.post("/compare", response_model=CompareResponse)
def compare(request: PredictionRequest):
    """
    Run BOTH models on the same review and return results side by side.

    The model_choice field in the request is ignored — both models run.
    """
    lstm_result = _predict(request.review_text, "lstm")
    rnn_result = _predict(request.review_text, "rnn")

    return CompareResponse(
        lstm_result=lstm_result,
        rnn_result=rnn_result,
    )
