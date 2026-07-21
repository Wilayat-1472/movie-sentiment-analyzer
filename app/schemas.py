"""
schemas.py — Pydantic models for request validation and response shape.

Why Pydantic schemas?
    FastAPI uses Pydantic to automatically:
      1. Validate incoming JSON (type checking, required fields)
      2. Generate OpenAPI/Swagger docs from the schema
      3. Serialize responses to JSON

    This means if someone sends a request without "review_text",
    FastAPI returns a clear 422 error BEFORE your code even runs.
"""

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """
    What the client sends to POST /predict.

    Fields:
        review_text:  The raw movie review typed by the user.
        model_choice: Which model to use — "lstm" or "rnn".
                      Defaults to "lstm" if not provided.
    """
    review_text: str = Field(
        ...,
        min_length=1,
        description="The raw movie review text to analyze"
    )
    model_choice: str = Field(
        default="lstm",
        description="Model to use for prediction: 'lstm' or 'rnn'"
    )


class PredictionResponse(BaseModel):
    """
    What the API returns after prediction.

    Fields:
        sentiment:  The predicted label — "positive" or "negative"
        confidence: Probability score (0.0 to 1.0)
        model_used: Which model produced this prediction
    """
    sentiment: str = Field(description="Predicted sentiment: 'positive' or 'negative'")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")
    model_used: str = Field(description="Which model was used: 'lstm' or 'rnn'")


class CompareResponse(BaseModel):
    """
    Response for the /compare endpoint — runs BOTH models on the same review.
    """
    lstm_result: PredictionResponse
    rnn_result: PredictionResponse
