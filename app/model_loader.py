"""
model_loader.py — Load trained models and artifacts ONCE at startup.

Why load once?
    Loading a .keras model takes several seconds (reading weights from disk,
    rebuilding the computation graph). If you loaded per-request, every API
    call would take 3-5 seconds just for loading — unusable.

    Instead, we load everything into module-level variables at startup.
    model.predict() is then fast (milliseconds for a single review).

Artifacts loaded:
    1. tokenizer.pkl       — maps words → integers (fit on training vocab)
    2. label_encoder.pkl   — maps model output index → "positive"/"negative"
    3. lstm_model.keras    — trained LSTM weights
    4. rnn_model.keras     — trained SimpleRNN weights
"""

import os
import pickle
import keras
from keras.models import load_model

# ──────────────────────────────────────────────────────────────────────
# Monkeypatch Keras 3 deserialization bug
# ──────────────────────────────────────────────────────────────────────
# Keras 3 models saved in newer versions might contain 'quantization_config'
# key in layer configs. Older/different Keras 3.x installations will throw
# "Unrecognized keyword arguments passed to Layer: {'quantization_config': None}".
# This global monkeypatch filters out 'quantization_config' from kwargs.
_orig_layer_init = keras.layers.Layer.__init__
def _patched_layer_init(self, *args, **kwargs):
    kwargs.pop('quantization_config', None)
    _orig_layer_init(self, *args, **kwargs)
keras.layers.Layer.__init__ = _patched_layer_init



# ──────────────────────────────────────────────────────────────────────
# Paths — relative to project root
# ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

TOKENIZER_PATH = os.path.join(MODELS_DIR, "tokenizer.pkl")
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")
LSTM_MODEL_PATH = os.path.join(MODELS_DIR, "lstm_model.keras")
RNN_MODEL_PATH = os.path.join(MODELS_DIR, "rnn_model.keras")


# ──────────────────────────────────────────────────────────────────────
# Module-level storage — populated by load_all_artifacts()
# ──────────────────────────────────────────────────────────────────────
_artifacts = {
    "tokenizer": None,
    "label_encoder": None,
    "lstm_model": None,
    "rnn_model": None,
}


def load_all_artifacts():
    """
    Load all four artifacts into memory.
    Called once during FastAPI's startup event.
    """
    print("Loading tokenizer …")
    with open(TOKENIZER_PATH, "rb") as f:
        _artifacts["tokenizer"] = pickle.load(f)
    print("  ✓ Tokenizer loaded")

    print("Loading label encoder …")
    with open(LABEL_ENCODER_PATH, "rb") as f:
        _artifacts["label_encoder"] = pickle.load(f)
    print(f"  ✓ Label encoder loaded — classes: {list(_artifacts['label_encoder'].classes_)}")

    print("Loading LSTM model …")
    _artifacts["lstm_model"] = load_model(LSTM_MODEL_PATH)
    print("  ✓ LSTM model loaded")

    print("Loading RNN model …")
    _artifacts["rnn_model"] = load_model(RNN_MODEL_PATH)
    print("  ✓ RNN model loaded")

    print("\nAll artifacts loaded successfully!\n")


def get_tokenizer():
    """Return the loaded tokenizer."""
    return _artifacts["tokenizer"]


def get_label_encoder():
    """Return the loaded LabelEncoder."""
    return _artifacts["label_encoder"]


def get_model(model_choice: str):
    """
    Return the requested model.

    Args:
        model_choice: "lstm" or "rnn" (case-insensitive)

    Returns:
        The loaded Keras model

    Raises:
        ValueError if model_choice is invalid
    """
    key = model_choice.lower()
    if key == "lstm":
        return _artifacts["lstm_model"]
    elif key == "rnn":
        return _artifacts["rnn_model"]
    else:
        raise ValueError(
            f"Invalid model_choice '{model_choice}'. Must be 'lstm' or 'rnn'."
        )
