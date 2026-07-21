"""
preprocessing.py — Text cleaning, tokenization, and padding for inference.

CRITICAL: Every step here MUST exactly mirror the training notebook.
If clean_text() differs even slightly from training, the tokenizer will
map words to wrong indices and predictions become garbage — silently.

Training config (verified from notebook):
    VOCAB_SIZE  = 20000
    MAX_LENGTH  = 200
    padding     = "post"
    truncating  = "post"
    oov_token   = "<OOV>"
"""

import re
import string

import nltk
import numpy as np
from keras.utils import pad_sequences

# ──────────────────────────────────────────────────────────────────────
# Constants — must match training notebook EXACTLY
# ──────────────────────────────────────────────────────────────────────
MAX_LENGTH = 200
PADDING = "post"
TRUNCATING = "post"


def ensure_nltk_data():
    """
    Download NLTK stopwords if not already present.
    Safe to call multiple times — nltk.download() is idempotent.
    This matters because your deployment env won't have Colab's packages.
    """
    nltk.download("stopwords", quiet=True)


# ──────────────────────────────────────────────────────────────────────
# Build stop_words set — must happen AFTER ensure_nltk_data() is called
# ──────────────────────────────────────────────────────────────────────
_stop_words = None


def _get_stop_words():
    """Lazy-load stop words so the module can be imported before NLTK data exists."""
    global _stop_words
    if _stop_words is None:
        from nltk.corpus import stopwords
        _stop_words = set(stopwords.words("english"))
    return _stop_words


# ──────────────────────────────────────────────────────────────────────
# clean_text() — EXACT replica from notebook
# ──────────────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """
    Replicate the notebook's clean_text() function byte-for-byte.

    Steps (in order — order matters!):
      1. lowercase
      2. remove HTML tags
      3. remove URLs
      4. remove numbers
      5. remove punctuation
      6. keep only alphabets
      7. collapse whitespace
      8. remove NLTK English stopwords
    """
    stop_words = _get_stop_words()

    # lowercase
    text = text.lower()

    # remove html tags
    text = re.sub(r'<.*?>', ' ', text)

    # remove urls
    text = re.sub(r'http\S+|www\S+', ' ', text)

    # remove numbers
    text = re.sub(r'\d+', ' ', text)

    # remove punctuation
    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )

    # keep only alphabets
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # remove stopwords
    words = [
        word
        for word in text.split()
        if word not in stop_words
    ]

    return " ".join(words)


# ──────────────────────────────────────────────────────────────────────
# preprocess_text() — full pipeline: clean → tokenize → pad
# ──────────────────────────────────────────────────────────────────────
def preprocess_text(text: str, tokenizer) -> np.ndarray:
    """
    Take raw user input and return a numpy array ready for model.predict().

    Args:
        text:      raw review string from the user
        tokenizer: the SAME tokenizer object that was fit on training data

    Returns:
        np.ndarray of shape (1, MAX_LENGTH) — one padded sequence

    Teaching notes:
        - texts_to_sequences([cleaned]) needs a LIST of strings, not a bare
          string. Pass a bare string and the tokenizer iterates over characters.
        - pad_sequences returns shape (batch, MAX_LENGTH) — that's exactly
          what model.predict() expects (batch dimension first).
    """
    # Step 1: clean — must be identical to training
    cleaned = clean_text(text)

    # Step 2: tokenize — note the [list] brackets around cleaned text
    sequence = tokenizer.texts_to_sequences([cleaned])

    # Step 3: pad — same maxlen, padding, truncating as training
    padded = pad_sequences(
        sequence,
        maxlen=MAX_LENGTH,
        padding=PADDING,
        truncating=TRUNCATING
    )

    return padded
