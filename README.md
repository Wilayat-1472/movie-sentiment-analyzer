# Movie Sentiment Analyzer

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-FF6F00?logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

AI-powered sentiment analysis for movie reviews using LSTM and SimpleRNN deep learning models trained on the IMDB 50K dataset via Google Colab (T4 GPU).

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Model Performance](#model-performance)
- [Training Details](#training-details)
- [Tech Stack](#tech-stack)

---

## Overview

This project provides a production-ready sentiment analysis pipeline with:

- **FastAPI backend** — Two trained neural network models served via REST API
- **Streamlit frontend** — Premium dark-themed UI for interactive analysis
- **Dual model comparison** — Compare LSTM vs RNN predictions side by side
- **Preprocessing pipeline** — Cleaning, tokenization, and padding matching training exactly

---

## Architecture

```
User Input  ──►  Streamlit UI  ──►  FastAPI API  ──►  Preprocessing
                         ▲                              │
                         │                              ▼
                         └────── JSON Response ◄── Model Inference
```

**Flow:**
1. User enters a movie review in the Streamlit UI
2. UI sends a POST request to the FastAPI backend
3. API cleans, tokenizes, and pads the text
4. Preprocessed input is passed to the selected Keras model
5. Prediction (sentiment + confidence) is returned as JSON
6. UI renders the result with animated cards and confidence gauges

---

## Project Structure

```
Sentiment Analyzer/
├── app/
│   ├── __init__.py          # Package init
│   ├── main.py              # FastAPI entry point & routes
│   ├── model_loader.py      # Load models/artifacts at startup
│   ├── preprocessing.py     # Text cleaning & tokenization
│   └── schemas.py           # Pydantic request/response models
├── models/
│   ├── lstm_model.keras     # Trained LSTM weights
│   ├── rnn_model.keras      # Trained SimpleRNN weights
│   ├── tokenizer.pkl        # Keras Tokenizer (vocab=20K)
│   └── label_encoder.pkl    # LabelEncoder (negative/positive)
├── fastapi.ipynb            # Google Colab training notebook (T4 GPU)
├── streamlit_app.py         # Streamlit frontend
├── requirements.txt         # Python dependencies
└── .gitignore
```

---

## Installation

### Prerequisites

- Python 3.10+
- Conda (recommended) or pip

### Setup

```bash
# Clone the repository
git clone https://github.com/Wilayat-1472/movie-sentiment-analyzer.git
cd movie-sentiment-analyzer

# Create and activate conda environment
conda create -n Api_integ python=3.10 -y
conda activate Api_integ

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### 1. Start the API server

```bash
conda activate Api_integ
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`.  
Interactive Swagger docs at `http://localhost:8000/docs`.

### 2. Start the Streamlit UI

In a new terminal:

```bash
conda activate Api_integ
streamlit run streamlit_app.py --server.port 8501
```

Open `http://localhost:8501` in your browser.

### 3. Analyze sentiment

- Type or paste a movie review
- Select **LSTM** or **RNN** model (or **Compare Both** mode)
- Click **Analyze Sentiment**
- View sentiment label, confidence score, and inference time

---

## API Reference

### `GET /health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true
}
```

### `POST /predict`

Predict sentiment of a single review.

**Request:**
```json
{
  "review_text": "This movie was absolutely amazing!",
  "model_choice": "lstm"
}
```

**Response:**
```json
{
  "sentiment": "positive",
  "confidence": 0.9987,
  "model_used": "lstm"
}
```

### `POST /compare`

Run both models on the same review.

**Request:**
```json
{
  "review_text": "The cinematography was breathtaking."
}
```

**Response:**
```json
{
  "lstm_result": {
    "sentiment": "positive",
    "confidence": 0.9972,
    "model_used": "lstm"
  },
  "rnn_result": {
    "sentiment": "positive",
    "confidence": 0.5231,
    "model_used": "rnn"
  }
}
```

---

## Model Performance

### LSTM (Recommended)

| Metric | Score |
|---|---|
| Test Accuracy | **87.26%** |
| Precision | 87.30% |
| Recall | 87.20% |
| F1-Score | 87.20% |

### SimpleRNN

The SimpleRNN model failed to learn meaningful patterns (~50% accuracy, equivalent to random guessing). It is included for comparison and educational purposes.

| Metric | Score |
|---|---|
| Test Accuracy | ~50% |
| Status | Random guessing (all-negative predictions) |

---

## Training Details

| Parameter | Value |
|---|---|
| **Dataset** | IMDB 50K Movie Reviews (Kaggle) |
| **Train / Val / Test** | 40K / 5K / 5K (stratified) |
| **Vocabulary Size** | 20,000 |
| **Max Sequence Length** | 200 tokens |
| **Embedding Dimension** | 128 |
| **Optimizer** | Adam (lr=0.001) |
| **Loss** | Binary Crossentropy |
| **Batch Size** | 64 |
| **Epochs** | 10 (early stopping via ModelCheckpoint) |
| **Training Environment** | Google Colab (T4 GPU) |
| **Notebook** | `fastapi.ipynb` — open in [Google Colab](https://colab.research.google.com/) |

### LSTM Architecture

```
Embedding(20000, 128, input_length=200)
  └── LSTM(128)
       └── Dropout(0.5)
            └── Dense(64, ReLU)
                 └── Dropout(0.3)
                      └── Dense(1, Sigmoid)
```

---

## Tech Stack

| Category | Technology |
|---|---|
| **API Framework** | FastAPI |
| **UI Framework** | Streamlit |
| **Deep Learning** | TensorFlow / Keras |
| **Text Processing** | NLTK, scikit-learn |
| **Data Validation** | Pydantic |
| **Format** | JSON via REST |
| **Training Platform** | Google Colab (T4 GPU) |
| **Environment** | Conda |

---

## License

This project is licensed under the MIT License.
