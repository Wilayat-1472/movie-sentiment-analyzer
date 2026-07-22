[![Movie Sentiment Analyzer](https://img.shields.io/badge/🎬-Movie%20Sentiment%20Analyzer-8b5cf6?style=for-the-badge)](https://github.com/Wilayat-1472/movie-sentiment-analyzer)

# Movie Sentiment Analyzer

*AI-powered sentiment analysis for movie reviews using LSTM and SimpleRNN deep learning models trained on the IMDB 50K dataset via Google Colab (T4 GPU).*

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Google Colab](https://img.shields.io/badge/Google%20Colab-T4%20GPU-F37626?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com)

[Overview](#-overview) • [Pipeline](#-pipeline) • [Quick Start](#-quick-start) • [API Reference](#-api-reference) • [Model Performance](#-model-performance) • [Training Details](#-training-details)

---

## 📋 Overview

A production-grade sentiment analysis system that classifies movie reviews as **positive** or **negative** using two neural architectures trained on the IMDB 50K dataset via **Google Colab (T4 GPU)**.

| Capability | Detail |
|---|---|
| **Task** | Binary Sentiment Classification (Positive / Negative) |
| **Dataset** | IMDB 50K Movie Reviews (Kaggle) — 25K positive, 25K negative |
| **Models** | LSTM (87% accuracy) · SimpleRNN (~50% — baseline comparison) |
| **Backend** | FastAPI REST API with CORS, Swagger docs, Pydantic validation |
| **Frontend** | Streamlit dark-themed UI with real-time confidence gauges |
| **Training Platform** | Google Colab (NVIDIA T4 GPU) |
| **Training Notebook** | [`fastapi.ipynb`](https://github.com/Wilayat-1472/movie-sentiment-analyzer/blob/main/fastapi.ipynb) |

---

## 🔬 Pipeline

The preprocessing pipeline mirrors the training notebook **exactly** to ensure inference matches training:

```
Raw Review Text → Lowercase → Remove HTML → Remove URLs → Remove Numbers →
Remove Punctuation → Keep Only Alphabets → Collapse Whitespace →
Remove NLTK Stopwords → Tokenize (vocab=20K) → Pad (maxlen=200) →
Model Inference → Sentiment + Confidence
```

**Inference flow:**

```
User Input (Streamlit UI) → POST /predict → FastAPI Router →
preprocess_text() → model.predict() → argmax + confidence →
JSON Response → Animated Result Card
```

**Project Structure:**

```
movie-sentiment-analyzer/
├── app/
│   ├── __init__.py           # Package init
│   ├── main.py               # FastAPI routes, startup, CORS
│   ├── model_loader.py       # Singleton model loader (loaded once at startup)
│   ├── preprocessing.py      # clean_text → tokenize → pad_sequences
│   └── schemas.py            # Pydantic request/response models
├── models/
│   ├── lstm_model.keras      # Trained LSTM (31 MB)
│   ├── rnn_model.keras       # Trained SimpleRNN (30 MB)
│   ├── tokenizer.pkl         # Keras Tokenizer — vocab 20K
│   └── label_encoder.pkl     # LabelEncoder: 0→negative, 1→positive
├── fastapi.ipynb             # Google Colab training notebook (T4 GPU)
├── streamlit_app.py          # Streamlit frontend
├── requirements.txt          # Python dependencies
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Conda (recommended) or pip
- Git

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Wilayat-1472/movie-sentiment-analyzer.git
cd movie-sentiment-analyzer

# 2. Create and activate the Conda environment
conda create -n Api_integ python=3.10 -y
conda activate Api_integ

# 3. Install dependencies
pip install -r requirements.txt
```

### Run the API

```bash
conda activate Api_integ
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`.  
Interactive Swagger docs at `http://localhost:8000/docs`.

### Run the Streamlit UI

In a separate terminal:

```bash
conda activate Api_integ
streamlit run streamlit_app.py --server.port 8501
```

Open **http://localhost:8501** in your browser.

### Usage

1. Type or paste a movie review
2. Select **LSTM** (recommended), **RNN**, or **Compare Both** mode
3. Click **Analyze Sentiment**
4. View sentiment label, confidence score, and inference time

---

## 📡 API Reference

All endpoints return JSON. The API runs on `http://localhost:8000`.

### Health

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `GET` | `/health` | — | Health check |

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true
}
```

### Predict

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `POST` | `/predict` | — | Predict sentiment using a single model |

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

### Compare

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `POST` | `/compare` | — | Run both LSTM and RNN on the same review |

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

## 📊 Model Performance

### LSTM (Recommended)

| Metric | Score |
|---|---|
| **Test Accuracy** | **87.26%** |
| Precision | 87.30% |
| Recall | 87.20% |
| F1-Score | 87.20% |

```
              precision    recall  f1-score   support

    Negative       0.87      0.87      0.87      2500
    Positive       0.87      0.87      0.87      2500

    accuracy                           0.87      5000
   macro avg       0.87      0.87      0.87      5000
```

### SimpleRNN

| Metric | Score |
|---|---|
| Test Accuracy | ~50% |
| Status | Random guessing (all-negative predictions) |

> The SimpleRNN collapsed to predicting all-negative. It is included purely for **educational comparison** against the LSTM.

---

## 🧠 Training Details

The models were trained in **Google Colab** on an **NVIDIA T4 GPU** using TensorFlow 2.15.

| Parameter | Value |
|---|---|
| **Dataset** | IMDB 50K Movie Reviews (Kaggle) |
| **Train / Val / Test** | 40K / 5K / 5K (stratified split) |
| **Vocabulary Size** | 20,000 words |
| **Max Sequence Length** | 200 tokens (post-padding, post-truncating) |
| **Embedding Dimension** | 128 |
| **Optimizer** | Adam (lr = 0.001) |
| **Loss** | Binary Crossentropy |
| **Batch Size** | 64 |
| **Epochs** | 10 (ModelCheckpoint — save best val_accuracy) |
| **Training Time** | ~15 minutes (both models combined) |
| **Platform** | Google Colab (NVIDIA T4 GPU) |
| **Notebook** | [`fastapi.ipynb`](https://github.com/Wilayat-1472/movie-sentiment-analyzer/blob/main/fastapi.ipynb) |

### LSTM Architecture

```
Embedding(20000, 128, input_length=200)
  └── LSTM(128, activation='tanh')
       └── Dropout(0.5)
            └── Dense(64, activation='relu')
                 └── Dropout(0.3)
                      └── Dense(1, activation='sigmoid')
```

**Total parameters:** ~1.2M

### SimpleRNN Architecture

```
Embedding(20000, 128, input_length=200)
  └── SimpleRNN(128, activation='tanh')
       └── Dropout(0.5)
            └── Dense(64, activation='relu')
                 └── Dropout(0.3)
                      └── Dense(1, activation='sigmoid')
```

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.10+ |
| **API Framework** | FastAPI 0.100+ |
| **UI Framework** | Streamlit 1.30+ |
| **Deep Learning** | TensorFlow 2.15 / Keras 3 |
| **Text Processing** | NLTK, scikit-learn (LabelEncoder) |
| **Data Validation** | Pydantic v2 |
| **Training Platform** | Google Colab (NVIDIA T4 GPU) |
| **Development** | Jupyter Notebook, Conda (Api_integ) |

---

## 📄 License

This project is licensed under the **MIT License**.

---

**Built with TensorFlow, FastAPI & Streamlit — trained on Google Colab T4 GPU**
