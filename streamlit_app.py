"""
streamlit_app.py — Premium Streamlit frontend for Movie Sentiment Analysis.

Run with:
    conda activate Api_integ
    streamlit run streamlit_app.py --server.port 8501
"""

import streamlit as st
import requests
import time

# ── Configuration ─────────────────────────────────────────────────────
API_BASE_URL = "http://localhost:8000"

# ── Page Config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Analyzer Pro",
    page_icon="🎬",
    layout="wide",
)

# ── Premium Dark Theme CSS ────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── Global Reset ── */
    .stApp {
        background: linear-gradient(160deg, #0a0a1a 0%, #121228 40%, #0d1117 100%);
        font-family: 'Inter', sans-serif;
    }
    .block-container { max-width: 1100px; padding-top: 2rem; }
    header[data-testid="stHeader"] { background: transparent; }

    /* ── Hero Section ── */
    .hero {
        text-align: center;
        padding: 2.5rem 1rem 1.5rem;
        position: relative;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -60px; left: 50%; transform: translateX(-50%);
        width: 400px; height: 400px;
        background: radial-gradient(circle, rgba(139,92,246,0.15) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-icon {
        font-size: 3.5rem;
        display: inline-block;
        animation: float 3s ease-in-out infinite;
        filter: drop-shadow(0 0 20px rgba(139,92,246,0.4));
    }
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-8px); }
    }
    .hero h1 {
        font-size: 2.8rem;
        font-weight: 900;
        background: linear-gradient(135deg, #a78bfa, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0.5rem 0 0.3rem;
        letter-spacing: -0.5px;
    }
    .hero .tagline {
        color: #6b7280;
        font-size: 1rem;
        font-weight: 400;
        letter-spacing: 0.5px;
    }

    /* ── Status Pill ── */
    .status-pill {
        display: inline-flex; align-items: center; gap: 8px;
        padding: 8px 20px;
        border-radius: 50px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 1rem auto;
        letter-spacing: 0.3px;
    }
    .status-online {
        background: rgba(16,185,129,0.1);
        border: 1px solid rgba(16,185,129,0.3);
        color: #34d399;
    }
    .status-offline {
        background: rgba(239,68,68,0.1);
        border: 1px solid rgba(239,68,68,0.3);
        color: #f87171;
    }
    .pulse-dot {
        width: 8px; height: 8px;
        border-radius: 50%;
        background: #34d399;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(52,211,153,0.5); }
        70% { box-shadow: 0 0 0 8px rgba(52,211,153,0); }
        100% { box-shadow: 0 0 0 0 rgba(52,211,153,0); }
    }

    /* ── Glass Card ── */
    .glass-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px;
        padding: 1.8rem;
        backdrop-filter: blur(20px);
        transition: border-color 0.3s;
    }
    .glass-card:hover { border-color: rgba(139,92,246,0.2); }
    .card-title {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #8b5cf6;
        margin-bottom: 1rem;
    }

    /* ── Textarea Styling ── */
    .stTextArea textarea {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 12px !important;
        color: #e5e7eb !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.95rem !important;
        padding: 1rem !important;
        transition: border-color 0.3s !important;
    }
    .stTextArea textarea:focus {
        border-color: rgba(139,92,246,0.5) !important;
        box-shadow: 0 0 0 3px rgba(139,92,246,0.1) !important;
    }
    .stTextArea textarea::placeholder { color: #4b5563 !important; }
    .stTextArea label { color: #9ca3af !important; font-weight: 500 !important; }

    /* ── Button ── */
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #6d28d9) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.8rem 2rem !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        letter-spacing: 0.3px !important;
        transition: all 0.3s !important;
        box-shadow: 0 4px 15px rgba(124,58,237,0.3) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(124,58,237,0.4) !important;
    }
    .stButton > button:active { transform: translateY(0px) !important; }
    .stButton > button[disabled] {
        background: rgba(255,255,255,0.05) !important;
        box-shadow: none !important;
        color: #4b5563 !important;
    }

    /* ── Radio & Select ── */
    .stRadio label, .stSelectbox label { color: #9ca3af !important; font-weight: 500 !important; }
    .stRadio > div { gap: 0.3rem; }
    .stRadio [data-testid="stMarkdownContainer"] p { color: #d1d5db !important; }
    .stSelectbox [data-testid="stMarkdownContainer"] p { color: #d1d5db !important; }

    /* ── Result Cards ── */
    .result-card {
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        position: relative;
        overflow: hidden;
        animation: slideUp 0.5s ease-out;
    }
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .result-positive {
        background: linear-gradient(135deg, rgba(16,185,129,0.12), rgba(52,211,153,0.06));
        border: 1px solid rgba(16,185,129,0.25);
    }
    .result-negative {
        background: linear-gradient(135deg, rgba(239,68,68,0.12), rgba(248,113,113,0.06));
        border: 1px solid rgba(239,68,68,0.25);
    }
    .result-emoji { font-size: 3rem; margin-bottom: 0.5rem; }
    .result-label {
        font-size: 1.6rem;
        font-weight: 800;
        margin: 0.3rem 0;
        letter-spacing: -0.3px;
    }
    .result-label.positive { color: #34d399; }
    .result-label.negative { color: #f87171; }

    /* ── Confidence Gauge ── */
    .gauge-container {
        margin: 1.2rem auto;
        max-width: 250px;
    }
    .gauge-bar-bg {
        height: 8px;
        background: rgba(255,255,255,0.06);
        border-radius: 10px;
        overflow: hidden;
    }
    .gauge-bar-fill {
        height: 100%;
        border-radius: 10px;
        animation: fillBar 1s ease-out;
    }
    @keyframes fillBar {
        from { width: 0%; }
    }
    .gauge-fill-positive { background: linear-gradient(90deg, #059669, #34d399); }
    .gauge-fill-negative { background: linear-gradient(90deg, #dc2626, #f87171); }
    .gauge-value {
        font-size: 2rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
        margin-top: 0.5rem;
    }
    .gauge-value.positive { color: #34d399; }
    .gauge-value.negative { color: #f87171; }
    .gauge-subtitle {
        font-size: 0.75rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }

    /* ── Model Badge ── */
    .model-badge {
        display: inline-block;
        padding: 5px 14px;
        border-radius: 50px;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-top: 0.8rem;
    }
    .badge-lstm {
        background: rgba(59,130,246,0.15);
        border: 1px solid rgba(59,130,246,0.3);
        color: #60a5fa;
    }
    .badge-rnn {
        background: rgba(168,85,247,0.15);
        border: 1px solid rgba(168,85,247,0.3);
        color: #c084fc;
    }

    /* ── Agreement Banner ── */
    .agree-banner, .disagree-banner {
        padding: 1rem 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-weight: 600;
        font-size: 0.9rem;
        margin-top: 1rem;
        animation: slideUp 0.6s ease-out;
    }
    .agree-banner {
        background: rgba(16,185,129,0.08);
        border: 1px solid rgba(16,185,129,0.2);
        color: #34d399;
    }
    .disagree-banner {
        background: rgba(251,191,36,0.08);
        border: 1px solid rgba(251,191,36,0.2);
        color: #fbbf24;
    }

    /* ── Timing Badge ── */
    .timing-badge {
        text-align: center;
        color: #4b5563;
        font-size: 0.8rem;
        font-family: 'JetBrains Mono', monospace;
        margin-top: 0.8rem;
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.03) !important;
        border-radius: 12px !important;
        color: #9ca3af !important;
    }

    /* ── Pipeline Steps ── */
    .pipeline-step {
        display: flex; align-items: center; gap: 12px;
        padding: 10px 0;
        color: #9ca3af;
        font-size: 0.85rem;
    }
    .step-num {
        width: 28px; height: 28px;
        border-radius: 8px;
        background: rgba(139,92,246,0.12);
        border: 1px solid rgba(139,92,246,0.25);
        color: #a78bfa;
        display: flex; align-items: center; justify-content: center;
        font-weight: 700;
        font-size: 0.75rem;
        flex-shrink: 0;
    }

    /* ── Divider ── */
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(139,92,246,0.2), transparent);
        margin: 1.5rem 0;
    }

    /* ── Footer ── */
    .footer {
        text-align: center;
        color: #374151;
        font-size: 0.75rem;
        padding: 2rem 0 1rem;
        letter-spacing: 0.5px;
    }

    /* ── Sample Review Chips ── */
    .sample-chip {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 12px 16px;
        color: #9ca3af;
        font-size: 0.85rem;
        font-style: italic;
        margin: 6px 0;
        cursor: default;
        transition: border-color 0.3s;
    }
    .sample-chip:hover { border-color: rgba(139,92,246,0.3); }
    .sample-label {
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 1rem 0 0.5rem;
    }
    .sample-pos { color: #34d399; }
    .sample-neg { color: #f87171; }

    /* Hide Streamlit defaults */
    #MainMenu, footer, .stDeployButton { display: none !important; }
    .stAlert { border-radius: 12px !important; }
</style>
""", unsafe_allow_html=True)


# ── Helper Functions ──────────────────────────────────────────────────
def check_api_health():
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return resp.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


def predict_sentiment(review_text: str, model_choice: str) -> dict:
    resp = requests.post(
        f"{API_BASE_URL}/predict",
        json={"review_text": review_text, "model_choice": model_choice},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def compare_models(review_text: str) -> dict:
    resp = requests.post(
        f"{API_BASE_URL}/compare",
        json={"review_text": review_text},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def render_result_card(result: dict):
    """Render a premium sentiment result card."""
    sentiment = result["sentiment"]
    confidence = result["confidence"]
    model_used = result["model_used"].upper()
    is_pos = sentiment == "positive"

    css = "result-positive" if is_pos else "result-negative"
    emoji = "🎉" if is_pos else "💔"
    label_css = "positive" if is_pos else "negative"
    gauge_css = "gauge-fill-positive" if is_pos else "gauge-fill-negative"
    badge_css = "badge-lstm" if result["model_used"] == "lstm" else "badge-rnn"
    pct = confidence * 100

    st.markdown(f"""
    <div class="result-card {css}">
        <div class="result-emoji">{emoji}</div>
        <div class="result-label {label_css}">{sentiment.upper()}</div>
        <div class="gauge-container">
            <div class="gauge-value {label_css}">{pct:.1f}%</div>
            <div class="gauge-subtitle">confidence</div>
            <div style="margin-top:8px">
                <div class="gauge-bar-bg">
                    <div class="gauge-bar-fill {gauge_css}" style="width:{pct}%"></div>
                </div>
            </div>
        </div>
        <span class="model-badge {badge_css}">{model_used} Model</span>
    </div>
    """, unsafe_allow_html=True)


# ── Hero Section ──────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-icon">🎬</div>
    <h1>Sentiment Analyzer</h1>
    <p class="tagline">AI-Powered Movie Sentiment Analysis</p>
</div>
""", unsafe_allow_html=True)

# ── API Status ────────────────────────────────────────────────────────
api_healthy = check_api_health()
if api_healthy:
    st.markdown("""
    <div style="text-align:center">
        <div class="status-pill status-online">
            <div class="pulse-dot"></div>
            API ENGINE ONLINE
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="text-align:center">
        <div class="status-pill status-offline">
            ⚠ API ENGINE OFFLINE
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.error(
        "**Cannot connect to the API server.**\n\n"
        "```bash\n"
        "conda activate Api_integ\n"
        "uvicorn app.main:app --host 0.0.0.0 --port 8000\n"
        "```"
    )
    st.stop()

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# ── Main Layout ───────────────────────────────────────────────────────
col_input, col_config = st.columns([3, 1.2], gap="large")

with col_config:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">⚙ Configuration</div>', unsafe_allow_html=True)

    mode = st.radio(
        "Analysis mode",
        options=["Single Model", "Compare Both"],
        index=0,
        label_visibility="collapsed",
    )

    if mode == "Single Model":
        model_choice = st.selectbox(
            "Model",
            options=["lstm", "rnn"],
            format_func=lambda x: "🧠 LSTM Network" if x == "lstm" else "🔄 SimpleRNN",
        )

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🔬 Pipeline</div>', unsafe_allow_html=True)

    steps = [
        "Clean raw text",
        "Tokenize words",
        "Pad to 200 tokens",
        "Neural network inference",
        "Decode prediction",
    ]
    for i, step in enumerate(steps, 1):
        st.markdown(f"""
        <div class="pipeline-step">
            <div class="step-num">{i}</div>
            <span>{step}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

with col_input:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">✏ Write Your Review</div>', unsafe_allow_html=True)

    review_text = st.text_area(
        "Movie review",
        height=180,
        placeholder="The cinematography was breathtaking, every frame felt like a painting. The storyline kept me engaged from start to finish…",
        label_visibility="collapsed",
    )

    analyze_btn = st.button(
        "⚡ Analyze Sentiment",
        type="primary",
        use_container_width=True,
        disabled=len(review_text.strip()) == 0,
    )

    st.markdown('</div>', unsafe_allow_html=True)


# ── Results Section ───────────────────────────────────────────────────
if analyze_btn and review_text.strip():
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="card-title" style="text-align:center; font-size:0.85rem;">📊 Analysis Results</div>', unsafe_allow_html=True)

    try:
        if mode == "Single Model":
            with st.spinner(""):
                start = time.time()
                result = predict_sentiment(review_text, model_choice)
                elapsed = time.time() - start

            col_pad1, col_res, col_pad2 = st.columns([1, 2, 1])
            with col_res:
                render_result_card(result)
                st.markdown(f'<div class="timing-badge">⏱ {elapsed:.2f}s</div>', unsafe_allow_html=True)

        else:
            with st.spinner(""):
                start = time.time()
                comparison = compare_models(review_text)
                elapsed = time.time() - start

            col_lstm, col_rnn = st.columns(2, gap="medium")

            with col_lstm:
                st.markdown('<div class="card-title" style="text-align:center">🧠 LSTM</div>', unsafe_allow_html=True)
                render_result_card(comparison["lstm_result"])

            with col_rnn:
                st.markdown('<div class="card-title" style="text-align:center">🔄 SimpleRNN</div>', unsafe_allow_html=True)
                render_result_card(comparison["rnn_result"])

            # Agreement banner
            if comparison["lstm_result"]["sentiment"] == comparison["rnn_result"]["sentiment"]:
                st.markdown('<div class="agree-banner">✅ Both models agree on the sentiment</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="disagree-banner">⚠ Models disagree — this review may be ambiguous</div>', unsafe_allow_html=True)

            st.markdown(f'<div class="timing-badge">⏱ {elapsed:.2f}s total</div>', unsafe_allow_html=True)

    except requests.exceptions.HTTPError as e:
        st.error(f"API error: {e.response.text}")
    except requests.exceptions.ConnectionError:
        st.error("Lost connection to the API server.")
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")


# ── Sample Reviews ────────────────────────────────────────────────────
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

with st.expander("💡 Sample reviews to try"):
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown('<div class="sample-label sample-pos">👍 Positive</div>', unsafe_allow_html=True)
        for r in [
            "This movie was absolutely amazing, the acting was superb and the storyline was captivating from start to finish!",
            "A masterpiece of cinema. Beautiful direction, powerful performances, and an unforgettable soundtrack.",
            "One of the best films I've ever seen. Every scene was perfectly crafted.",
        ]:
            st.markdown(f'<div class="sample-chip">"{r}"</div>', unsafe_allow_html=True)

    with col_s2:
        st.markdown('<div class="sample-label sample-neg">👎 Negative</div>', unsafe_allow_html=True)
        for r in [
            "Terrible movie, waste of time. The plot made no sense and the acting was wooden.",
            "I couldn't even finish watching this disaster. Boring, predictable, and poorly written.",
            "The worst film of the year. Bad acting, terrible script, and horrible special effects.",
        ]:
            st.markdown(f'<div class="sample-chip">"{r}"</div>', unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────
st.markdown(
    '<div class="footer">SENTIMENT ANALYZER · LSTM & RNN Models · Trained on IMDB 50K Dataset · Built with FastAPI + Streamlit</div>',
    unsafe_allow_html=True,
)
