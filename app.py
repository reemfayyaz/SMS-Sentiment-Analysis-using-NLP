import os
import warnings
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st
from sklearn.exceptions import InconsistentVersionWarning


# =========================================================
# Application Configuration
# =========================================================

st.set_page_config(
    page_title="Sentiment Intelligence",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_NAME = "Sentiment Intelligence"
APP_VERSION = "1.0.0"

# These match the uploaded files.
MODEL_CANDIDATES = [
    "model (1).pkl",
    "model.pkl",
]

VECTORIZER_CANDIDATES = [
    "vectorizer (2).pkl",
    "vectorizer.pkl",
]

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


# =========================================================
# Professional Styling
# =========================================================

st.markdown(
    """
    <style>
        /* Main page */
        .stApp {
            background:
                radial-gradient(circle at 10% 0%, rgba(99, 102, 241, 0.10), transparent 28%),
                radial-gradient(circle at 90% 10%, rgba(14, 165, 233, 0.08), transparent 25%),
                #f7f8fc;
        }

        .block-container {
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        /* Hero */
        .hero {
            padding: 2rem 2.2rem;
            border-radius: 24px;
            background: linear-gradient(135deg, #111827 0%, #312e81 55%, #1d4ed8 100%);
            color: white;
            box-shadow: 0 18px 45px rgba(15, 23, 42, 0.18);
            margin-bottom: 1.6rem;
        }

        .hero-badge {
            display: inline-block;
            padding: 0.35rem 0.75rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.12);
            border: 1px solid rgba(255,255,255,0.20);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            margin-bottom: 0.8rem;
        }

        .hero h1 {
            margin: 0;
            font-size: 2.35rem;
            font-weight: 800;
            letter-spacing: -0.03em;
        }

        .hero p {
            margin: 0.7rem 0 0 0;
            max-width: 760px;
            color: rgba(255,255,255,0.86);
            font-size: 1rem;
            line-height: 1.65;
        }

        /* Cards */
        .card {
            background: rgba(255,255,255,0.92);
            border: 1px solid #e5e7eb;
            border-radius: 20px;
            padding: 1.3rem 1.4rem;
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.06);
        }

        .section-title {
            font-size: 1.12rem;
            font-weight: 800;
            color: #111827;
            margin-bottom: 0.2rem;
        }

        .section-subtitle {
            color: #6b7280;
            font-size: 0.90rem;
            margin-bottom: 0.8rem;
        }

        /* Result cards */
        .result-positive,
        .result-negative,
        .result-neutral {
            border-radius: 20px;
            padding: 1.35rem 1.5rem;
            color: white;
            margin: 0.4rem 0 1rem 0;
        }

        .result-positive {
            background: linear-gradient(135deg, #047857, #10b981);
        }

        .result-negative {
            background: linear-gradient(135deg, #b91c1c, #ef4444);
        }

        .result-neutral {
            background: linear-gradient(135deg, #475569, #64748b);
        }

        .result-label {
            font-size: 0.80rem;
            text-transform: uppercase;
            letter-spacing: 0.09em;
            opacity: 0.88;
            font-weight: 700;
        }

        .result-value {
            font-size: 2rem;
            font-weight: 800;
            margin-top: 0.2rem;
        }

        .result-confidence {
            font-size: 0.95rem;
            margin-top: 0.3rem;
            opacity: 0.92;
        }

        /* Metrics */
        div[data-testid="stMetric"] {
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 16px;
            padding: 0.9rem 1rem;
            box-shadow: 0 6px 20px rgba(15, 23, 42, 0.04);
        }

        /* Inputs */
        div[data-testid="stTextArea"] textarea {
            border-radius: 14px;
            border: 1px solid #d1d5db;
            background: #ffffff;
            font-size: 1rem;
            line-height: 1.55;
        }

        div[data-testid="stTextArea"] textarea:focus {
            border-color: #4f46e5;
            box-shadow: 0 0 0 1px #4f46e5;
        }

        /* Buttons */
        .stButton > button {
            border-radius: 12px;
            min-height: 46px;
            font-weight: 700;
            border: none;
        }

        div.stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #4f46e5, #2563eb);
            color: white;
            box-shadow: 0 8px 20px rgba(79, 70, 229, 0.22);
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid #e5e7eb;
        }

        .sidebar-brand {
            padding: 0.8rem 0 0.4rem 0;
            font-size: 1.2rem;
            font-weight: 800;
            color: #111827;
        }

        .small-note {
            color: #6b7280;
            font-size: 0.82rem;
            line-height: 1.55;
        }

        .footer {
            text-align: center;
            color: #9ca3af;
            font-size: 0.78rem;
            padding-top: 1.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# Model Utilities
# =========================================================

def find_existing_file(candidates):
    """Return the first existing file from a list of possible names."""
    for filename in candidates:
        if os.path.exists(filename):
            return filename
    return None


@st.cache_resource(show_spinner=False)
def load_assets():
    """Load the trained model and TF-IDF vectorizer."""
    model_path = find_existing_file(MODEL_CANDIDATES)
    vectorizer_path = find_existing_file(VECTORIZER_CANDIDATES)

    if not model_path:
        raise FileNotFoundError(
            "Model file not found. Expected one of: "
            + ", ".join(MODEL_CANDIDATES)
        )

    if not vectorizer_path:
        raise FileNotFoundError(
            "Vectorizer file not found. Expected one of: "
            + ", ".join(VECTORIZER_CANDIDATES)
        )

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    return model, vectorizer, model_path, vectorizer_path


def predict_sentiment(text, model, vectorizer):
    """Transform text and return prediction plus class probabilities."""
    cleaned_text = text.strip()

    if not cleaned_text:
        raise ValueError("Please enter some text before analyzing.")

    features = vectorizer.transform([cleaned_text])
    prediction = str(model.predict(features)[0])

    probabilities = {}
    if hasattr(model, "predict_proba"):
        values = model.predict_proba(features)[0]
        probabilities = {
            str(label): float(probability)
            for label, probability in zip(model.classes_, values)
        }

    confidence = probabilities.get(prediction, 0.0)

    return prediction, confidence, probabilities


def result_card(label, confidence):
    """Return styled HTML for the prediction result."""
    normalized = label.lower()

    if normalized == "positive":
        css_class = "result-positive"
        icon = "😊"
    elif normalized == "negative":
        css_class = "result-negative"
        icon = "😟"
    else:
        css_class = "result-neutral"
        icon = "😐"

    return f"""
    <div class="{css_class}">
        <div class="result-label">Detected Sentiment</div>
        <div class="result-value">{icon} {label}</div>
        <div class="result-confidence">
            Model confidence: <strong>{confidence * 100:.1f}%</strong>
        </div>
    </div>
    """


def add_to_history(text, prediction, confidence):
    """Store recent analyses in the current Streamlit session."""
    if "history" not in st.session_state:
        st.session_state.history = []

    st.session_state.history.insert(
        0,
        {
            "Time": datetime.now().strftime("%H:%M:%S"),
            "Text": text.strip(),
            "Sentiment": prediction,
            "Confidence": f"{confidence * 100:.1f}%",
        },
    )

    st.session_state.history = st.session_state.history[:10]


# =========================================================
# Session State
# =========================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "example_text" not in st.session_state:
    st.session_state.example_text = ""


# =========================================================
# Load Model
# =========================================================

try:
    model, vectorizer, model_path, vectorizer_path = load_assets()
    model_ready = True
except Exception as exc:
    model_ready = False
    load_error = str(exc)


# =========================================================
# Sidebar
# =========================================================

with st.sidebar:
    st.markdown('<div class="sidebar-brand">💬 Sentiment Intelligence</div>', unsafe_allow_html=True)
    st.caption(f"Version {APP_VERSION}")

    st.divider()

    st.markdown("### Model Status")

    if model_ready:
        st.success("Model loaded successfully")
        st.caption(f"Model: `{model_path}`")
        st.caption(f"Vectorizer: `{vectorizer_path}`")

        if hasattr(model, "classes_"):
            st.markdown("**Supported classes**")
            for label in model.classes_:
                st.markdown(f"- {label}")
    else:
        st.error("Model unavailable")

    st.divider()

    st.markdown("### Try an Example")

    examples = {
        "Positive": "I love this product. The experience was fantastic and wonderful.",
        "Neutral": "The meeting starts on Monday at 10 and the train is on time.",
        "Negative": "This was the worst purchase. The product arrived damaged and I am disappointed.",
    }

    selected_example = st.selectbox(
        "Choose a sample",
        options=["Select an example"] + list(examples.keys()),
        label_visibility="collapsed",
    )

    if selected_example != "Select an example":
        if st.button("Use this example", use_container_width=True):
            st.session_state.example_text = examples[selected_example]

    st.divider()

    st.markdown(
        """
        <div class="small-note">
        <strong>How it works</strong><br>
        Your text is converted into TF-IDF features and classified by a
        Multinomial Naive Bayes model as Positive, Neutral, or Negative.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# Main Header
# =========================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-badge">AI-POWERED TEXT ANALYTICS</div>
        <h1>Sentiment Intelligence</h1>
        <p>
            Analyze written feedback, reviews, comments, and short messages
            with a clean machine-learning workflow and an easy-to-understand
            confidence breakdown.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# Error State
# =========================================================

if not model_ready:
    st.error(
        "The application could not load the required machine-learning files."
    )
    st.code(load_error)
    st.info(
        "Place the model and vectorizer in the same folder as app.py. "
        "Supported filenames are: "
        f"{', '.join(MODEL_CANDIDATES)} and {', '.join(VECTORIZER_CANDIDATES)}."
    )
    st.stop()


# =========================================================
# Main Application
# =========================================================

left_col, right_col = st.columns([1.35, 1], gap="large")

with left_col:
    st.markdown(
        """
        <div class="section-title">Analyze Text</div>
        <div class="section-subtitle">
            Enter a sentence, customer review, comment, or short paragraph.
        </div>
        """,
        unsafe_allow_html=True,
    )

    text_input = st.text_area(
        "Text to analyze",
        value=st.session_state.example_text,
        height=220,
        max_chars=3000,
        placeholder="Example: I really enjoyed this product. The service was excellent...",
        label_visibility="collapsed",
    )

    char_count = len(text_input)
    word_count = len(text_input.split()) if text_input.strip() else 0

    stat1, stat2, stat3 = st.columns(3)
    stat1.metric("Characters", char_count)
    stat2.metric("Words", word_count)
    stat3.metric("Max Length", "3,000")

    button_col1, button_col2 = st.columns([2, 1])

    with button_col1:
        analyze_clicked = st.button(
            "Analyze Sentiment",
            type="primary",
            use_container_width=True,
            disabled=not text_input.strip(),
        )

    with button_col2:
        clear_clicked = st.button(
            "Clear",
            use_container_width=True,
        )

    if clear_clicked:
        st.session_state.example_text = ""
        st.rerun()

with right_col:
    st.markdown(
        """
        <div class="section-title">Prediction Result</div>
        <div class="section-subtitle">
            The strongest predicted class and probability distribution appear here.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if analyze_clicked:
        try:
            prediction, confidence, probabilities = predict_sentiment(
                text_input,
                model,
                vectorizer,
            )

            add_to_history(text_input, prediction, confidence)

            st.markdown(
                result_card(prediction, confidence),
                unsafe_allow_html=True,
            )

            st.markdown("#### Confidence by Class")

            probability_df = pd.DataFrame(
                {
                    "Sentiment": list(probabilities.keys()),
                    "Confidence": [
                        round(value * 100, 2)
                        for value in probabilities.values()
                    ],
                }
            ).sort_values("Confidence", ascending=False)

            st.bar_chart(
                probability_df.set_index("Sentiment"),
                y="Confidence",
                height=250,
            )

            st.caption(
                "Confidence values show the model's relative probability for each class."
            )

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

    else:
        st.info(
            "Enter text on the left and select **Analyze Sentiment** to view the result."
        )

        st.markdown(
            """
            **Best results:** use clear text containing meaningful words
            that express an opinion, reaction, experience, or factual statement.
            """
        )


# =========================================================
# Prediction History
# =========================================================

st.divider()

history_header, history_action = st.columns([4, 1])

with history_header:
    st.markdown("### Recent Analysis")
    st.caption("Your latest predictions from this browser session.")

with history_action:
    if st.session_state.history:
        if st.button("Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()

if st.session_state.history:
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Text": st.column_config.TextColumn("Text", width="large"),
            "Sentiment": st.column_config.TextColumn("Sentiment", width="small"),
            "Confidence": st.column_config.TextColumn("Confidence", width="small"),
        },
    )
else:
    st.caption("No predictions yet. Your completed analyses will appear here.")


# =========================================================
# Footer
# =========================================================

st.markdown(
    f"""
    <div class="footer">
        {APP_NAME} • Machine Learning Sentiment Classification • v{APP_VERSION}
    </div>
    """,
    unsafe_allow_html=True,
)
