"""
Industry-Level Sentiment Analysis Application
=============================================

A production-oriented Streamlit interface for a scikit-learn sentiment model.

The supplied model is a MultinomialNB classifier trained on 59 numeric features
with the classes: Negative, Neutral, Positive.

IMPORTANT:
Raw text prediction requires the SAME fitted vectorizer/preprocessor that was
used during model training. If a compatible vectorizer is placed next to this
file, the application automatically enables text prediction. Without it, the
application provides a safe 59-feature prediction interface instead of making
invalid predictions.

Recommended project structure:
    app.py
    requirements.txt
    README.md
    sentiment_model.pkl
    vectorizer.pkl          # optional but required for raw-text prediction
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Application configuration
# ---------------------------------------------------------------------------

APP_NAME = "Sentiment Intelligence"
APP_VERSION = "1.0.0"
APP_ICON = "💬"

BASE_DIR = Path(__file__).resolve().parent
MODEL_ENV = os.getenv("MODEL_PATH", "").strip()
VECTORIZER_ENV = os.getenv("VECTORIZER_PATH", "").strip()

MODEL_CANDIDATES = [
    MODEL_ENV,
    "sentiment_model.pkl",
    "model.pkl",
    "sentiment_classifier.pkl",
]

VECTORIZER_CANDIDATES = [
    VECTORIZER_ENV,
    "vectorizer.pkl",
    "tfidf_vectorizer.pkl",
    "count_vectorizer.pkl",
    "preprocessor.pkl",
]

EXPECTED_FEATURES = 59

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger(APP_NAME)


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title=f"{APP_NAME} | AI Sentiment Analysis",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at top right, rgba(76, 110, 245, 0.08), transparent 28rem),
                radial-gradient(circle at bottom left, rgba(18, 184, 134, 0.06), transparent 26rem);
        }

        .block-container {
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .hero {
            padding: 2.3rem 2.4rem;
            border-radius: 24px;
            background: linear-gradient(135deg, #111827 0%, #1e293b 55%, #26376b 100%);
            color: white;
            margin-bottom: 1.4rem;
            box-shadow: 0 18px 45px rgba(15, 23, 42, 0.18);
        }

        .hero-badge {
            display: inline-block;
            padding: 0.35rem 0.75rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.12);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            margin-bottom: 0.9rem;
        }

        .hero h1 {
            margin: 0;
            font-size: clamp(2rem, 5vw, 3.45rem);
            line-height: 1.05;
            letter-spacing: -0.04em;
        }

        .hero p {
            margin: 0.8rem 0 0 0;
            max-width: 760px;
            color: #dbe4ff;
            font-size: 1.03rem;
            line-height: 1.65;
        }

        .status-card, .result-card {
            border: 1px solid rgba(148, 163, 184, 0.24);
            border-radius: 18px;
            padding: 1.2rem 1.25rem;
            background: rgba(255, 255, 255, 0.82);
            box-shadow: 0 8px 25px rgba(15, 23, 42, 0.05);
        }

        .result-card {
            text-align: center;
            padding: 1.7rem;
        }

        .result-label {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            opacity: 0.65;
            font-weight: 800;
        }

        .result-value {
            font-size: 2rem;
            font-weight: 800;
            margin-top: 0.3rem;
        }

        .tiny {
            font-size: 0.83rem;
            opacity: 0.74;
        }

        div[data-testid="stMetric"] {
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 15px;
            padding: 0.8rem 1rem;
            background: rgba(255,255,255,0.75);
        }

        div.stButton > button {
            width: 100%;
            border-radius: 12px;
            font-weight: 700;
            min-height: 2.8rem;
        }

        div[data-testid="stTextArea"] textarea {
            border-radius: 14px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Artifact discovery / loading
# ---------------------------------------------------------------------------

def _resolve_candidate(candidates: list[str]) -> Optional[Path]:
    """Return the first existing artifact from an ordered candidate list."""
    for candidate in candidates:
        if not candidate:
            continue

        path = Path(candidate)
        if not path.is_absolute():
            path = BASE_DIR / path

        if path.is_file():
            return path

    return None


def _looks_like_model(obj: Any) -> bool:
    return callable(getattr(obj, "predict", None))


def _looks_like_vectorizer(obj: Any) -> bool:
    return callable(getattr(obj, "transform", None)) and not _looks_like_model(obj)


def _discover_pickle_artifacts() -> Tuple[Optional[Path], Optional[Path]]:
    """
    Discover a model and vectorizer when standard filenames are not available.

    This makes the app more robust when files are uploaded with generated names.
    """
    model_path = _resolve_candidate(MODEL_CANDIDATES)
    vectorizer_path = _resolve_candidate(VECTORIZER_CANDIDATES)

    if model_path and vectorizer_path:
        return model_path, vectorizer_path

    for path in sorted(BASE_DIR.glob("*.pkl")):
        try:
            obj = joblib.load(path)
        except Exception:
            continue

        if model_path is None and _looks_like_model(obj):
            model_path = path
            continue

        if vectorizer_path is None and _looks_like_vectorizer(obj):
            vectorizer_path = path

    return model_path, vectorizer_path


@st.cache_resource(show_spinner=False)
def load_artifact(path: str) -> Any:
    """Load a trusted joblib/pickle artifact once per Streamlit session."""
    LOGGER.info("Loading ML artifact: %s", path)
    return joblib.load(path)


def load_model_and_vectorizer() -> Tuple[Any, Any, Optional[Path], Optional[Path], list[str]]:
    """Load available ML artifacts and return any validation warnings."""
    warnings: list[str] = []
    model_path, vectorizer_path = _discover_pickle_artifacts()

    model = None
    vectorizer = None

    if model_path:
        try:
            model = load_artifact(str(model_path))
        except Exception as exc:
            LOGGER.exception("Model loading failed")
            warnings.append(f"Model could not be loaded: {exc}")

    if vectorizer_path:
        try:
            candidate = load_artifact(str(vectorizer_path))
            if _looks_like_vectorizer(candidate):
                vectorizer = candidate
            else:
                warnings.append(
                    f"'{vectorizer_path.name}' is not a compatible vectorizer/preprocessor."
                )
        except Exception as exc:
            LOGGER.exception("Vectorizer loading failed")
            warnings.append(f"Vectorizer could not be loaded: {exc}")

    if model is not None and not _looks_like_model(model):
        warnings.append("The discovered model artifact does not expose a predict() method.")
        model = None

    return model, vectorizer, model_path, vectorizer_path, warnings


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def get_class_labels(model: Any) -> list[str]:
    classes = getattr(model, "classes_", [])
    return [str(item) for item in classes]


def get_expected_feature_count(model: Any) -> Optional[int]:
    value = getattr(model, "n_features_in_", None)
    return int(value) if value is not None else None


def predict_with_probabilities(model: Any, features: Any) -> Tuple[str, Optional[np.ndarray]]:
    prediction = model.predict(features)
    label = str(prediction[0])

    probabilities = None
    if callable(getattr(model, "predict_proba", None)):
        probabilities = np.asarray(model.predict_proba(features)[0], dtype=float)

    return label, probabilities


def probability_frame(model: Any, probabilities: Optional[np.ndarray]) -> Optional[pd.DataFrame]:
    if probabilities is None:
        return None

    labels = get_class_labels(model)
    if len(labels) != len(probabilities):
        labels = [f"Class {i + 1}" for i in range(len(probabilities))]

    frame = pd.DataFrame(
        {"Sentiment": labels, "Probability": probabilities}
    ).sort_values("Probability", ascending=False)

    frame["Confidence"] = frame["Probability"].map(lambda x: f"{x:.1%}")
    return frame


def render_result(model: Any, label: str, probabilities: Optional[np.ndarray]) -> None:
    """Render a polished prediction result."""
    confidence = None
    if probabilities is not None and len(probabilities):
        confidence = float(np.max(probabilities))

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">Predicted Sentiment</div>
            <div class="result-value">{label}</div>
            <div class="tiny">
                {"Confidence: " + format(confidence, ".1%") if confidence is not None else "Prediction completed successfully"}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    frame = probability_frame(model, probabilities)
    if frame is not None:
        st.markdown("#### Confidence breakdown")
        st.bar_chart(
            frame.set_index("Sentiment")[["Probability"]],
            horizontal=True,
        )
        st.dataframe(
            frame[["Sentiment", "Confidence"]],
            hide_index=True,
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# Load artifacts
# ---------------------------------------------------------------------------

model, vectorizer, model_path, vectorizer_path, artifact_warnings = (
    load_model_and_vectorizer()
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown(
    f"""
    <div class="hero">
        <div class="hero-badge">PRODUCTION-READY ML INTERFACE</div>
        <h1>{APP_ICON} {APP_NAME}</h1>
        <p>
            A clean decision-support interface for sentiment classification,
            model confidence analysis, and deployment-safe artifact validation.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(f"## {APP_ICON} {APP_NAME}")
    st.caption(f"Application version {APP_VERSION}")
    st.divider()

    if model is not None:
        st.success("Model ready")
        st.metric("Expected features", get_expected_feature_count(model) or "Unknown")
        st.caption("Classes")
        st.write(" • ".join(get_class_labels(model)) or "Not available")
    else:
        st.error("Model unavailable")

    if vectorizer is not None:
        st.success("Text preprocessor ready")
    else:
        st.warning("Text preprocessor not found")

    st.divider()
    st.caption(
        "Security: only load .pkl/.joblib files that you trust. "
        "Python pickle artifacts can execute code during deserialization."
    )


# ---------------------------------------------------------------------------
# Artifact validation messages
# ---------------------------------------------------------------------------

for warning in artifact_warnings:
    st.warning(warning)

if model is None:
    st.error(
        "No usable model was found. Place your trained `.pkl` model in the same "
        "folder as `app.py`, preferably named `sentiment_model.pkl`."
    )
    st.stop()

expected_features = get_expected_feature_count(model)

status_col1, status_col2, status_col3 = st.columns(3)
with status_col1:
    st.metric("Model", type(model).__name__)
with status_col2:
    st.metric("Classes", len(get_class_labels(model)))
with status_col3:
    st.metric("Text prediction", "Enabled" if vectorizer is not None else "Needs vectorizer")

st.write("")

# ---------------------------------------------------------------------------
# Prediction experience
# ---------------------------------------------------------------------------

text_tab, feature_tab, about_tab = st.tabs(
    ["💬 Text Analysis", "📊 Feature Prediction", "ℹ️ Model Information"]
)

with text_tab:
    st.subheader("Analyze text sentiment")

    if vectorizer is None:
        st.info(
            "The classifier is loaded correctly, but the fitted text vectorizer "
            "used during training is not included. Raw text cannot be converted "
            "into the model's 59 expected features reliably without that artifact."
        )

        st.markdown(
            """
            **To enable this page:** add the original fitted vectorizer next to
            `app.py` and name it `vectorizer.pkl`. Common examples are a fitted
            `CountVectorizer` or `TfidfVectorizer`.
            """
        )

        st.code(
            "joblib.dump(vectorizer, 'vectorizer.pkl')",
            language="python",
        )
    else:
        with st.form("text_prediction_form", clear_on_submit=False):
            text = st.text_area(
                "Text to analyze",
                height=170,
                placeholder="Example: The customer service experience was excellent...",
                help="Enter a review, comment, feedback message, or other text.",
            )

            analyze = st.form_submit_button(
                "Analyze sentiment",
                type="primary",
                use_container_width=True,
            )

        if analyze:
            clean_text = text.strip()

            if not clean_text:
                st.warning("Enter some text before running the analysis.")
            else:
                try:
                    transformed = vectorizer.transform([clean_text])

                    if (
                        expected_features is not None
                        and transformed.shape[1] != expected_features
                    ):
                        st.error(
                            "The loaded vectorizer is not compatible with this model. "
                            f"The model expects {expected_features} features, but the "
                            f"vectorizer produced {transformed.shape[1]}."
                        )
                    else:
                        label, probabilities = predict_with_probabilities(
                            model, transformed
                        )
                        render_result(model, label, probabilities)
                except Exception as exc:
                    LOGGER.exception("Text prediction failed")
                    st.error(f"Prediction failed: {exc}")


with feature_tab:
    st.subheader("Predict from preprocessed features")
    st.caption(
        "Use this mode when you already have the numeric feature vector produced "
        "by the original preprocessing pipeline."
    )

    feature_count = expected_features or EXPECTED_FEATURES

    input_method = st.radio(
        "Input method",
        ["Paste feature values", "Upload CSV"],
        horizontal=True,
    )

    if input_method == "Paste feature values":
        default_values = ", ".join(["0"] * feature_count)

        with st.form("manual_feature_form"):
            raw_values = st.text_area(
                f"Enter exactly {feature_count} non-negative values",
                value=default_values,
                height=180,
                help="Separate values with commas, spaces, tabs, or new lines.",
            )

            predict_manual = st.form_submit_button(
                "Run prediction",
                type="primary",
                use_container_width=True,
            )

        if predict_manual:
            try:
                normalized = raw_values.replace(",", " ")
                values = np.array(
                    [float(value) for value in normalized.split()],
                    dtype=float,
                )

                if len(values) != feature_count:
                    st.error(
                        f"Expected exactly {feature_count} values; received {len(values)}."
                    )
                elif np.any(~np.isfinite(values)):
                    st.error("All feature values must be finite numbers.")
                elif np.any(values < 0):
                    st.error(
                        "Multinomial Naive Bayes requires non-negative feature values."
                    )
                else:
                    features = values.reshape(1, -1)
                    label, probabilities = predict_with_probabilities(model, features)
                    render_result(model, label, probabilities)

            except ValueError:
                st.error("Use numeric values only.")
            except Exception as exc:
                LOGGER.exception("Manual prediction failed")
                st.error(f"Prediction failed: {exc}")

    else:
        uploaded = st.file_uploader(
            "Upload a CSV containing the preprocessed feature columns",
            type=["csv"],
        )

        if uploaded is not None:
            try:
                data = pd.read_csv(uploaded)
                st.dataframe(data.head(10), use_container_width=True)

                numeric = data.select_dtypes(include=[np.number])

                if numeric.shape[1] != feature_count:
                    st.error(
                        f"The model expects {feature_count} numeric feature columns, "
                        f"but the uploaded file contains {numeric.shape[1]}."
                    )
                elif numeric.empty:
                    st.error("No valid numeric feature data was found.")
                elif numeric.isnull().any().any():
                    st.error("The CSV contains missing numeric values.")
                elif (numeric < 0).any().any():
                    st.error(
                        "Multinomial Naive Bayes requires non-negative feature values."
                    )
                else:
                    if st.button(
                        "Predict all rows",
                        type="primary",
                        use_container_width=True,
                    ):
                        predictions = model.predict(numeric.to_numpy())
                        results = data.copy()
                        results["Predicted_Sentiment"] = predictions

                        if callable(getattr(model, "predict_proba", None)):
                            probabilities = model.predict_proba(numeric.to_numpy())
                            results["Prediction_Confidence"] = np.max(
                                probabilities, axis=1
                            )

                        st.success(f"Predicted {len(results):,} row(s).")
                        st.dataframe(results, use_container_width=True)

                        csv_bytes = results.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "Download prediction results",
                            data=csv_bytes,
                            file_name="sentiment_predictions.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )

            except Exception as exc:
                LOGGER.exception("CSV processing failed")
                st.error(f"Could not process the CSV file: {exc}")


with about_tab:
    st.subheader("Model information")

    info = {
        "Classifier": type(model).__name__,
        "Classes": ", ".join(get_class_labels(model)) or "Unavailable",
        "Expected input features": expected_features or "Unavailable",
        "Model artifact": model_path.name if model_path else "Auto-discovered",
        "Vectorizer artifact": (
            vectorizer_path.name if vectorizer_path and vectorizer is not None
            else "Not available"
        ),
    }

    for key, value in info.items():
        st.markdown(f"**{key}:** {value}")

    st.divider()

    st.markdown(
        """
        ### Deployment notes

        This application intentionally validates model/preprocessor compatibility
        before prediction. A text classifier and its fitted vectorizer are a
        single inference pipeline in practice: using a different vocabulary or
        feature order can silently produce incorrect results.

        For a stronger production architecture, retrain and export a single
        scikit-learn `Pipeline` containing both the vectorizer and classifier.
        The app can then load one artifact and call `pipeline.predict([text])`.
        """
    )

st.divider()
st.caption(
    f"{APP_NAME} v{APP_VERSION} · Built with Streamlit and scikit-learn · "
    "Predictions should be validated before use in consequential decisions."
)
