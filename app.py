import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Sentiment AI",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_VERSION = "1.0.0"
BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = os.getenv("MODEL_PATH", "sentiment_model.pkl")
EXPECTED_FEATURES = 59

# ============================================================
# PROFESSIONAL UI
# ============================================================
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 50%, #ecfeff 100%);
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        background: linear-gradient(120deg, #0f172a 0%, #312e81 52%, #0f766e 100%);
        padding: 2.6rem;
        border-radius: 26px;
        color: white;
        box-shadow: 0 20px 50px rgba(15, 23, 42, .18);
        margin-bottom: 1.5rem;
    }

    .hero-badge {
        display: inline-block;
        background: rgba(255,255,255,.13);
        border: 1px solid rgba(255,255,255,.20);
        padding: .35rem .75rem;
        border-radius: 999px;
        font-size: .75rem;
        font-weight: 700;
        letter-spacing: .08em;
        margin-bottom: .9rem;
    }

    .hero h1 {
        font-size: 3.2rem;
        line-height: 1.05;
        margin: 0;
        font-weight: 800;
    }

    .hero p {
        color: #dbeafe;
        font-size: 1.05rem;
        max-width: 760px;
        line-height: 1.65;
        margin-top: .8rem;
    }

    .result-card {
        background: white;
        border: 1px solid #e2e8f0;
        padding: 1.8rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 12px 30px rgba(15,23,42,.07);
    }

    .result-title {
        color: #64748b;
        font-size: .78rem;
        font-weight: 800;
        letter-spacing: .12em;
        text-transform: uppercase;
    }

    .result-value {
        font-size: 2.3rem;
        font-weight: 850;
        margin: .35rem 0;
        background: linear-gradient(90deg, #4f46e5, #0f766e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,.85);
        border: 1px solid #e2e8f0;
        padding: 1rem;
        border-radius: 17px;
        box-shadow: 0 8px 22px rgba(15,23,42,.05);
    }

    div.stButton > button,
    div[data-testid="stFormSubmitButton"] > button,
    div.stDownloadButton > button {
        width: 100%;
        min-height: 2.9rem;
        border-radius: 13px;
        border: none;
        font-weight: 750;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a, #1e1b4b);
    }

    section[data-testid="stSidebar"] * {
        color: #f8fafc;
    }

    div[data-testid="stAlert"] {
        border-radius: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# MODEL LOADING
# ============================================================
@st.cache_resource(show_spinner=False)
def load_model():
    path = BASE_DIR / MODEL_FILE

    if not path.exists():
        # Fall back to the first compatible pickle in the app directory.
        for candidate in BASE_DIR.glob("*.pkl"):
            try:
                obj = joblib.load(candidate)
                if hasattr(obj, "predict"):
                    return obj, candidate
            except Exception:
                continue
        raise FileNotFoundError(
            f"Model file '{MODEL_FILE}' was not found. "
            "Place sentiment_model.pkl next to app.py."
        )

    model = joblib.load(path)
    return model, path


try:
    model, model_path = load_model()
except Exception as error:
    st.error(f"Unable to load the machine-learning model: {error}")
    st.stop()

classes = [str(x) for x in getattr(model, "classes_", [])]
feature_count = int(getattr(model, "n_features_in_", EXPECTED_FEATURES))

# ============================================================
# HEADER
# ============================================================
st.markdown(
    """
    <div class="hero">
        <div class="hero-badge">MACHINE LEARNING • SENTIMENT INTELLIGENCE</div>
        <h1>✨ Sentiment AI</h1>
        <p>
            A professional machine-learning dashboard for classifying sentiment
            into Negative, Neutral, and Positive categories using a trained
            Multinomial Naive Bayes model.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.title("✨ Sentiment AI")
    st.caption(f"Version {APP_VERSION}")
    st.divider()

    st.success("Model loaded successfully")
    st.write("**Model:**", type(model).__name__)
    st.write("**Input features:**", feature_count)
    st.write("**Classes:**")
    for label in classes:
        st.write(f"• {label}")

    st.divider()
    st.caption(
        "Use only trusted pickle files. Pickle artifacts may execute Python "
        "code while loading."
    )

# ============================================================
# MODEL OVERVIEW
# ============================================================
c1, c2, c3 = st.columns(3)
c1.metric("Model", type(model).__name__)
c2.metric("Features", feature_count)
c3.metric("Sentiment Classes", len(classes))

st.write("")

# ============================================================
# MAIN APPLICATION
# ============================================================
single_tab, batch_tab, info_tab = st.tabs(
    ["🎯 Single Prediction", "📊 Batch Prediction", "ℹ️ Model Details"]
)

with single_tab:
    st.subheader("Single sentiment prediction")
    st.caption(
        "Enter the preprocessed numeric feature vector used by the trained model."
    )

    with st.form("prediction_form"):
        default_vector = ", ".join(["0"] * feature_count)

        values_text = st.text_area(
            f"Feature vector — exactly {feature_count} values",
            value=default_vector,
            height=180,
            help="Separate values using commas, spaces, tabs, or new lines.",
        )

        submitted = st.form_submit_button(
            "Analyze Sentiment",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        try:
            values = np.array(
                [float(v) for v in values_text.replace(",", " ").split()],
                dtype=float,
            )

            if len(values) != feature_count:
                st.error(
                    f"The model requires exactly {feature_count} values. "
                    f"You entered {len(values)}."
                )
            elif not np.all(np.isfinite(values)):
                st.error("All feature values must be valid finite numbers.")
            elif np.any(values < 0):
                st.error(
                    "This Multinomial Naive Bayes model requires non-negative values."
                )
            else:
                X = values.reshape(1, -1)
                prediction = str(model.predict(X)[0])

                confidence = None
                probabilities = None

                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(X)[0]
                    confidence = float(np.max(probabilities))

                confidence_text = (
                    f"{confidence:.1%} confidence"
                    if confidence is not None
                    else "Prediction complete"
                )

                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-title">Predicted Sentiment</div>
                        <div class="result-value">{prediction}</div>
                        <div>{confidence_text}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if probabilities is not None:
                    st.markdown("#### Prediction confidence")

                    probability_df = pd.DataFrame(
                        {
                            "Sentiment": classes,
                            "Probability": probabilities,
                        }
                    ).sort_values("Probability", ascending=False)

                    st.bar_chart(
                        probability_df.set_index("Sentiment"),
                        horizontal=True,
                    )

                    display_df = probability_df.copy()
                    display_df["Probability"] = display_df["Probability"].map(
                        lambda x: f"{x:.2%}"
                    )
                    st.dataframe(
                        display_df,
                        hide_index=True,
                        use_container_width=True,
                    )

        except ValueError:
            st.error("Please enter numeric feature values only.")
        except Exception as error:
            st.error(f"Prediction failed: {error}")


with batch_tab:
    st.subheader("Batch CSV prediction")
    st.caption(
        f"Upload a CSV containing exactly {feature_count} numeric model features."
    )

    uploaded_file = st.file_uploader(
        "Upload preprocessed CSV",
        type=["csv"],
        help="The CSV must contain the same features and order used during training.",
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("#### Data preview")
            st.dataframe(df.head(10), use_container_width=True)

            numeric_df = df.select_dtypes(include=np.number)

            if numeric_df.shape[1] != feature_count:
                st.error(
                    f"Expected {feature_count} numeric columns, but found "
                    f"{numeric_df.shape[1]}."
                )
            elif numeric_df.isnull().any().any():
                st.error("The feature data contains missing values.")
            elif (numeric_df < 0).any().any():
                st.error(
                    "Multinomial Naive Bayes requires non-negative feature values."
                )
            elif st.button(
                "Run Batch Prediction",
                type="primary",
                use_container_width=True,
            ):
                X = numeric_df.to_numpy()
                predictions = model.predict(X)

                results = df.copy()
                results["Predicted_Sentiment"] = predictions

                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X)
                    results["Confidence"] = np.max(probs, axis=1)

                st.success(f"Successfully analyzed {len(results):,} records.")
                st.dataframe(results, use_container_width=True)

                st.download_button(
                    "Download Results",
                    data=results.to_csv(index=False).encode("utf-8"),
                    file_name="sentiment_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        except Exception as error:
            st.error(f"Unable to process the CSV: {error}")


with info_tab:
    st.subheader("Model information")

    st.write(f"**Classifier:** {type(model).__name__}")
    st.write(f"**Model file:** {model_path.name}")
    st.write(f"**Required features:** {feature_count}")
    st.write(f"**Classes:** {', '.join(classes)}")

    st.info(
        "The uploaded artifacts are MultinomialNB classifiers expecting 59 "
        "preprocessed numeric features. Raw text prediction requires the exact "
        "fitted vectorizer or preprocessing pipeline used during training."
    )

    st.markdown(
        """
        ### Production recommendation

        For a text-facing application, save the trained vectorizer and classifier
        together in a scikit-learn `Pipeline`. This guarantees that raw text is
        transformed using the exact vocabulary and feature ordering used during
        model training.
        """
    )

st.divider()
st.caption(
    "Sentiment AI • Professional Streamlit ML Application • "
    "Validate predictions before consequential use."
)
