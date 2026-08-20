import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# Configuration
# ============================================================
APP_TITLE = "Sentiment Analysis"
APP_ICON = "💬"
MODEL_PATH = "sentiment_model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ============================================================
# Professional styling
# ============================================================
st.markdown(
    """
    <style>
        .block-container {
            max-width: 900px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .hero {
            padding: 2.2rem 2rem;
            border-radius: 24px;
            background: linear-gradient(135deg, #111827 0%, #312e81 55%, #6d28d9 100%);
            color: white;
            margin-bottom: 1.5rem;
            box-shadow: 0 14px 35px rgba(49, 46, 129, 0.22);
        }

        .hero h1 {
            margin: 0;
            font-size: 2.35rem;
            font-weight: 800;
        }

        .hero p {
            margin: .65rem 0 0 0;
            opacity: .88;
            font-size: 1.05rem;
        }

        .result-card {
            padding: 1.4rem 1.5rem;
            border-radius: 18px;
            border: 1px solid rgba(128,128,128,.22);
            margin-top: 1rem;
        }

        div.stButton > button {
            width: 100%;
            border-radius: 12px;
            height: 3rem;
            font-weight: 700;
        }

        [data-testid="stMetric"] {
            border: 1px solid rgba(128,128,128,.20);
            border-radius: 16px;
            padding: 1rem;
        }

        .footer {
            text-align: center;
            opacity: .6;
            margin-top: 2.5rem;
            font-size: .85rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Model utilities
# ============================================================
@st.cache_resource
def load_pickle(path):
    return joblib.load(path)


def confidence_table(model, X):
    """Return class probabilities when supported by the model."""
    if not hasattr(model, "predict_proba"):
        return None

    probabilities = model.predict_proba(X)[0]
    classes = getattr(model, "classes_", [f"Class {i}" for i in range(len(probabilities))])

    return (
        pd.DataFrame(
            {"Sentiment": classes, "Probability": probabilities}
        )
        .sort_values("Probability", ascending=False)
        .reset_index(drop=True)
    )


def sentiment_icon(label):
    label = str(label).lower()
    if "positive" in label:
        return "😊"
    if "negative" in label:
        return "😞"
    if "neutral" in label:
        return "😐"
    return "💬"


# ============================================================
# Header
# ============================================================
st.markdown(
    """
    <div class="hero">
        <h1>💬 Sentiment Analysis</h1>
        <p>Classify text as Positive, Neutral, or Negative using a trained machine-learning model.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Load model
# ============================================================
if not os.path.exists(MODEL_PATH):
    st.error(
        f"Model file not found: `{MODEL_PATH}`\n\n"
        "Place the trained model in the same folder as `app.py` "
        "and name it `sentiment_model.pkl`."
    )
    st.stop()

try:
    model = load_pickle(MODEL_PATH)
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.stop()

expected_features = getattr(model, "n_features_in_", None)

# ============================================================
# Main application
# ============================================================
tab1, tab2 = st.tabs(["Text Prediction", "Model Information"])

with tab1:
    st.subheader("Analyze Text")

    if os.path.exists(VECTORIZER_PATH):
        try:
            vectorizer = load_pickle(VECTORIZER_PATH)

            user_text = st.text_area(
                "Enter text",
                height=160,
                placeholder="Example: I really enjoyed this product and would recommend it.",
            )

            predict_clicked = st.button("Analyze Sentiment", type="primary")

            if predict_clicked:
                if not user_text.strip():
                    st.warning("Please enter some text before running the prediction.")
                else:
                    try:
                        X = vectorizer.transform([user_text])

                        if expected_features is not None and X.shape[1] != expected_features:
                            st.error(
                                f"Feature mismatch: the model expects {expected_features} features "
                                f"but the vectorizer produced {X.shape[1]}. "
                                "Please use the vectorizer that was fitted with this model."
                            )
                        else:
                            prediction = model.predict(X)[0]
                            icon = sentiment_icon(prediction)

                            st.markdown(
                                f"""
                                <div class="result-card">
                                    <h3>{icon} Prediction: {prediction}</h3>
                                    <p>The model classified the submitted text as <b>{prediction}</b>.</p>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                            probabilities = confidence_table(model, X)
                            if probabilities is not None:
                                top_probability = float(probabilities.iloc[0]["Probability"])
                                st.metric("Model Confidence", f"{top_probability:.1%}")

                                chart_data = probabilities.set_index("Sentiment")[["Probability"]]
                                st.bar_chart(chart_data)

                                with st.expander("View probability details"):
                                    display_df = probabilities.copy()
                                    display_df["Probability"] = display_df["Probability"].map(
                                        lambda value: f"{value:.2%}"
                                    )
                                    st.dataframe(display_df, use_container_width=True, hide_index=True)

                    except Exception as exc:
                        st.error(f"Prediction failed: {exc}")

        except Exception as exc:
            st.error(f"Could not load `{VECTORIZER_PATH}`: {exc}")

    else:
        st.warning(
            "The uploaded model expects transformed numeric features, but a fitted text "
            "`vectorizer.pkl` was not included. A text model cannot safely convert raw sentences "
            "to the model's 59 expected features without the original fitted vectorizer."
        )

        st.info(
            "Add the fitted vectorizer used during model training to this project and name it "
            "`vectorizer.pkl`. The text prediction interface will then activate automatically."
        )

        if expected_features:
            with st.expander("Advanced: test pre-transformed feature values"):
                st.caption(
                    f"For testing only. Enter exactly {expected_features} numeric feature values "
                    "in the same order used during training."
                )

                feature_text = st.text_area(
                    "Numeric features",
                    placeholder="0, 1.25, 0, 3.8, ...",
                    height=120,
                )

                if st.button("Predict From Features"):
                    try:
                        values = [
                            float(value.strip())
                            for value in feature_text.replace("\n", ",").split(",")
                            if value.strip()
                        ]

                        if len(values) != expected_features:
                            st.error(
                                f"Expected {expected_features} values, but received {len(values)}."
                            )
                        else:
                            X = np.asarray(values, dtype=float).reshape(1, -1)
                            prediction = model.predict(X)[0]
                            st.success(
                                f"{sentiment_icon(prediction)} Prediction: **{prediction}**"
                            )

                            probabilities = confidence_table(model, X)
                            if probabilities is not None:
                                st.dataframe(
                                    probabilities,
                                    use_container_width=True,
                                    hide_index=True,
                                )
                    except ValueError:
                        st.error("Please enter numbers separated by commas.")
                    except Exception as exc:
                        st.error(f"Prediction failed: {exc}")

with tab2:
    st.subheader("Model Information")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Model Type", type(model).__name__)

    with col2:
        st.metric(
            "Expected Features",
            expected_features if expected_features is not None else "Unknown",
        )

    classes = getattr(model, "classes_", None)
    if classes is not None:
        st.write("**Prediction classes:**")
        st.write(", ".join(map(str, classes)))

    st.write("**Required project files:**")
    st.code(
        """project/
├── app.py
├── sentiment_model.pkl
├── vectorizer.pkl
├── requirements.txt
└── README.md"""
    )

    st.caption(
        "Important: pickle/joblib files should only be loaded from trusted sources."
    )

st.markdown(
    '<div class="footer">Sentiment Analysis • Machine Learning + Streamlit</div>',
    unsafe_allow_html=True,
)
