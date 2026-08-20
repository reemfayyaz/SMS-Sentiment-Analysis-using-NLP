import joblib
import numpy as np
import streamlit as st

# Page setup
st.set_page_config(
    page_title="Sentiment Analysis App", page_icon="😊", layout="centered"
)


# Load model and vectorizer
@st.cache_resource
def load_assets():
    # Replace filenames if your saved models use different paths
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    model = joblib.load("naive_bayes_model.pkl")
    return vectorizer, model


try:
    vectorizer, model = load_assets()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.info(
        "Ensure your serialized model files (e.g., `.pkl` or `.joblib`) are in the same directory."
    )
    st.stop()

# Streamlit UI
st.title("😊 Sentiment Analysis Classifier")
st.write(
    "Enter a sentence below to analyze its sentiment (Negative, Neutral, or Positive)."
)

# Text input
user_input = st.text_area("Input Text", placeholder="Type your text here...")

if st.button("Predict Sentiment", type="primary"):
    if user_input.strip() == "":
        st.warning("Please enter some text before submitting.")
    else:
        # Transform input text using the TF-IDF vectorizer
        transformed_input = vectorizer.transform([user_input])

        # Make prediction and get probabilities
        prediction = model.predict(transformed_input)[0]
        probabilities = model.predict_proba(transformed_input)[0]
        classes = model.classes_

        # Display result
        st.subheader("Prediction Result")

        if prediction == "Positive":
            st.success(f"**Predicted Sentiment:** {prediction}")
        elif prediction == "Negative":
            st.error(f"**Predicted Sentiment:** {prediction}")
        else:
            st.info(f"**Predicted Sentiment:** {prediction}")

        # Display class probabilities
        st.write("---")
        st.write("**Confidence Breakdown:**")
        prob_dict = {
            cls: f"{prob * 100:.2f}%" for cls, prob in zip(classes, probabilities)
        }
        st.json(prob_dict)