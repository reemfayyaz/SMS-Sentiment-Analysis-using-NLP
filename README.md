# Sentiment Analysis Streamlit App

A professional Streamlit application for sentiment classification using a trained
scikit-learn machine-learning model.

The inspected model is a **Multinomial Naive Bayes** classifier with three output
classes:

- Negative
- Neutral
- Positive

The model expects **59 transformed input features**.

## Important

The supplied `.pkl` files contain the classifier, but a fitted text vectorizer was
not included. If the model was trained from text using `CountVectorizer` or
`TfidfVectorizer`, you must also save the **same fitted vectorizer** used during
training.

Name the files:

```text
sentiment_model.pkl
vectorizer.pkl
```

Do not fit a new vectorizer only for deployment. It must be the original fitted
vectorizer so that its vocabulary and feature order match the model.

## Project Structure

```text
sentiment-analysis/
├── app.py
├── sentiment_model.pkl
├── vectorizer.pkl
├── requirements.txt
└── README.md
```

## Save the Vectorizer During Training

If your training code uses a vectorizer, save it after fitting:

```python
import joblib

joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
```

## Installation

Create and activate a virtual environment if desired, then install the packages:

```bash
pip install -r requirements.txt
```

## Run the App

```bash
streamlit run app.py
```

Streamlit will open the application in your browser.

## GitHub / Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload `app.py`, `requirements.txt`, `README.md`, `sentiment_model.pkl`, and
   `vectorizer.pkl`.
3. Sign in to Streamlit Community Cloud.
4. Select your GitHub repository.
5. Set the main file path to `app.py`.
6. Deploy the application.

## Features

- Professional responsive interface
- Positive, Neutral, and Negative predictions
- Prediction confidence when supported by the model
- Probability chart
- Model-information panel
- Cached model loading
- Error handling for missing/incompatible files
- Advanced numeric-feature test mode when the vectorizer is unavailable

## Security

Only load `.pkl` or Joblib files that you trust. Python pickle-based model files
can execute code while loading.

## Troubleshooting

### `sentiment_model.pkl` not found

Rename your classifier file to:

```text
sentiment_model.pkl
```

and place it beside `app.py`.

### `vectorizer.pkl` not found

Export the fitted vectorizer from the original training notebook and place it beside
`app.py`.

### Feature mismatch

The classifier inspected for this project expects 59 features. If the vectorizer
produces a different number, the vectorizer and classifier were not fitted as the
same training pipeline. Use the matching original vectorizer.

## Technology

- Python
- Streamlit
- scikit-learn
- pandas
- NumPy
- Joblib
