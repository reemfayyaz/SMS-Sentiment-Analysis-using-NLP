# 💬 Sentiment Intelligence

A professional Streamlit application for deploying a scikit-learn sentiment classification model.

The supplied classifier is a **Multinomial Naive Bayes (`MultinomialNB`)** model trained to predict:

- **Negative**
- **Neutral**
- **Positive**

The model expects **59 preprocessed numeric input features**.

## Important model note

The two `.pkl` files supplied with this project are identical copies of the same trained `MultinomialNB` classifier. They are **not** a model and a text vectorizer.

A text classification model trained from `CountVectorizer`, `TfidfVectorizer`, or similar preprocessing cannot safely accept raw text unless the **same fitted vectorizer used during training** is available.

For that reason, this application does not fabricate or guess the missing preprocessing step.

The app supports:

1. **Raw-text prediction** when the original fitted vectorizer is supplied.
2. **59-feature numeric prediction** when only the classifier is available.
3. **Batch CSV prediction** for already-preprocessed data.
4. Probability/confidence visualization when `predict_proba()` is supported.
5. Automatic model artifact discovery and validation.

---

## Project structure

```text
sentiment-project/
│
├── app.py
├── requirements.txt
├── README.md
├── sentiment_model.pkl
└── vectorizer.pkl          # required only for raw-text prediction
```

Rename one of your supplied model files to:

```text
sentiment_model.pkl
```

You only need **one** of the two uploaded model files because they are identical.

---

## Run locally

### 1. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the Streamlit application

```bash
streamlit run app.py
```

Streamlit will display the local application address in your terminal.

---

## Enable raw-text sentiment prediction

If your training code used a vectorizer such as:

```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(text_data)
```

save the **already fitted** vectorizer after training:

```python
import joblib

joblib.dump(vectorizer, "vectorizer.pkl")
```

Place `vectorizer.pkl` beside `app.py`.

The vectorizer must generate exactly **59 features** to match the supplied model.

Do **not** create a new vectorizer and fit it on different text. The vocabulary and feature ordering must be identical to the training pipeline.

---

## Recommended training architecture

For production deployments, save preprocessing and classification together as one scikit-learn `Pipeline`.

Example:

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

pipeline = Pipeline(
    [
        ("vectorizer", TfidfVectorizer()),
        ("classifier", MultinomialNB()),
    ]
)

pipeline.fit(X_train, y_train)

joblib.dump(pipeline, "sentiment_pipeline.pkl")
```

A single pipeline prevents mismatches between the model and text preprocessing.

---

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload:
   - `app.py`
   - `requirements.txt`
   - `README.md`
   - `sentiment_model.pkl`
   - `vectorizer.pkl` if available
3. Push the files to GitHub.
4. Open Streamlit Community Cloud.
5. Create a new app from your GitHub repository.
6. Select `app.py` as the entry point.
7. Deploy.

---

## Environment variables

You can override artifact paths without modifying the code.

```bash
MODEL_PATH=/path/to/model.pkl
VECTORIZER_PATH=/path/to/vectorizer.pkl
LOG_LEVEL=INFO
```

The app also automatically scans `.pkl` files in its project directory when standard filenames are not found.

---

## Features

- Modern responsive Streamlit interface
- Cached ML artifact loading
- Defensive artifact discovery
- Input validation
- Model/vectorizer compatibility checks
- Text sentiment classification
- Manual feature-vector prediction
- Batch CSV inference
- Confidence probabilities
- Downloadable prediction results
- Logging and exception handling
- Clear production/deployment guidance

---

## Model compatibility

The uploaded model was serialized with **scikit-learn 1.6.1**.

`requirements.txt` pins:

```text
scikit-learn==1.6.1
```

Using the same scikit-learn version is recommended because pickle/joblib models are not guaranteed to be compatible across library versions.

---

## Security

Never load a `.pkl` or `.joblib` artifact from an untrusted source.

Python pickle-based files can execute code when loaded. Only deploy artifacts that you created yourself or received from a trusted source.

---

## Limitations

- The current classifier cannot perform reliable raw-text inference without its original fitted vectorizer/preprocessor.
- Model performance depends on the quality and representativeness of its training data.
- Prediction confidence is not the same as certainty.
- Validate the model before using predictions in consequential workflows.

---

## Suggested GitHub repository files

```text
app.py
README.md
requirements.txt
sentiment_model.pkl
vectorizer.pkl
.gitignore
```

Example `.gitignore`:

```text
.venv/
__pycache__/
*.pyc
.DS_Store
.streamlit/secrets.toml
```

---

## License

Add the license that is appropriate for your project before public distribution.
