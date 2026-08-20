# 💬 Sentiment Intelligence

A professional machine-learning web application built with **Streamlit** for classifying text sentiment as **Positive**, **Neutral**, or **Negative**.

The application uses a trained **Multinomial Naive Bayes** classifier together with a **TF-IDF vectorizer** to analyze reviews, comments, feedback, and other short text.

## ✨ Features

- Modern and professional Streamlit interface
- Positive, Neutral, and Negative sentiment classification
- Prediction confidence score
- Confidence comparison chart for all sentiment classes
- Character and word counters
- Built-in sample text
- Recent prediction history
- Model status panel
- Input validation and error handling
- Cached model loading for improved performance
- Ready for GitHub and Streamlit Community Cloud

## 📁 Project Structure

```text
sentiment-analysis/
│
├── app.py
├── model (1).pkl
├── vectorizer (2).pkl
├── requirements.txt
└── README.md
```

The application also supports the simplified filenames `model.pkl` and `vectorizer.pkl`.

## 🧠 Machine Learning Model

The application uses:

- **TF-IDF Vectorizer** for converting text into numerical features
- **Multinomial Naive Bayes** for sentiment classification
- **Joblib** for loading the trained model and vectorizer

Supported sentiment classes:

- Positive
- Neutral
- Negative

## 🚀 Run the Project Locally

### 1. Download or clone the project

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd YOUR_PROJECT_FOLDER
```

You can also simply place all project files inside the same folder.

### 2. Install the dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the Streamlit application

```bash
streamlit run app.py
```

Streamlit will display a local URL in the terminal. Open it in your browser to use the application.

## ☁️ Deploy on Streamlit Community Cloud

1. Upload `app.py`, the model files, `requirements.txt`, and `README.md` to a GitHub repository.
2. Sign in to Streamlit Community Cloud.
3. Create a new application.
4. Select your GitHub repository.
5. Set the main file path to `app.py`.
6. Deploy the application.

Make sure the `.pkl` model and vectorizer files are included in the repository and use the filenames expected by `app.py`.

## 📦 Requirements

The project requires:

```text
streamlit
pandas
scikit-learn
joblib
```

The exact dependency ranges are provided in `requirements.txt`.

## 💡 Example

Enter text such as:

```text
I absolutely love this product. The quality is excellent.
```

Select **Analyze Sentiment** and the application will display the predicted sentiment together with the model confidence.

## ⚠️ Important Note

The `.pkl` files should be loaded with a compatible Python/scikit-learn environment. If a serialized model was trained with a substantially different scikit-learn version, retraining or re-exporting the model in the deployment environment may be necessary.

## 🛠️ Technologies

- Python
- Streamlit
- Pandas
- Scikit-learn
- Joblib
- TF-IDF
- Multinomial Naive Bayes

## 📄 License

This project can be used for educational, portfolio, and machine-learning demonstration purposes.
