# Sentiment Analysis Streamlit App

An interactive web application built with Streamlit that classifies input text into sentiment categories (**Negative**, **Neutral**, or **Positive**) using a pre-trained Naive Bayes classifier and TF-IDF vectorizer.

---

## 📁 Project Structure

```text
.
├── app.py                  # Main Streamlit application script
├── naive_bayes_model.pkl   # Serialized MultinomialNB model
├── tfidf_vectorizer.pkl    # Serialized TfidfVectorizer model
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation