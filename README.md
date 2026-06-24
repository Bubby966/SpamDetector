# 📧 Spam Email Detection System

## Overview

A Machine Learning-based Spam Email Detection System that classifies messages as Spam or Not Spam using Natural Language Processing (NLP) and TF-IDF Vectorization.

## Features

* Detects spam emails/messages
* TF-IDF text vectorization
* Multinomial Naive Bayes classifier
* Interactive Streamlit web interface
* Trained and saved ML model for fast predictions

## Tech Stack

* Python
* Pandas
* Scikit-learn
* Joblib
* Streamlit

## Project Structure

```
SpamDetector/
│
├── dataset/
│   └── spam.csv
│
├── models/
│   └── spam_model.pkl
│
├── vectorizer/
│   └── tfidf_vectorizer.pkl
│
├── app.py
├── predict.py
├── train_model.py
├── requirements.txt
└── README.md
```

## Model Used

* TF-IDF Vectorizer
* Multinomial Naive Bayes

## Installation

```bash
pip install -r requirements.txt
```

## Run Application

```bash
streamlit run app.py
```

## Future Enhancements

* Email attachment scanning
* Deep learning models
* Multi-language support
* Phishing detection

## Author

T.Rakesh
B.Tech CSE (AI & ML)
JNTUH University College of Engineering Manthani
