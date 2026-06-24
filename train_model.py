import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load Dataset
df = pd.read_csv("dataset/spam.csv")

# Features and Labels
X = df["text"]
y = df["target"]

# TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')

X = vectorizer.fit_transform(X)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = MultinomialNB()

# Train
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

print("\nClassification Report\n")
print(classification_report(y_test, y_pred))

# Save Model
joblib.dump(model, "models/spam_model.pkl")

# Save Vectorizer
joblib.dump(vectorizer, "vectorizer/tfidf_vectorizer.pkl")

print("\nModel Saved Successfully!")