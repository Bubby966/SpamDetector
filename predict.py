import joblib

model = joblib.load("models/spam_model.pkl")
vectorizer = joblib.load("vectorizer/tfidf_vectorizer.pkl")

email = input("Enter Email Text: ")

email_vector = vectorizer.transform([email])

prediction = model.predict(email_vector)[0]

if prediction == 1:
    print("\n🚫 Spam Email")
else:
    print("\n✅ Not Spam Email")