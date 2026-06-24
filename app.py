import streamlit as st
import joblib

# Load Model and Vectorizer
model = joblib.load("models/spam_model.pkl")
vectorizer = joblib.load("vectorizer/tfidf_vectorizer.pkl")

# Page Config
st.set_page_config(
    page_title="Spam Email Detector",
    page_icon="📧",
    layout="centered"
)

st.title("📧 Spam Email Detector")
st.write("Enter an email or message below to check whether it is Spam or Not Spam.")

# Input Box
email_text = st.text_area(
    "Enter Email Text",
    height=150,
    placeholder="Type or paste your email/message here..."
)

# Prediction Button
if st.button("Check Message"):
    if email_text.strip() == "":
        st.warning("Please enter a message.")
    else:
        email_vector = vectorizer.transform([email_text])
        prediction = model.predict(email_vector)[0]

        if prediction == 1:
            st.error("🚫 This message is SPAM")
        else:
            st.success("✅ This message is NOT SPAM")