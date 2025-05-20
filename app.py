import streamlit as st
import joblib

# Load the trained components
model = joblib.load('decision_tree_model.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')
label_encoder = joblib.load('label_encoder.joblib')

# Streamlit UI
st.set_page_config(page_title="Sentiment Predictor", layout="centered")

st.title("🔍 Social Media Sentiment Detector")
st.write("Enter a message and a hashtag to predict the sentiment.")

# User input
text_input = st.text_input("Enter Text:")
hashtag_input = st.text_input("Enter Hashtag (optional):")

if st.button("Predict Sentiment"):
    if text_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        combined_text = text_input + ' ' + hashtag_input
        vector = vectorizer.transform([combined_text])
        prediction = model.predict(vector)
        predicted_label = label_encoder.inverse_transform(prediction)
        st.success(f"**Predicted Sentiment:** {predicted_label[0]}")
