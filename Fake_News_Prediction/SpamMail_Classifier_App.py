import streamlit as st
import pickle
import re
import contractions
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download("punkt")
nltk.download("stopwords")

def clean_text(text):
    if not isinstance(text, str):  # Handle NaN and non-string inputs
        return ""
    
    text = text.lower()  # Convert to lowercase
    text = contractions.fix(text)  # Expand contractions
    text = re.sub(r"[^a-zA-Z\s]", " ", text)  # Remove special characters & numbers
    
    tokens = word_tokenize(text)  # Tokenize
    stop_words = set(stopwords.words("english"))  # Load stop words
    
    # Remove stopwords efficiently
    tokens = [word for word in tokens if word not in stop_words]
    
    return " ".join(tokens)  # Convert tokens back to string

# Load the saved model and vectorizer
with open("C:/Users/akash/OneDrive/Documents/Own_Projects/SpamMail_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("C:/Users/akash/OneDrive/Documents/Own_Projects/vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Streamlit UI
st.title("📧 Spam Mail Detector")
st.write("Enter an email below to check if it is Spam or Ham.")

# User input
email_text = st.text_area("Enter the mail:")

if st.button("Predict"):
    if email_text:
        cleaned_text = clean_text(email_text)  # Clean the input text
        input_data = vectorizer.transform([cleaned_text])  # Transform input
        prediction = model.predict(input_data)  # Predict using the model
        
        # Display result
        if prediction[0] == 1:
            st.error("🚨 This is a Spam mail!")
        else:
            st.success("✅ This is a Ham mail!")
    else:
        st.warning("Please enter some text before predicting.")
