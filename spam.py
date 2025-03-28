import pandas as pd
import re
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
file_path = "spam.csv"
df = pd.read_csv(file_path, encoding="latin-1")

# Rename columns for clarity
df = df.rename(columns={"v1": "label", "v2": "message"})

# Drop unnecessary columns
df = df[["label", "message"]]

# Convert labels to binary values
df["label"] = df["label"].map({"ham": 0, "spam": 1})


def preprocess_text(text):    # Convert text to numerical features using TF-IDF
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

df["clean_message"] = df["message"].apply(preprocess_text)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df["clean_message"], df["label"], test_size=0.2, random_state=42)


# Convert text to numerical features using TF-IDF
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


model = MultinomialNB()     # Train a Naive Bayes classifier
model.fit(X_train_tfidf, y_train)

# Predictions
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print results
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:\n", report)


st.title("SMS Spam Detector")     # Streamlit GUI
st.write("Enter a message to check if it's Spam or Ham")


user_input = st.text_area("Enter your message:")   # User input

if st.button("Predict"):
    if user_input:
        processed_input = preprocess_text(user_input)
        input_tfidf = vectorizer.transform([processed_input])
        prediction = model.predict(input_tfidf)[0]
        result = "Spam" if prediction == 1 else "Ham"
        st.write(f"Prediction: **{result}**")
    else:
        st.write("Please enter a message.")






