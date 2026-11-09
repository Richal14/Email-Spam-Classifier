

import streamlit as st
import pickle

# Load trained model
with open("spam_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load vectorizer
with open("vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

# Website settings
st.set_page_config(
    page_title="Email Spam Classifier",
    page_icon="📩"
)

# Heading
st.title("📩 Email Spam Classifier")

st.write(
    "Enter your message below to check "
    "whether it is Spam or Legitimate."
)

# Message input
message = st.text_area("Enter your message:")

# Prediction button
if st.button("Check Message"):

    if message.strip() == "":
        st.warning("Please enter a message.")

    else:
        message_vector = vectorizer.transform([message])

        prediction = model.predict(message_vector)[0]

        if prediction == 1:
            st.error("🚨 This is a SPAM message!")
        else:
            st.success("✅ This is a LEGITIMATE message!")

        probabilities = model.predict_proba(message_vector)[0]

        st.write(
            "Prediction confidence:",
            f"{max(probabilities) * 100:.2f}%"
        )
