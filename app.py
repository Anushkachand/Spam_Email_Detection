import streamlit as st
import pickle

st.set_page_config(
    page_title="Spam Email Detection",
    page_icon="📧",
    layout="centered"
)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("📧 Spam Email Detection")
st.write("Detect whether an email is **Spam** or **Ham (Not Spam)** using Machine Learning.")

st.markdown("---")

email = st.text_area(
    "✉️ Enter Email Message",
    height=200,
    placeholder="Type or paste an email here..."
)

if st.button("🔍 Predict"):

    if email.strip() == "":
        st.warning("Please enter an email message.")
    else:

        email_vector = vectorizer.transform([email])

        prediction = model.predict(email_vector)[0]

        probability = model.predict_proba(email_vector)

        confidence = probability.max() * 100

        st.markdown("---")

        if prediction == "spam":

            st.error("🚨 This Email is SPAM")

        else:

            st.success("✅ This Email is NOT SPAM")

        st.write(f"### Confidence : **{confidence:.2f}%**")

st.sidebar.title("About Project")

st.sidebar.info(
"""
### Spam Email Detection

This project predicts whether an email is:

✅ Ham (Not Spam)

🚨 Spam

Machine Learning Algorithm:
- Multinomial Naive Bayes

Feature Extraction:
- TF-IDF

Developed using:
- Python
- Scikit-learn
- Streamlit
"""
)

st.sidebar.markdown("---")
st.sidebar.write("Made for B.Tech ML Project")