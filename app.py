import streamlit as st
import pickle
import os

st.set_page_config(
    page_title="Spam Email Detection",
    page_icon="📧",
    layout="centered"
)

# Load model and vectorizer
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    with open(os.path.join(BASE_DIR, "model.pkl"), "rb") as f:
        model = pickle.load(f)

    with open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)

except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

st.title("📧 Spam Email Detection")
st.write("Detect whether an email is **Spam** or **Ham (Not Spam)** using Machine Learning.")

st.markdown("---")

email = st.text_area(
    "✉️ Enter Email Message",
    height=200,
    placeholder="Type or paste an email here..."
)

if st.button("🔍 Predict"):

    if not email.strip():
        st.warning("Please enter an email message.")
    else:
        email_vector = vectorizer.transform([email])

        prediction = model.predict(email_vector)[0]

        confidence = None
        if hasattr(model, "predict_proba"):
            confidence = model.predict_proba(email_vector).max() * 100

        st.markdown("---")

        if str(prediction).lower() == "spam":
            st.error("🚨 This Email is SPAM")
        else:
            st.success("✅ This Email is NOT SPAM")

        if confidence is not None:
            st.write(f"### Confidence: **{confidence:.2f}%**")
        else:
            st.info("Confidence score is not available for this model.")

st.sidebar.title("About Project")

st.sidebar.info("""
### Spam Email Detection

This project predicts whether an email is:

✅ **Ham (Not Spam)**

🚨 **Spam**

**Machine Learning Algorithm**
- Multinomial Naive Bayes

**Feature Extraction**
- TF-IDF

**Developed using**
- Python
- Scikit-learn
- Streamlit
""")

st.sidebar.markdown("---")
st.sidebar.write("Made for B.Tech ML Project")
