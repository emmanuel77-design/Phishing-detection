import streamlit as st
import joblib
import pandas as pd
import re
import scipy.sparse as sp

# =========================
# Utility functions
# =========================
def extract_domain(email):
    match = re.search(r"@([\w\.-]+)", email)
    return match.group(1).lower() if match else "unknown"

def count_urls(text):
    return len(re.findall(r"http[s]?://", text))

# =========================
# Load model + encoders + vectorizer
# =========================
@st.cache_resource
def load_assets():
    model = joblib.load("models/phishing_model_with_meta.joblib")
    vectorizer = joblib.load("models/tfidf_vectorizer.joblib")
    encoder = joblib.load("models/metadata_encoder.joblib")
    scaler = joblib.load("models/metadata_scaler.joblib")
    return model, vectorizer, encoder, scaler

model, vectorizer, encoder, scaler = load_assets()

# =========================
# Streamlit UI
# =========================
st.title("🛡️ Phishing Email Detector (with Metadata)")
st.write("Detect phishing or spam emails using machine learning and metadata features.")

with st.form("email_form"):
    sender = st.text_input("Sender Email", placeholder="e.g. support@paypal.com")
    subject = st.text_input("Subject")
    body = st.text_area("Email Body", height=200)
    submitted = st.form_submit_button("Analyze Email")

if submitted:
    if not body.strip():
        st.warning("Please enter an email body.")
    else:
        # Combine text and metadata
        email_text = (subject or "") + " " + body
        num_urls = count_urls(email_text)
        sender_domain = extract_domain(sender) if sender else "unknown"

        # Create DataFrame
        df_input = pd.DataFrame({
            "text_clean": [email_text],
            "sender_domain": [sender_domain],
            "num_urls": [num_urls]
        })

        # ====== TEXT VECTOR ======
        X_text = vectorizer.transform(df_input["text_clean"])

        # ====== METADATA ======
        X_domain = encoder.transform(df_input[["sender_domain"]])
        X_meta = scaler.transform(df_input[["num_urls"]])
        X_meta_full = sp.hstack([X_domain, X_meta])

        # ====== COMBINE ======
        X_final = sp.hstack([X_text, X_meta_full])

        # ====== PREDICT ======
        prediction = model.predict(X_final)[0]
        label = "🚨 Phishing / Suspicious" if prediction == 1 else "✅ Safe Email"

        st.subheader("Prediction Result:")
        st.write(label)

        # Confidence (if available)
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X_final)[0][prediction]
            st.progress(prob)
            st.caption(f"Confidence: {prob*100:.2f}%")

st.markdown("---")
st.caption("Trained using combined datasets (Enron, SpamAssassin, CEAS08, Nazario, and more).")
st.caption("Built with ❤️ using Python, Scikit-learn, and Streamlit")
st.caption("By Tolu Oderinde")