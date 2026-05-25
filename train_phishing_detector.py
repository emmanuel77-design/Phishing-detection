import pandas as pd
import re
import os
import joblib
import nltk
import numpy as np
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from scipy.sparse import hstack

# --- Ensure stopwords are available ---
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

# --- Text cleaning ---
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', ' ', str(text))  # keep only letters
    text = text.lower()
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# --- Paths ---
DATA_PATH = "C:/Users/issa/Downloads/archive/data-raw/combined_clean.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "phishing_model_with_meta.joblib")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
ENCODER_PATH = os.path.join(MODEL_DIR, "metadata_encoder.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "metadata_scaler.joblib")

os.makedirs(MODEL_DIR, exist_ok=True)

# --- Load dataset ---
print(f"📂 Loading dataset from {DATA_PATH} ...")
df = pd.read_csv(DATA_PATH)
print(f"✅ Loaded {len(df)} samples.")

# --- Preprocess text ---
print("🧹 Cleaning email text ...")
df["Clean_Text"] = df["text"].apply(preprocess_text)

# --- Extract metadata features ---
print("🧩 Extracting metadata features ...")

# Handle URL-related column
url_col = next((c for c in df.columns if "url" in c.lower()), None)
if url_col:
    df["num_urls"] = df[url_col].fillna(0)
else:
    df["num_urls"] = 0

# Handle sender domain
sender_col = next((c for c in df.columns if "sender" in c.lower()), None)
if sender_col:
    df["sender_domain"] = df[sender_col].fillna("").apply(
        lambda s: s.split("@")[-1].lower() if "@" in s else "unknown"
    )
else:
    df["sender_domain"] = "unknown"

# Handle receiver domain
receiver_col = next((c for c in df.columns if "receiver" in c.lower()), None)
if receiver_col:
    df["receiver_domain"] = df[receiver_col].fillna("").apply(
        lambda s: s.split("@")[-1].lower() if "@" in s else "unknown"
    )
else:
    df["receiver_domain"] = "unknown"

# --- Encode categorical metadata (sender/receiver domains) ---
enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
sender_encoded = enc.fit_transform(df[["sender_domain"]])
print(f"Encoded {len(enc.categories_[0])} sender domains.")

# --- Scale numeric metadata ---
scaler = StandardScaler(with_mean=False)
urls_scaled = scaler.fit_transform(df[["num_urls"]])

# --- TF-IDF text vectorization ---
print("🔤 Vectorizing text with TF-IDF ...")
vectorizer = TfidfVectorizer(max_features=7000)
text_tfidf = vectorizer.fit_transform(df["Clean_Text"])

# --- Combine all features ---
print("🔗 Combining text + metadata features ...")
X = hstack([text_tfidf, sender_encoded, urls_scaled])
y = df["label"].astype(int)

# --- Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Train model ---
print("🤖 Training Logistic Regression model with metadata ...")
model = LogisticRegression(random_state=42, solver="liblinear", max_iter=1000)
model.fit(X_train, y_train)

# --- Evaluate ---
print("📊 Evaluating model ...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=["Safe", "Phishing"])

print(f"\n✅ Accuracy: {accuracy:.4f}")
print(report)

# --- Save model + transformers ---
print("💾 Saving model and feature encoders ...")
joblib.dump(model, MODEL_PATH)
joblib.dump(vectorizer, VECTORIZER_PATH)
joblib.dump(enc, ENCODER_PATH)
joblib.dump(scaler, SCALER_PATH)

print(f"\n🚀 Training complete! Files saved in '{MODEL_DIR}' folder.")
