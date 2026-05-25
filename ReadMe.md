# 🛡️ Phishing Email Detection System

A machine learning–powered system that detects **phishing or spam emails** using both **email text** and **metadata features** such as sender domain and URL count.  
This project combines multiple real-world email datasets (e.g., Enron, SpamAssassin, CEAS08, Nazario, etc.) to build a more reliable phishing classifier.

---

## 🚀 Features

- **Text-based detection:** Uses TF-IDF vectorization to capture important terms and patterns.  
- **Metadata features:** Considers email attributes such as sender domain and number of URLs.  
- **Integrated model:** Logistic Regression trained on over 150,000 samples.  
- **Interactive interface:** Built with **Streamlit** for quick testing and demonstrations.  
- **Persistent artifacts:** Includes saved model, encoder, scaler, and vectorizer for reuse.

---

## 🧠 Model Architecture

| Component | Description |
|------------|-------------|
| **TF-IDF Vectorizer** | Converts email subject and body text into numerical feature vectors. |
| **Metadata Encoder** | One-hot encodes sender domain information. |
| **Metadata Scaler** | Normalizes numerical metadata (e.g., number of URLs). |
| **Classifier** | Logistic Regression model trained to predict phishing vs. safe emails. |

---

## 📂 Project Structure

```
phishing-detector/
│
├── app.py                              # Streamlit web application
├── models/                             # Trained models and transformers
│   ├── tfidf_vectorizer.joblib
│   ├── phishing_model_with_meta.joblib
│   ├── metadata_encoder.joblib
│   └── metadata_scaler.joblib
├── train_phishing_detector.py          # Model training script
└── README.md                           # Project documentation
```

---

## ⚙️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/userIssa/Phishing-Email-Detector-Machine-Learning.git
   cd Phishing-Email-Detector-Machine-Learning
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   venv\Scripts\activate     # On Windows
   source venv/bin/activate  # On Mac/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

5. **Access the interface**
   - Open your browser and go to **http://localhost:8501**

---

## 🧾 Input Example

| Field | Example |
|--------|----------|
| **Sender Email** | support@paypal.com |
| **Subject** | Important: Verify your account now |
| **Body** | "Dear user, your account is locked. Click here to restore access..." |

**Result:** 🚨 *Phishing / Suspicious*

---

## 🧩 Datasets Used

The combined dataset (`combined_clean.csv`) was built from multiple open sources:

- **phishing_email.csv**
- **Nigerian_Fraud.csv**
- **SpamAssassin.csv**
- **Ling.csv**
- **enron.csv**
- **CEAS_08.csv**
- **Nazario.csv**

Each dataset contributed to creating a more balanced, realistic model capable of identifying a wide range of phishing and spam patterns.

---

## 📊 Model Evaluation

| Metric | Score |
|--------|-------|
| **Accuracy** | ~96% |
| **Precision** | High for both phishing and safe classes |
| **Recall** | Robust across multiple test subsets |

*(Exact metrics depend on the specific training split.)*

---

## 💡 Future Improvements

- Add transformer-based models (e.g., BERT or DistilBERT).  
- Include email header analysis (DKIM, SPF, etc.).  
- Expand dataset diversity for multilingual phishing detection.

---

## 🧑‍💻 Author

**Toluwanimi Oderinde**  
Cybersecurity Analyst | Software Engineer  
[LinkedIn](https://linkedin.com/in/toluwanimi-oderinde) · [GitHub](https://github.com/userIssa)

---

## 🛠️ License

This project is licensed under the **MIT License** — you are free to use, modify, and distribute with attribution.

---
