import pandas as pd
from sklearn.utils import resample
import os

# 📁 Path to your dataset folder
DATA_DIR = "C:/Users/issa/Downloads/archive/data-raw"

# ✅ Helper function to load and normalize any dataset
def load_dataset(filename):
    path = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(path, encoding='latin-1')
    df.columns = [c.strip().lower() for c in df.columns]  # normalize column names

    # Combine text columns
    if 'text_combined' in df.columns:
        df['text'] = df['text_combined']
    elif {'subject', 'body'}.issubset(df.columns):
        df['text'] = df['subject'].fillna('') + ' ' + df['body'].fillna('')
    else:
        df['text'] = df[df.columns[0]].astype(str)

    # Normalize labels
    if 'label' in df.columns:
        df['label'] = df['label'].astype(str).str.lower()
    elif 'label (num)' in df.columns:
        df['label'] = df['label (num)']
    else:
        df['label'] = 0  # fallback if not present

    # Map label values to 0/1
    df['label'] = df['label'].replace({
        'ham': 0, 'legit': 0, 'safe': 0, '0': 0, 0: 0,
        'spam': 1, 'phishing': 1, 'fraud': 1, '1': 1, 1: 1
    })

    return df[['text', 'label']].dropna()

# 🧾 Datasets
datasets = [
    "phishing_email.csv",
    "Nigerian_Fraud.csv",
    "SpamAssasin.csv",
    "Ling.csv",
    "enron.csv",
    "CEAS_08.csv",
    "Nazario.csv"
]

# 📚 Load and merge all datasets
dfs = [load_dataset(f) for f in datasets]
df = pd.concat(dfs, ignore_index=True)

print(f"Total samples before balancing: {len(df)}")
print(df['label'].value_counts())

# ⚖️ Rebalance (equal phishing/safe samples)
phish = df[df['label'] == 1]
safe = df[df['label'] == 0]
min_size = min(len(phish), len(safe))

phish_bal = resample(phish, replace=False, n_samples=min_size, random_state=42)
safe_bal = resample(safe, replace=False, n_samples=min_size, random_state=42)

df_balanced = pd.concat([phish_bal, safe_bal]).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"✅ Balanced dataset: {len(df_balanced)} samples ({len(phish_bal)} phishing / {len(safe_bal)} safe)")

# 💾 Save cleaned dataset
output_path = os.path.join(DATA_DIR, "combined_clean.csv")
df_balanced.to_csv(output_path, index=False)
print(f"💾 Saved combined dataset to: {output_path}")
