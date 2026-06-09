import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from scipy.sparse import hstack

import matplotlib.pyplot as plt
import seaborn as sns

# ============================================
# LOAD DATASET
# ============================================

# Replace with your dataset path if needed
df = pd.read_csv("/content/CEAS_08.csv")

# ============================================
# CHECK DATASET COLUMNS
# ============================================

print("Dataset Columns:\n")
print(df.columns)

# ============================================
# SELECT EMAIL TEXT COLUMN AND LABEL COLUMN
# ============================================

# Change these column names according to your dataset
# Example:
# email_body = email content column
# label = phishing/safe column

X_text_data = df["body"]     # Email text column
y = df["label"]              # Target column

# ============================================
# CLEAN NULL VALUES
# ============================================

X_text_data = X_text_data.fillna("")

# ============================================
# FEATURE EXTRACTION FUNCTION
# ============================================

def extract_url_features(emails):

    features = []

    suspicious_words = [
        "verify",
        "login",
        "password",
        "account",
        "urgent",
        "bank",
        "click",
        "update",
        "suspended",
        "security",
        "free",
        "winner"
    ]

    for email in emails:

        # Extract URLs
        urls = re.findall(r'https?://\S+|www\.\S+', str(email))

        num_urls = len(urls)

        # Count suspicious keywords
        keyword_count = sum(
            str(email).lower().count(word)
            for word in suspicious_words
        )

        # Email length
        email_length = len(str(email))

        features.append([
            num_urls,
            keyword_count,
            email_length
        ])

    return np.array(features)

# ============================================
# TF-IDF TEXT FEATURES
# ============================================

tfidf = TfidfVectorizer(
    stop_words='english',
    max_features=5000
)

X_text = tfidf.fit_transform(X_text_data)

# ============================================
# URL + KEYWORD FEATURES
# ============================================

X_url = extract_url_features(X_text_data)

# ============================================
# COMBINE FEATURES
# ============================================

X = hstack((X_text, X_url))

# ============================================
# TRAIN TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ============================================
# TRAIN MODEL
# ============================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# ============================================
# PREDICTIONS
# ============================================

y_pred = model.predict(X_test)

# ============================================
# EVALUATION
# ============================================

accuracy = accuracy_score(y_test, y_pred)

print("\n========== RESULTS ==========\n")

print("Accuracy:", round(accuracy * 100, 2), "%\n")

print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# ============================================
# CONFUSION MATRIX
# ============================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Phishing', 'Safe'],
    yticklabels=['Phishing', 'Safe']
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()

# ============================================
# TEST CUSTOM EMAIL
# ============================================

def predict_email(email_text):

    text_vector = tfidf.transform([email_text])

    url_vector = extract_url_features([email_text])

    combined = hstack((text_vector, url_vector))

    prediction = model.predict(combined)[0]

    return prediction

# ============================================
# CUSTOM TEST
# ============================================

test_email = """
Dear User,

Your account has been suspended.

Please login immediately:

http://secure-bank-login.xyz
"""

result = predict_email(test_email)

print("\n========== CUSTOM EMAIL TEST ==========\n")
print("Prediction:", result)
