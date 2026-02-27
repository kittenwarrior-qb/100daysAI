"""
Comment Filter - Sentiment Analysis
Project AI đầu tiên: Phân loại comment tích cực/tiêu cực
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# 1. Load dữ liệu
print("📂 Đang load dữ liệu...")
df = pd.read_csv('reviews_sentiment_clean.csv')
print(f"✅ Đã load {len(df)} comments")
print(f"\nPhân bố nhãn:")
print(df['label'].value_counts())

# 2. Chuẩn bị dữ liệu
X = df['text']
y = df['label']

# Chia train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n📊 Train: {len(X_train)} | Test: {len(X_test)}")

# 3. Vectorize text (chuyển text thành số)
print("\n🔄 Đang vectorize text...")
vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 4. Train model
print("\n🤖 Đang train model...")
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 5. Đánh giá model
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✨ Kết quả:")
print(f"Accuracy: {accuracy:.2%}")
print(f"\n📈 Classification Report:")
print(classification_report(y_test, y_pred))

# 6. Lưu model
print("\n💾 Đang lưu model...")
with open('comment_filter_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
print("✅ Đã lưu model!")

# 7. Test thử với vài comment
print("\n🧪 Test thử:")
test_comments = [
    "This is awesome, I love it!",
    "Terrible service, waste of money",
    "Pretty good, would recommend"
]

for comment in test_comments:
    vec = vectorizer.transform([comment])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    print(f"\n'{comment}'")
    print(f"→ {pred} (confidence: {max(prob):.2%})")
