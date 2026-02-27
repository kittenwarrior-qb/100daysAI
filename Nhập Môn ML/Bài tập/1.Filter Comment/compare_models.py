"""
So sánh các ML Models cho Comment Filter
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
import time

# Import các models
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

print("=" * 60)
print("🔬 SO SÁNH CÁC ML MODELS CHO COMMENT FILTER")
print("=" * 60)

# 1. Load dữ liệu
print("\n📂 Đang load dữ liệu...")
df = pd.read_csv('reviews_sentiment_clean.csv')
print(f"✅ Đã load {len(df)} comments")

X = df['text']
y = df['label']

# 2. Chia train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Vectorize
print("\n🔄 Đang vectorize text...")
vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 4. Định nghĩa các models
models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'SVM (Linear)': SVC(kernel='linear', random_state=42),
    'SVM (RBF)': SVC(kernel='rbf', random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'KNN (k=5)': KNeighborsClassifier(n_neighbors=5)
}

# 5. Train và đánh giá từng model
results = []

print("\n" + "=" * 60)
print("🤖 BẮT ĐẦU TRAINING VÀ ĐÁNH GIÁ")
print("=" * 60)

for name, model in models.items():
    print(f"\n{'─' * 60}")
    print(f"📊 Model: {name}")
    print(f"{'─' * 60}")
    
    # Training time
    start_time = time.time()
    model.fit(X_train_vec, y_train)
    train_time = time.time() - start_time
    
    # Prediction time
    start_time = time.time()
    y_pred = model.predict(X_test_vec)
    pred_time = time.time() - start_time
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Cross-validation (5-fold)
    print("⏳ Đang chạy cross-validation...")
    cv_scores = cross_val_score(model, X_train_vec, y_train, cv=5)
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    # Lưu kết quả
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'CV Mean': cv_mean,
        'CV Std': cv_std,
        'Train Time (s)': train_time,
        'Predict Time (s)': pred_time
    })
    
    print(f"✅ Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"📊 CV Score: {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"⏱️  Train Time: {train_time:.4f}s")
    print(f"⚡ Predict Time: {pred_time:.4f}s")

# 6. Tổng hợp kết quả
print("\n" + "=" * 60)
print("📈 BẢNG TỔNG HỢP KẾT QUẢ")
print("=" * 60)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Accuracy', ascending=False)

print("\n" + results_df.to_string(index=False))

# 7. Phân tích
print("\n" + "=" * 60)
print("💡 PHÂN TÍCH VÀ KHUYẾN NGHỊ")
print("=" * 60)

best_accuracy = results_df.iloc[0]
fastest_train = results_df.loc[results_df['Train Time (s)'].idxmin()]
fastest_predict = results_df.loc[results_df['Predict Time (s)'].idxmin()]

print(f"\n🏆 Model tốt nhất (Accuracy): {best_accuracy['Model']}")
print(f"   → Accuracy: {best_accuracy['Accuracy']:.4f}")

print(f"\n⚡ Model train nhanh nhất: {fastest_train['Model']}")
print(f"   → Train time: {fastest_train['Train Time (s)']:.4f}s")

print(f"\n🚀 Model predict nhanh nhất: {fastest_predict['Model']}")
print(f"   → Predict time: {fastest_predict['Predict Time (s)']:.4f}s")

print("\n📝 Khuyến nghị:")
print("   • Nếu cần ACCURACY cao: Dùng", best_accuracy['Model'])
print("   • Nếu cần SPEED: Dùng", fastest_predict['Model'])
print("   • Nếu cần BALANCE: Dùng Logistic Regression hoặc Naive Bayes")

# 8. Detailed report cho model tốt nhất
print("\n" + "=" * 60)
print(f"📊 CHI TIẾT MODEL TỐT NHẤT: {best_accuracy['Model']}")
print("=" * 60)

best_model = models[best_accuracy['Model']]
best_model.fit(X_train_vec, y_train)
y_pred_best = best_model.predict(X_test_vec)

print("\n" + classification_report(y_test, y_pred_best))

print("\n✅ Hoàn thành!")
