# 🚀 Quick Start Guide

Hướng dẫn nhanh để bắt đầu với Gold Price VND Prediction.

## ⚡ Bắt đầu trong 5 phút

### Bước 1: Cài đặt

```bash
# Di chuyển vào thư mục project
cd "Projects/2.Gold Price VND Prediction"

# Cài đặt dependencies
pip install -r requirements.txt
```

### Bước 2: Chạy Notebook

```bash
# Mở Jupyter Notebook
jupyter notebook notebook/Gold_Price_VND_Prediction.ipynb

# Chạy tất cả cells (Kernel > Restart & Run All)
```

### Bước 3: Xem kết quả

Sau khi chạy notebook, bạn sẽ có:
- ✅ 6 models đã được train và đánh giá
- ✅ Dự đoán 7 ngày: `data/predictions_7days.csv`
- ✅ Dự đoán 30 ngày: `data/predictions_30days.csv`
- ✅ So sánh models: `data/model_comparison.csv`
- ✅ Best model: `models/best_model.pkl`

## 📊 Xem Predictions

```python
import pandas as pd

# Đọc dự đoán 7 ngày
pred_7 = pd.read_csv('data/predictions_7days.csv')
print(pred_7)

# Đọc dự đoán 30 ngày
pred_30 = pd.read_csv('data/predictions_30days.csv')
print(pred_30)
```

## 🎯 Sử dụng Model đã train

```python
import joblib
import pandas as pd
import numpy as np

# Load model và scalers
model = joblib.load('models/best_model.pkl')
scaler_X = joblib.load('models/scaler_X.pkl')
scaler_y = joblib.load('models/scaler_y.pkl')
feature_cols = joblib.load('models/feature_cols.pkl')

# Load dữ liệu
df = pd.read_csv('data/vietdataverse_gold_2026-03-01.csv')
# ... (feature engineering như trong notebook)

# Dự đoán
X_new = df[feature_cols].iloc[-1:] # Lấy features mới nhất
X_new_scaled = scaler_X.transform(X_new)
pred_scaled = model.predict(X_new_scaled)
pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]

print(f"Dự đoán giá vàng: {pred:,.0f} VND")
```

## 🎨 Chạy Streamlit App (Coming Soon)

```bash
streamlit run src/app.py
```

## 📈 Workflow tổng quan

```
1. Load Data (CSV)
   ↓
2. Feature Engineering
   ↓
3. Train Multiple Models
   ↓
4. Compare & Select Best
   ↓
5. Make Predictions
   ↓
6. Save Results
```

## 🔍 Các Models được thử nghiệm

1. **Linear Regression** ⚡ - Nhanh, đơn giản
2. **Ridge Regression** 🎯 - Regularized
3. **Random Forest** 🌲 - Ensemble
4. **Gradient Boosting** 📈 - Boosting
5. **XGBoost** 🚀 - Hiệu suất cao
6. **LightGBM** ⚡ - Nhanh nhất

## 📊 Metrics

- **MAE**: Sai số trung bình (VND)
- **RMSE**: Sai số bình phương (VND)
- **R²**: Độ chính xác (0-1, càng cao càng tốt)
- **MAPE**: Sai số phần trăm (%)

## 🎓 Tips

### Cải thiện Model

1. **Thêm features**:
   - Giá USD/VND
   - Lãi suất
   - Chỉ số chứng khoán
   - Giá dầu

2. **Tune hyperparameters**:
   ```python
   from sklearn.model_selection import GridSearchCV
   
   param_grid = {
       'n_estimators': [100, 200, 300],
       'max_depth': [5, 10, 15],
       'learning_rate': [0.01, 0.1, 0.2]
   }
   
   grid_search = GridSearchCV(XGBRegressor(), param_grid, cv=5)
   grid_search.fit(X_train, y_train)
   ```

3. **Ensemble methods**:
   - Kết hợp nhiều models
   - Voting hoặc Stacking

### Xử lý Outliers

```python
# Phát hiện outliers
Q1 = df['Avg_Price'].quantile(0.25)
Q3 = df['Avg_Price'].quantile(0.75)
IQR = Q3 - Q1

# Loại bỏ outliers
df_clean = df[
    (df['Avg_Price'] >= Q1 - 1.5*IQR) & 
    (df['Avg_Price'] <= Q3 + 1.5*IQR)
]
```

## ⚠️ Troubleshooting

### Lỗi thường gặp

**1. ModuleNotFoundError**
```bash
pip install <missing-module>
```

**2. Memory Error**
```python
# Giảm kích thước data hoặc sử dụng sampling
df_sample = df.sample(frac=0.5)
```

**3. Model không converge**
```python
# Tăng số iterations
model = XGBRegressor(n_estimators=500)
```

## 📚 Tài nguyên học thêm

- [Time Series Analysis](https://www.kaggle.com/learn/time-series)
- [XGBoost Tutorial](https://xgboost.readthedocs.io/en/stable/tutorials/index.html)
- [Feature Engineering](https://www.kaggle.com/learn/feature-engineering)

## 🤔 FAQ

**Q: Model có chính xác không?**
A: Model đạt R² > 0.98 trên test set, nhưng không đảm bảo cho tương lai.

**Q: Bao lâu nên retrain?**
A: Nên retrain hàng tuần hoặc khi có dữ liệu mới.

**Q: Có thể dự đoán xa hơn 30 ngày?**
A: Có, nhưng độ chính xác giảm dần theo thời gian.

**Q: Model nào tốt nhất?**
A: Thường là XGBoost hoặc LightGBM, nhưng cần test trên data của bạn.

## 🎯 Next Steps

1. ✅ Chạy notebook và xem kết quả
2. 📊 Phân tích feature importance
3. 🔧 Thử tune hyperparameters
4. 📈 Thêm features mới
5. 🚀 Deploy model (Streamlit/FastAPI)

---

**Chúc bạn thành công! 🎉**

Nếu có vấn đề, hãy tạo issue trên GitHub.
