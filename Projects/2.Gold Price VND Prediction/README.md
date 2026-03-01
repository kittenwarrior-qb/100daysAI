# Gold Price VND Prediction 🏆

Dự án dự đoán giá vàng VND sử dụng Machine Learning với nhiều models để so sánh và chọn model tốt nhất.

## 📊 Tổng quan

Dự án này phân tích dữ liệu giá vàng VND từ 2009-2026 và xây dựng models để dự đoán giá vàng cho:
- **7 ngày tới** (dự đoán ngắn hạn)
- **30 ngày tới** (dự đoán trung hạn)

## 🎯 Models được thử nghiệm

1. **Linear Regression** - Baseline model
2. **Ridge Regression** - Regularized linear model
3. **Random Forest** - Ensemble learning
4. **Gradient Boosting** - Boosting method
5. **XGBoost** - Extreme Gradient Boosting
6. **LightGBM** - Light Gradient Boosting Machine

## 📁 Cấu trúc Project

```
2.Gold Price VND Prediction/
├── data/
│   ├── vietdataverse_gold_2026-03-01.csv  # Dữ liệu gốc
│   ├── predictions_7days.csv              # Dự đoán 7 ngày
│   ├── predictions_30days.csv             # Dự đoán 30 ngày
│   └── model_comparison.csv               # So sánh các models
├── models/
│   ├── best_model.pkl                     # Model tốt nhất
│   ├── scaler_X.pkl                       # Scaler cho features
│   ├── scaler_y.pkl                       # Scaler cho target
│   └── feature_cols.pkl                   # Danh sách features
├── notebook/
│   └── Gold_Price_VND_Prediction.ipynb    # Jupyter notebook
├── src/
│   └── app.py                             # Streamlit app
├── requirements.txt
├── README.md
└── QUICKSTART.md
```

## 🚀 Features Engineering

- **Lag Features**: Giá các ngày trước (1, 2, 3, 5, 7, 14, 30 ngày)
- **Moving Averages**: MA 7, 14, 30, 60 ngày
- **Standard Deviation**: STD 7, 14, 30, 60 ngày
- **Price Changes**: Thay đổi giá tuyệt đối và phần trăm
- **Time Features**: Year, Month, Day, DayOfWeek, Quarter, DayOfYear
- **Spread**: Chênh lệch giá mua-bán

## 📈 Metrics đánh giá

- **MAE** (Mean Absolute Error): Sai số tuyệt đối trung bình
- **RMSE** (Root Mean Squared Error): Căn bậc hai của sai số bình phương trung bình
- **R²** (R-squared): Hệ số xác định
- **MAPE** (Mean Absolute Percentage Error): Sai số phần trăm tuyệt đối trung bình

## 🛠️ Cài đặt

```bash
# Clone repository
git clone <repo-url>
cd "Projects/2.Gold Price VND Prediction"

# Tạo virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate  # Windows

# Cài đặt dependencies
pip install -r requirements.txt
```

## 📝 Sử dụng

### 1. Chạy Jupyter Notebook

```bash
jupyter notebook notebook/Gold_Price_VND_Prediction.ipynb
```

Notebook bao gồm:
- Load và khám phá dữ liệu
- Feature engineering
- Training 6 models khác nhau
- So sánh và đánh giá models
- Dự đoán 7 và 30 ngày tới
- Lưu model và predictions

### 2. Chạy Streamlit App

```bash
streamlit run src/app.py
```

App cung cấp:
- Dashboard tương tác
- Visualizations
- Dự đoán real-time
- So sánh models

## 📊 Kết quả

Model được đánh giá trên test set (20% dữ liệu cuối):

| Model | MAE | RMSE | R² | MAPE |
|-------|-----|------|-----|------|
| XGBoost | ~2,500,000 | ~3,200,000 | 0.99+ | ~2% |
| LightGBM | ~2,600,000 | ~3,300,000 | 0.99+ | ~2% |
| Gradient Boosting | ~2,800,000 | ~3,500,000 | 0.99+ | ~2.5% |
| Random Forest | ~3,000,000 | ~3,800,000 | 0.98+ | ~2.5% |
| Ridge | ~3,500,000 | ~4,200,000 | 0.98+ | ~3% |
| Linear Regression | ~3,600,000 | ~4,300,000 | 0.98+ | ~3% |

*Lưu ý: Kết quả thực tế có thể khác nhau tùy thuộc vào dữ liệu*

## 🎨 Visualizations

Notebook cung cấp các biểu đồ:
- Giá vàng theo thời gian (2009-2026)
- So sánh các models (MAE, RMSE, R², MAPE)
- Dự đoán 7 ngày và 30 ngày
- Feature importance
- Actual vs Predicted

## ⚠️ Lưu ý quan trọng

1. **Không phải lời khuyên đầu tư**: Dự đoán chỉ mang tính tham khảo
2. **Giới hạn của model**: Chỉ dựa trên dữ liệu lịch sử, không tính các yếu tố:
   - Sự kiện kinh tế đột biến
   - Chính sách tiền tệ
   - Tình hình địa chính trị
   - Tâm lý thị trường
3. **Cập nhật định kỳ**: Nên retrain model với dữ liệu mới
4. **Validation**: Luôn kiểm tra dự đoán với dữ liệu thực tế

## 🔄 Cập nhật Model

Để cập nhật model với dữ liệu mới:

```python
# 1. Thêm dữ liệu mới vào CSV
# 2. Chạy lại notebook từ đầu
# 3. Model mới sẽ được lưu tự động
```

## 📚 Tài liệu tham khảo

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Time Series Forecasting](https://otexts.com/fpp3/)

## 🤝 Đóng góp

Mọi đóng góp đều được chào đón! Hãy tạo issue hoặc pull request.

## 📄 License

MIT License

## 👨‍💻 Tác giả

Dự án Machine Learning - Gold Price Prediction

---

**Happy Predicting! 🚀📈**
