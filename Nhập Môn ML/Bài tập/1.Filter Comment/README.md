# Comment Filter - Sentiment Analysis

Dự án phân loại sentiment của comment (tích cực/tiêu cực) sử dụng Machine Learning.

## Tổng quan

Project này sử dụng Natural Language Processing (NLP) để tự động phân loại comment thành 2 loại:
- **Positive**: Comment tích cực, khen ngợi
- **Negative**: Comment tiêu cực, phàn nàn

## Công nghệ sử dụng

### Machine Learning
- **Model**: Multinomial Naive Bayes
  - Thuật toán phân loại xác suất dựa trên định lý Bayes
  - Phù hợp cho bài toán phân loại văn bản
  - Nhanh, hiệu quả với dữ liệu text

- **Vectorizer**: TF-IDF (Term Frequency-Inverse Document Frequency)
  - Chuyển đổi text thành vector số
  - `max_features=1000`: Giới hạn 1000 từ quan trọng nhất
  - `ngram_range=(1, 2)`: Sử dụng unigram và bigram

### Libraries
```
scikit-learn  # Machine Learning
pandas        # Xử lý dữ liệu
numpy         # Tính toán số học
flask         # Web framework
```

## Dataset

- File: `reviews_sentiment_clean.csv`
- Cấu trúc:
  - `text`: Nội dung comment
  - `label`: Nhãn (positive/negative)
- Train/Test split: 80/20

## Hướng dẫn sử dụng

### 1. Cài đặt dependencies

```bash
pip install pandas numpy scikit-learn flask
```

### 2. Train model

```bash
python comment_filter.py
```

Kết quả:
- `comment_filter_model.pkl`: Model đã train
- `vectorizer.pkl`: TF-IDF vectorizer
- In ra accuracy và classification report

### 3. Predict comment mới

#### Cách 1: Chạy với Terminal

```bash
python predict.py
```

Nhập comment và nhận kết quả ngay lập tức.

#### Cách 2: Chạy với Flask Web App

```bash
python app_flask.py
```

Mở browser tại: http://localhost:5000

Web app cung cấp giao diện đẹp với:
- Form nhập comment
- Hiển thị kết quả real-time
- Confidence score dạng phần trăm
- UI responsive, dễ sử dụng



## Cách hoạt động

1. **Preprocessing**: Text được làm sạch và chuẩn hóa
2. **Vectorization**: TF-IDF chuyển text thành vector số
3. **Classification**: Naive Bayes phân loại dựa trên vector
4. **Output**: Trả về nhãn (positive/negative) và confidence score

## Kết quả

Model đạt accuracy cao trên test set (xem chi tiết khi chạy `comment_filter.py`)

## Ví dụ

```python
from predict import predict_comment

result = predict_comment("This product is amazing!")
# Output: {'sentiment': 'positive', 'confidence': 0.95}

result = predict_comment("Terrible service, very disappointed")
# Output: {'sentiment': 'negative', 'confidence': 0.89}
```

## Tùy chỉnh

### Thay đổi model
Trong `comment_filter.py`, thay thế:
```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
```

### Điều chỉnh vectorizer
```python
vectorizer = TfidfVectorizer(
    max_features=2000,      # Tăng số features
    ngram_range=(1, 3),     # Thêm trigram
    min_df=2                # Bỏ từ xuất hiện < 2 lần
)
```

## Lưu ý

- Đảm bảo file `.pkl` tồn tại trước khi chạy predict/web app
- Model hoạt động tốt nhất với comment tiếng Anh
- Với tiếng Việt, cần thêm bước tiền xử lý (loại bỏ dấu, tokenize)

## Flask Web App

Web app được xây dựng với Flask framework:
- `app_flask.py`: Backend xử lý logic (load model, predict, API endpoint)
- `templates/index.html`: Frontend giao diện người dùng (HTML/CSS/JavaScript)

Flask cho phép:
- Tùy chỉnh giao diện hoàn toàn
- Tạo REST API để tích hợp với app khác
- Deploy lên production dễ dàng (Heroku, Railway, AWS, etc.)

## Nâng cao

- [ ] Thêm preprocessing cho tiếng Việt
- [ ] Thử các model khác (SVM, Random Forest, Deep Learning)
- [ ] Deploy lên cloud (Heroku, Railway, Render)
- [ ] Thêm phân loại nhiều cấp độ (1-5 sao)
- [ ] Tạo REST API endpoint cho mobile app

## License

Free to use for learning purposes.
