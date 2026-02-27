Project
Dự Đoán Cảm Xúc
Sentiment Analysis


Giới thiệu / Motivation
Giả sử bạn có một đống review phim hoặc sản phẩm:
Có review kiểu: “This movie is amazing, I love it!”
Có review kiểu: “This joint sucks big time, don’t bother.”
Có review kiểu: “The support team is super helpful, I’m impressed.”
👉 Nhiệm vụ của bạn: xây dựng một công cụ để tự động đoán review đó là tích cực (positive) hay tiêu cực (negative).
Tại sao cần công cụ này?
Nếu bạn là shop bán hàng → có thể phân tích nhanh 10.000 feedback của khách hàng.
Nếu bạn quản lý phim/nhà hàng → biết ngay khách đang hài lòng hay bức xúc.
Nếu bạn là người học AI → có ngay một project portfolio “ngon lành” để show.
Nói đơn giản:
“Đọc một câu review → đoán xem người viết đang khen hay chê.”
Dataset (Bộ dữ liệu)
Nguồn dữ liệu
Bộ dữ liệu được chuẩn bị sẵn với 2000 review tiếng Anh.
Mỗi review đã có nhãn positive/negative.
Được lưu trong file CSV: reviews_sentiment_clean.csv.
Thông tin tổng quan
Số dòng: 2000 (mỗi dòng = 1 review).
Số cột:
text → nội dung review.
label → cảm xúc (positive / negative).

Ví dụ 5 dòng đầu:

👉 Dữ liệu này khá gọn, cân bằng (1000 positive, 1000 negative), nên dễ để bắt đầu.
Quy trình xử lý dữ liệu (Data Preprocessing)
Trước khi cho vào mô hình, ta cần “dọn dẹp” dữ liệu:
 Làm sạch dữ liệu
Bỏ dòng bị thiếu (NA).
Đưa toàn bộ text về dạng chữ thường.
Bỏ khoảng trắng thừa.
Mã hóa nhãn
Đổi positive → 1, negative → 0:
Chia train/test
80% dữ liệu để huấn luyện (train).
20% để kiểm tra (test).
Biến chữ thành số (TF-IDF)
Máy tính không hiểu chữ, nên ta dùng TF-IDF để biến chữ thành vector số.
👉 Hiểu đơn giản: TF-IDF = “tính xem từ nào quan trọng trong câu”.

Mô hình (Modeling)
Chọn mô hình nào?
Ở đây ta chọn Logistic Regression.
Dễ hiểu, chạy nhanh.
Rất phù hợp với bài toán phân loại 2 lớp (positive / negative).
Logistic Regression là gì? 🤔
Nó không phải “hồi quy” theo nghĩa dự đoán số, mà là một công cụ phân loại.
Nôm na: logistic regression vẽ một “đường ranh giới” chia dữ liệu thành 2 nhóm:
Nhóm review tích cực.
Nhóm review tiêu cực.
👉 Sau khi “fit”, mô hình đã học được cách phân biệt review khen/chê.

Đánh giá mô hình 🎯
 Độ chính xác
=== Logistic Regression (train 80% / test 20%) ===
Accuracy: 1.0
Khi huấn luyện xong, ta kiểm tra mô hình bằng tập test (20% dữ liệu chưa từng “cho học”).
Kết quả: Accuracy = 1.0 (100%).
👉 Nghĩa là: trong 400 review test, mô hình đoán đúng cả 400/400.
Điều này nghe có vẻ “hoàn hảo”, nhưng cần lưu ý:
Bộ dữ liệu này cân bằng (50% positive, 50% negative).
Dữ liệu khá sạch và rõ ràng (review chê thì toàn từ tiêu cực, review khen thì toàn từ tích cực).
Vì vậy Logistic Regression có thể phân biệt rất dễ → dẫn tới accuracy = 100%.
📌 Trong thực tế, với dữ liệu phức tạp (review lẫn lộn, có từ đa nghĩa, viết tắt, emoji…), độ chính xác thường chỉ khoảng 80–90%, chứ không “perfect” như ở đây.
👉 Cách hiểu đơn giản:
“Độ chính xác = bao nhiêu phần trăm review mà mô hình đoán đúng. Ở đây model làm đúng hết nên được 100%. Nhưng ngoài đời, gặp dữ liệu thật thì sẽ khó hơn nhiều, và accuracy hiếm khi nào đạt 100%.”


Confusion Matrix (Ma trận nhầm lẫn)

Đây là một ma trận thể hiện chi tiết mô hình đoán đúng/sai bao nhiêu.
Trong hình trên:
Trục dọc (Actual) = giá trị thật (ground truth).
Trục ngang (Predicted) = giá trị mà mô hình dự đoán.
0 = Negative (tiêu cực).
1 = Positive (tích cực).
📌 Ý nghĩa các ô:
Ô trên bên trái (200): review thật là Negative, mô hình đoán cũng là Negative → đoán đúng.
Ô dưới bên phải (200): review thật là Positive, mô hình đoán cũng là Positive → đoán đúng.
Hai ô còn lại (0, 0): không có trường hợp nào mô hình đoán sai.

👉 Nói nôm na:
Có 200 review tiêu cực → đoán đúng cả 200.
Có 200 review tích cực → đoán đúng cả 200.
Không có sai sót nào.
🎯 Kết luận: Mô hình phân loại chính xác 100% trên tập test này.
⚠️ Lưu ý: Đây là kết quả “đẹp như mơ” vì dataset nhỏ và rõ ràng. Với dữ liệu thực tế (review lẫn lộn, dùng từ mập mờ, viết tắt, emoji…), confusion matrix thường sẽ có thêm số ở ô “sai” (false positive/false negative).
Classification Report
--------------------------------------------------------------------------
Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00       200
           1       1.00      1.00      1.00       200

    accuracy                           1.00       400
   macro avg       1.00      1.00      1.00       400
weighted avg       1.00      1.00      1.00       400
Khi đánh giá mô hình phân loại, ngoài Accuracy, ta còn quan tâm 3 chỉ số quan trọng khác: Precision, Recall, F1-score.

Ý nghĩa từng chỉ số:
Precision (Độ chính xác theo Positive)
👉 Trong số những review mà mô hình dự đoán là Positive, có bao nhiêu review thật sự Positive?
Ví dụ: nếu mô hình đoán 100 review là tích cực, và đúng 95 cái, sai 5 cái 
→ precision = 95%.
Recall (Độ bao phủ)
👉 Trong số tất cả review Positive thật, mô hình tìm lại được bao nhiêu?
Ví dụ: có 100 review tích cực, mô hình nhận ra đúng 90 cái, bỏ sót 10 cái → recall = 90%.
F1-score
👉 Trung bình “hài hòa” giữa precision và recall (nếu một cái cao, một cái thấp thì F1-score sẽ cân bằng lại).
Dùng để đánh giá tổng thể mô hình có “ổn định” không.
Support
👉 Số lượng mẫu thật sự trong mỗi lớp (ở đây: 200 review Negative, 200 review Positive).
Kết quả ở đây:
Tất cả chỉ số đều = 1.00 (100%) → nghĩa là mô hình đoán hoàn hảo cả hai lớp.
Accuracy tổng thể = 100% trên 400 review test.
⚠️ Tuy nhiên: Giống như đã nói ở phần Accuracy và Confusion Matrix, kết quả này có được vì dữ liệu cân bằng và sạch. Trong thực tế, review thường “mập mờ” hơn (có câu khen mà chêm thêm chê), nên precision/recall/F1-score sẽ thấp hơn.

👉 Cách hiểu đơn giản:
Precision = “Máy có hay bị đoán nhầm không?”
Recall = “Máy có bỏ sót nhiều không?”
F1-score = “Cân bằng giữa không nhầm và không sót.”
Giao diện ứng dụng (Interface)
Dùng Streamlit để làm app web:
Dashboard: giới thiệu model, vectorizer, accuracy.
Test Demo: nhập 1 review → dự đoán ngay.
Upload CSV: tải file nhiều review → phân tích hàng loạt.


Tóm lại
Toàn bộ quy trình có thể tóm gọn:
Data: lấy file CSV review.
Clean: xử lý dữ liệu → bỏ NA, chuẩn hóa text.
Split: chia train/test.
Vectorize: dùng TF-IDF biến chữ thành số.
Train: huấn luyện Logistic Regression.
Evaluate: đo độ chính xác, confusion matrix.
Deploy: làm app Streamlit với 3 tính năng: Dashboard, Test Demo, Upload CSV.
👉 Kết quả: một ứng dụng AI nhỏ gọn, dễ dùng, beginner-friendly, vừa học được ML cơ bản, vừa có giao diện đẹp để show portfolio.

