"""
Gradio App - Dự Đoán Giá Vàng VND
Đơn giản, chỉ cần 2 inputs: USD/VND và Gold Price USD
"""

import gradio as gr
import numpy as np
import joblib
import os

# Đường dẫn models
MODEL_PATH = "../models/best_model.pkl"
SCALER_PATH = "../models/scaler.pkl"

def load_models():
    """Load trained model và scaler"""
    try:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
            return None, None
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except Exception as e:
        print(f"Lỗi khi load models: {e}")
        return None, None

def predict_gold_price(usd_vnd, gold_price_usd):
    """
    Dự đoán giá vàng VND
    
    Args:
        usd_vnd: Tỷ giá USD/VND
        gold_price_usd: Giá vàng thế giới (USD/oz)
    
    Returns:
        Giá vàng dự đoán (VND/chỉ)
    """
    # Load models
    model, scaler = load_models()
    
    if model is None or scaler is None:
        return """
        ⚠️ **Model chưa được train!**
        
        **Hướng dẫn:**
        1. Mở notebook `Gold_Price_VND_Prediction.ipynb`
        2. Chạy tất cả cells để train model
        3. Models sẽ được lưu tự động vào thư mục `models/`
        4. Restart app này
        
        **Hoặc:**
        - Chạy: `jupyter notebook notebook/Gold_Price_VND_Prediction.ipynb`
        """
    
    try:
        # Validate inputs
        if usd_vnd <= 0 or gold_price_usd <= 0:
            return "❌ Lỗi: Giá trị phải lớn hơn 0"
        
        # Prepare input
        input_data = np.array([[usd_vnd, gold_price_usd]])
        input_scaled = scaler.transform(input_data)
        
        # Predict
        prediction = model.predict(input_scaled)[0]
        
        # Format output
        result = f"""
        ### 🏆 Kết Quả Dự Đoán
        
        **Giá Vàng VND:** {prediction:,.0f} VND/chỉ
        
        ---
        
        **Thông Tin Input:**
        - Tỷ giá USD/VND: {usd_vnd:,.0f}
        - Giá vàng thế giới: ${gold_price_usd:,.2f}/oz
        
        ---
        
        **Giá quy đổi:**
        - 1 chỉ = 3.75 gram
        - Giá vàng/gram: {prediction/3.75:,.0f} VND
        """
        
        return result
        
    except Exception as e:
        return f"❌ Lỗi: {str(e)}"

# Tạo Gradio interface
with gr.Blocks(title="Dự Đoán Giá Vàng VND", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("""
    # 🏆 Dự Đoán Giá Vàng VND
    
    Ứng dụng Machine Learning dự đoán giá vàng tại Việt Nam dựa trên tỷ giá USD/VND và giá vàng thế giới.
    """)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📊 Nhập Thông Tin")
            
            usd_vnd = gr.Number(
                label="Tỷ giá USD/VND",
                value=24500,
                info="Tỷ giá hiện tại (VD: 24500)"
            )
            
            gold_price_usd = gr.Number(
                label="Giá vàng thế giới (USD/oz)",
                value=2100,
                info="Giá vàng quốc tế (VD: 2100)"
            )
            
            predict_btn = gr.Button("🔮 Dự Đoán Giá Vàng", variant="primary", size="lg")
        
        with gr.Column():
            output = gr.Markdown(label="Kết Quả")
    
    predict_btn.click(
        fn=predict_gold_price,
        inputs=[usd_vnd, gold_price_usd],
        outputs=output
    )
    
    gr.Markdown("""
    ---
    
    ### 📝 Hướng Dẫn:
    1. Nhập tỷ giá USD/VND hiện tại (kiểm tra tại vietcombank.com.vn)
    2. Nhập giá vàng thế giới (kiểm tra tại kitco.com)
    3. Nhấn "Dự Đoán Giá Vàng"
    
    ### 📌 Lưu Ý:
    - Model được train với dữ liệu mẫu
    - Kết quả chỉ mang tính tham khảo
    - Để cải thiện độ chính xác, cần train với dữ liệu thực tế nhiều hơn
    
    ### 🔗 Nguồn Dữ Liệu:
    - Tỷ giá: [Vietcombank](https://vietcombank.com.vn)
    - Giá vàng VN: [SJC](https://sjc.com.vn)
    - Giá vàng thế giới: [Kitco](https://kitco.com)
    """)

if __name__ == "__main__":
    demo.launch()
