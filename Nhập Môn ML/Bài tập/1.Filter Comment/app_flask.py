"""
Web App Flask đơn giản (alternative cho Streamlit)
Chạy: python app_flask.py
"""

from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)

# Load model
with open('comment_filter_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    comment = data.get('comment', '')
    
    if not comment.strip():
        return jsonify({'error': 'Comment không được để trống'}), 400
    
    # Predict
    vec = vectorizer.transform([comment])
    prediction = model.predict(vec)[0]
    probability = model.predict_proba(vec)[0]
    confidence = float(max(probability))
    
    return jsonify({
        'sentiment': prediction,
        'confidence': confidence
    })

if __name__ == '__main__':
    print("🚀 Server đang chạy tại: http://localhost:5000")
    app.run(debug=True, port=5000)
