import pickle

# Load model đã train
with open('comment_filter_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

def predict_comment(text):
    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]
    probability = model.predict_proba(vec)[0]
    confidence = max(probability)
    
    return {
        'text': text,
        'sentiment': prediction,
        'confidence': confidence
    }

# Test
if __name__ == "__main__":
    print("Comment Filter - Nhập comment để phân tích\n")
    
    while True:
        comment = input("Nhập comment (hoặc 'quit' để thoát): ")
        if comment.lower() == 'quit':
            break
            
        result = predict_comment(comment)
        emoji = "😊" if result['sentiment'] == 'positive' else "😞"
        
        print(f"{emoji} Sentiment: {result['sentiment']}")
        print(f"📊 Confidence: {result['confidence']:.2%}\n")
