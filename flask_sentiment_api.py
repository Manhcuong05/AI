from flask import Flask, request, jsonify
import joblib
from predict_sentiment import preprocess_text, predict_sentiment

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    comment = data.get("comment", "")
    if not comment:
        return jsonify({"error": "No comment provided"}), 400
    
    sentiment = predict_sentiment(comment)
    return jsonify({"comment": comment, "sentiment": sentiment})

if __name__ == '__main__':
    print("✅ API đang chạy trên http://127.0.0.1:5000")
    app.run(debug=True)
