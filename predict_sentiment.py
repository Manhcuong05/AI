import joblib
from underthesea import word_tokenize
import re

def preprocess_text(text):
    """Tiền xử lý văn bản giống như khi train model."""
    stopwords = set(["là", "của", "và", "đã", "cũng", "này", "được", "bị", "với", "cho", "có"])
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Loại bỏ dấu câu
    text = word_tokenize(text, format="text")  # Tách từ tiếng Việt
    text = " ".join([word for word in text.split() if word not in stopwords])  # Loại bỏ stopwords
    return text

# Load mô hình
model = joblib.load("sentiment_model.pkl")

def predict_sentiment(comment):
    """Dự đoán cảm xúc của bình luận."""
    processed_comment = preprocess_text(comment)
    prediction = model.predict([processed_comment])[0]
    return prediction

# Ví dụ sử dụng
if __name__ == "__main__":
    test_comment = "Sản phẩm này dùng rất tệ"
    sentiment = predict_sentiment(test_comment)
    print(f"🔍 Bình luận: {test_comment}")
    print(f"💡 Cảm xúc dự đoán: {sentiment}")
