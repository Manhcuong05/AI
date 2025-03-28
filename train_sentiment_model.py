import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# 1. Load dữ liệu đã tiền xử lý
df = pd.read_csv("preprocessed_data.csv")

# 2. Chia tập dữ liệu
X_train, X_test, y_train, y_test = train_test_split(
    df["processed_comment"], df["label"], test_size=0.2, random_state=42
)

# 3. Xây dựng pipeline mô hình
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])

# 4. Huấn luyện mô hình
model.fit(X_train, y_train)

# 5. Đánh giá độ chính xác
accuracy = model.score(X_test, y_test)
print(f"✅ Accuracy: {accuracy:.2f}")

# 6. Lưu mô hình
joblib.dump(model, "sentiment_model.pkl")
print("✅ Mô hình đã được lưu thành công!")
