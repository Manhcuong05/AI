import pandas as pd
import re
from underthesea import word_tokenize

# 1. Load dữ liệu
df = pd.read_csv("comments_dataset.csv")

# 2. Danh sách stopwords (có thể mở rộng thêm)
stopwords = set(["là", "của", "và", "đã", "cũng", "này", "được", "bị", "với", "cho", "có"])

# 3. Hàm tiền xử lý văn bản
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Loại bỏ dấu câu
    text = word_tokenize(text, format="text")  # Tách từ tiếng Việt
    text = " ".join([word for word in text.split() if word not in stopwords])  # Loại bỏ stopwords
    return text

df["processed_comment"] = df["comment"].apply(preprocess_text)

# 4. Lưu dữ liệu đã tiền xử lý
df.to_csv("preprocessed_data.csv", index=False, encoding="utf-8")

print("✅ File preprocessed_data.csv đã được tạo thành công!")
