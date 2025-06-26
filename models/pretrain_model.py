# pretrain_model.py
import numpy as np
import tensorflow as tf
import json
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load pretrained model
pretrained_model = MobileNetV2(weights="imagenet")

# Load class mapping from pretrained.json
with open("data/pretrained.json", "r", encoding="utf-8") as f:
    MAPPING = json.load(f)


def predict_with_pretrain(image_path):
    """Predict fruit using pretrained MobileNetV2 model."""
    try:
        # Tiền xử lý ảnh
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = preprocess_input(np.expand_dims(img_array, axis=0))  # Chuẩn hóa theo MobileNetV2
        
        # Dự đoán
        predictions = pretrained_model.predict(img_array)
        predicted_class = str(np.argmax(predictions))  # Lấy chỉ số dự đoán cao nhất
        
        # Ánh xạ kết quả từ MAPPING
        if predicted_class in MAPPING:
            class_id, class_name = MAPPING[predicted_class]
            return f"{class_name}"
        else:
            return f"Unknown (Class ID: {predicted_class})"
    
    except Exception as e:
        return f"Lỗi: {str(e)}"
