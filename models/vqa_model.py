import os
import json
import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pretrain_model import predict_with_pretrain  # Thêm

# Configuration
TOKENIZER_FILE = "data/tokenizer.json"
ANSWER_TOKENIZER_FILE = "data/answer_tokenizer.json"
VQA_MODEL_PATH = "models/vqa_model.h5"
MAX_LENGTH = 20

# Load Model and Tokenizers
print("\U0001F504 Loading model...")
vqa_model = load_model(VQA_MODEL_PATH)
print("\u2705 Model ready!")

# Load question tokenizer
with open(TOKENIZER_FILE, "r", encoding="utf-8") as f:
    tokenizer_data = json.load(f)
question_tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
question_tokenizer.word_index = tokenizer_data["word_index"]

# Load answer tokenizer
with open(ANSWER_TOKENIZER_FILE, "r", encoding="utf-8") as f:
    answer_tokenizer_data = json.load(f)
answer_tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
answer_tokenizer.word_index = answer_tokenizer_data["word_index"]
answer_tokenizer.index_word = {v: k for k, v in answer_tokenizer.word_index.items()}

# Image Preprocessing
def preprocess_image(image_path, target_size=(100, 100)):
    """Preprocess a single image for prediction."""
    image = load_img(image_path, target_size=target_size)
    image = img_to_array(image) / 255.0
    return np.expand_dims(image, axis=0)

# Question Preprocessing
def preprocess_question(question):
    """Tokenize and pad a single question."""
    seq = question_tokenizer.texts_to_sequences([question.lower()])
    return pad_sequences(seq, maxlen=MAX_LENGTH, padding="post")

# Prediction
def predict_fruit(image_path, question):
    if not os.path.exists(image_path):
        return "\u26a0️ Image not found!", ""
    
    # Dự đoán với model tự train
    image = preprocess_image(image_path)
    question_seq = preprocess_question(question)
    prediction = vqa_model.predict([image, question_seq])[0]
    
    answer_ids = np.argmax(prediction, axis=-1)
    answer_tokens = []
    for idx in answer_ids:
        if idx == 0:
            continue
        token = answer_tokenizer.index_word.get(idx, "<OOV>")
        if token != "<OOV>":
            answer_tokens.append(token)
    custom_answer = " ".join(answer_tokens).strip()
    
    # Dự đoán với pretrained model
    pretrain_answer = predict_with_pretrain(image_path)
    
    return custom_answer, pretrain_answer
# GUI Functions
def select_image():
    """Open file dialog to select an image and display it."""
    global selected_image_path
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
    if file_path:
        selected_image_path = file_path
        img = Image.open(file_path).resize((200, 200))
        img = ImageTk.PhotoImage(img)
        image_label.config(image=img)
        image_label.image = img  # Keep reference to avoid garbage collection

def classify_image():
    global selected_image_path
    if not selected_image_path:
        result_label.config(text="⚠️ Please select an image!")
        return
    
    question = question_entry.get().strip()
    if not question:
        result_label.config(text="⚠️ Please enter a question!")
        return
    
    custom_answer, pretrain_answer = predict_fruit(selected_image_path, question)
    result_text = f"""
    \nModel: {custom_answer if custom_answer else 'No answer'}
    \nPretrained Model: {pretrain_answer}
    """
    result_label.config(text=result_text)
# GUI Setup
root = tk.Tk()
root.title("Fruit Recognition - VQA")

tk.Label(root, text="Select an image to identify").pack()
btn_select = tk.Button(root, text="chọn ảnh", command=select_image)
btn_select.pack()

image_label = tk.Label(root)
image_label.pack()

tk.Label(root, text="Nhập câu hỏi:").pack()
question_entry = tk.Entry(root, width=50)
question_entry.pack()

tk.Button(root, text="nhận diện", command=classify_image).pack()

result_label = tk.Label(root, text="kết quả:")
result_label.pack()

selected_image_path = ""

root.mainloop()