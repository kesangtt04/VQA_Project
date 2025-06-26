import os
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LSTM, Embedding, Input, Concatenate, Dropout, RepeatVector
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Configuration
DATA_DIR = "data"
MODEL_DIR = "models"
TOKENIZER_FILE = os.path.join(DATA_DIR, "tokenizer.json")
ANSWER_TOKENIZER_FILE = os.path.join(DATA_DIR, "answer_tokenizer.json")
VQA_MODEL_PATH = os.path.join(MODEL_DIR, "vqa_model.h5")
MAX_LENGTH = 20

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Data Preprocessing
def load_data():
    """Load training data from JSON file."""
    file_path = os.path.join(DATA_DIR, "train_data.json")
    if not os.path.exists(file_path):
        raise FileNotFoundError("Error: train_data.json not found")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def preprocess_images(data, target_size=(100, 100)):
    """Preprocess images to a uniform size and normalize."""
    images = []
    for item in data:
        image_path = item["image_path"]
        if os.path.exists(image_path):
            image = load_img(image_path, target_size=target_size)
            image = img_to_array(image) / 255.0
            images.append(image)
    return np.array(images)

def preprocess_texts(texts, max_length=MAX_LENGTH):
    """Tokenize and pad question texts."""
    tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    with open(TOKENIZER_FILE, "w", encoding="utf-8") as f:
        json.dump({"word_index": tokenizer.word_index}, f, ensure_ascii=False, indent=4)
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=max_length, padding="post"), tokenizer

def preprocess_answers(answers, max_length=MAX_LENGTH):
    """Tokenize and pad answer texts."""
    answer_tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
    answer_tokenizer.fit_on_texts(answers)
    with open(ANSWER_TOKENIZER_FILE, "w", encoding="utf-8") as f:
        json.dump({"word_index": answer_tokenizer.word_index}, f, ensure_ascii=False, indent=4)
    sequences = answer_tokenizer.texts_to_sequences(answers)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding="post")
    return padded_sequences, answer_tokenizer

# Model Building
def build_vqa_model(vocab_size, max_length=MAX_LENGTH):
    """Build the VQA model with CNN for images and LSTM for questions and answers."""
    # CNN for image features    
    image_input = Input(shape=(100, 100, 3))
    x = Conv2D(32, (3, 3), activation='relu', padding="same")(image_input)
    x = MaxPooling2D((2, 2))(x  )
    x = Conv2D(64, (3, 3), activation='relu', padding="same")(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    image_features = Dense(128, activation='relu')(x)   

    # LSTM for question features
    question_input = Input(shape=(max_length,))
    x = Embedding(input_dim=vocab_size, output_dim=128)(question_input)
    question_features = LSTM(128, return_sequences=False)(x)

    # Combine features
    combined = Concatenate()([image_features, question_features])
    combined = Dense(128, activation="relu")(combined)
    combined = Dropout(0.2)(combined)
    
    # LSTM for answer generation
    combined = RepeatVector(max_length)(combined)
    answer_output = LSTM(128, return_sequences=True)(combined)
    answer_output = Dense(vocab_size, activation="softmax")(answer_output)

    model = Model(inputs=[image_input, question_input], outputs=answer_output)
    return model

# Training
def train():
    """Train the VQA model and save it."""
    data = load_data()
    images = preprocess_images(data)
    questions, question_tokenizer = preprocess_texts([item["question"] for item in data])
    answers, answer_tokenizer = preprocess_answers([item["answer"] for item in data])

    X_train_images, X_val_images, X_train_questions, X_val_questions, y_train, y_val = train_test_split(
        images, questions, answers, test_size=0.2, random_state=42
    )

    vqa_model = build_vqa_model(vocab_size=len(answer_tokenizer.word_index) + 1)
    vqa_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    vqa_model.fit(
        [X_train_images, X_train_questions], y_train,
        validation_data=([X_val_images, X_val_questions], y_val),
        epochs=10, batch_size=32
    )
    vqa_model.save(VQA_MODEL_PATH)
    print(f"✅ Model saved at {VQA_MODEL_PATH}")

if __name__ == "__main__":
    train()