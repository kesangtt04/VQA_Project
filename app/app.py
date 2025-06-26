import os
import json
import subprocess
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# Configuration
TRAIN_DATA_FILE = "data/train_data.json"
TRAIN_SCRIPT = "models/train.py"    

def save_to_json(data, file_path):
    """Save data to a JSON file, appending if the file already exists."""
    try:
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
                if not isinstance(existing_data, list):
                    existing_data = []
        else:
            existing_data = []

        for item in data:
            if item not in existing_data:
                existing_data.append(item)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=4)
        print(f"✅ Data appended to {file_path}")
    except (json.JSONDecodeError, FileNotFoundError):
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"File error. Created new {file_path} and saved data.")

def save_train_data(image_paths, question, answer):
    """Save multiple image paths with the same question and answer to JSON."""
    data = [{"image_path": img_path, "question": question, "answer": answer} for img_path in image_paths]
    save_to_json(data, TRAIN_DATA_FILE)

def select_images():
    """Open file dialog to select multiple images and display the first one."""
    global selected_image_paths
    file_paths = filedialog.askopenfilenames()
    if not file_paths:
        return
    selected_image_paths = file_paths
    
    # Display first selected image as preview
    image = Image.open(selected_image_paths[0]).convert("RGB")
    image = image.resize((224, 224))
    img_tk = ImageTk.PhotoImage(image)
    image_label.config(image=img_tk)
    image_label.image = img_tk  # Keep reference to avoid garbage collection

def process_data():
    """Save data and initiate training if all inputs are provided."""
    if not selected_image_paths or not question_entry.get().strip() or not answer_entry.get().strip():
        result_label.config(text="⚠️ Please select images and enter both question and answer!")
        return
    
    save_train_data(selected_image_paths, question_entry.get(), answer_entry.get())
    result_label.config(text="✅ Data saved!")
    
    if os.path.exists(TRAIN_SCRIPT):
        subprocess.run(["python", TRAIN_SCRIPT], check=True)
        result_label.config(text="✅ Training completed!")
    else:
        result_label.config(text="⚠️ train.py not found!")

# GUI Setup
root = tk.Tk()
root.title("VQA System - Data Input")

selected_image_paths = []

btn_select = tk.Button(root, text="Select Images", command=select_images)
btn_select.pack()

image_label = tk.Label(root)
image_label.pack()

tk.Label(root, text="Enter Question:").pack()
question_entry = tk.Entry(root, width=50)
question_entry.pack()

tk.Label(root, text="Enter Answer:").pack()
answer_entry = tk.Entry(root, width=50)
answer_entry.pack()

btn_process = tk.Button(root, text="Save & Train", command=process_data)
btn_process.pack()

result_label = tk.Label(root, text="")
result_label.pack()

root.mainloop()