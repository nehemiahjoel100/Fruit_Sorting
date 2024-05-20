import os
import cv2
import numpy as np
from tkinter import Tk, Label, Button, Entry, filedialog
from PIL import Image, ImageTk  # Install Pillow library for working with images
from tensorflow.keras.models import load_model

# Function to load and preprocess data
def preprocess_image(img_path):
    img = cv2.imread(img_path)

    # Check if the image is not empty
    if img is not None:
        img = cv2.resize(img, (64, 64))
        img = img / 255.0  # Normalize pixel values
        img = np.reshape(img, (1, 64, 64, 3))  # Reshape for prediction
        return img
    else:
        return None

# Function to display the predicted image
def display_predicted_image(img_path):
    img = Image.open(img_path)
    img.thumbnail((200, 200))  # Resize image for display

    img = ImageTk.PhotoImage(img)
    img_label.img = img
    img_label.config(image=img)

# Function to perform classification on a sample image
def classify_image():
    fruit_type = fruit_entry.get().lower()
    model_path = f'{fruit_type}_classification_model.h5'

    if not os.path.exists(model_path):
        result_label.config(text=f"Model file '{model_path}' not found. Please train the model first.")
        return

    loaded_model = load_model(model_path)
    sample_img_path = filedialog.askopenfilename(title="Select Sample Image")

    if sample_img_path:
        sample_img = preprocess_image(sample_img_path)

        if sample_img is not None:
            prediction = loaded_model.predict(sample_img)
            predicted_class = np.argmax(prediction)
            result_label.config(text=f'Predicted Class: {predicted_class} (Grade {"One" if predicted_class == 0 else "Two"})')
            display_predicted_image(sample_img_path)
        else:
            result_label.config(text="Invalid image. Please select a valid image.")
    else:
        result_label.config(text="No image selected. Please select a sample image.")

# GUI setup
root = Tk()
root.title("Fruit Image Classifier")

# Entry for user to input fruit type
fruit_label = Label(root, text="Enter fruit type (oranges, apples, or tomatoes):")
fruit_label.pack()

fruit_entry = Entry(root)
fruit_entry.pack()

# Button to initiate image classification
classify_button = Button(root, text="Classify Image", command=classify_image)
classify_button.pack()

# Label to display the classification result
result_label = Label(root, text="")
result_label.pack()

# Label to display the predicted image
img_label = Label(root)
img_label.pack()

# Run the GUI
root.mainloop()
