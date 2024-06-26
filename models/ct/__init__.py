import tensorflow as tf
import cv2
import numpy as np
from io import BytesIO

# Load the trained model
model = tf.keras.models.load_model("models/ct/ct.h5")


# Function to preprocess the image
def preprocess_image(image_content, img_size=(128, 128)):
    file_bytes = np.asarray(bytearray(image_content.read()), dtype=np.uint8)
    img_array = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    resized_array = cv2.resize(img_array, img_size)
    normalized_array = resized_array / 255.0  # Normalize the image
    reshaped_array = normalized_array.reshape(1, img_size[0], img_size[1], 1)
    return reshaped_array


# Function to predict the group
def predict_group(image_content, model, categories):
    preprocessed_image = preprocess_image(image_content)
    prediction = model.predict(preprocessed_image)
    predicted_class = np.argmax(prediction)
    return categories[predicted_class]


# Example usage
categories = ["tumor", "cancer", "aneurysm"]


def run(image_content: BytesIO) -> str:
    predicted_group = predict_group(image_content, model, categories)
    return f"Result of Analyzing the CT image By ML Model: {predicted_group}"
