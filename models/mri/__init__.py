import cv2
import tensorflow as tf
import numpy as np

categories = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
model = tf.keras.models.load_model("models/mri/mri.h5")


def preprocess_image(image_path, img_size=(128, 128)):
    img_array = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    resized_array = cv2.resize(img_array, img_size)
    normalized_array = resized_array / 255.0  # Normalize the image
    reshaped_array = normalized_array.reshape(1, img_size[0], img_size[1], 1)
    return reshaped_array


def predict_group(image_path, model, categories):
    preprocessed_image = preprocess_image(image_path)
    prediction = model.predict(preprocessed_image)
    predicted_class = np.argmax(prediction)
    print(f"Predicted class: {categories[predicted_class]}")
    return categories[predicted_class]


# Example usage
def run(image_path: str) -> str:

    predicted_group = predict_group(image_path, model, categories)
    return f"Result of Analyzing the MRI image By ML Model: {predicted_group}"
