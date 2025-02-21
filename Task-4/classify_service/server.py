import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, request, jsonify, send_from_directory
import io

class_labels = ['Rock', 'Paper', 'Scissors']

app = Flask(__name__)

# Load the retrained MobileNetV2 model
MODEL_PATH = "mobilenetv2_rps_20250221-120117_acc0.97.h5"  # Ensure this file exists
model = load_model(MODEL_PATH)

# Preprocessing function
def preprocess_image(img):
    img = img.resize((160, 160))  # Resize to MobileNetV2 input size
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img

# # Define a function to preprocess the image
# def preprocess_image(image_path):
#     img = image.load_img(image_path, target_size=(160, 160))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array /= 255.  # Normalize the image
#     return img_array

# Serve the HTML file
@app.route("/")
def serve_client():
    return send_from_directory(".", "client.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    img = image.load_img(io.BytesIO(file.read()), target_size=(224, 224))  # Load image

    # Preprocess and make prediction
    img_array = preprocess_image(img)
    predictions = model.predict(img_array)

    predicted_class = np.argmax(predictions[0])
    # print(f"Image: {image_path}, Predicted Class: {class_labels[predicted_class]}")

    return jsonify({"predictions": class_labels[predicted_class]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
