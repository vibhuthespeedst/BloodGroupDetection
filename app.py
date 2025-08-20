from flask import Flask, request, render_template
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load Model from JSON
with open("model.json", "r") as json_file:
    loaded_model_json = json_file.read()

model = model_from_json(loaded_model_json)
model.load_weights("model_weights.weights.h5")

# Load class indices
with open("class_indices.pkl", "rb") as f:
    class_indices = pickle.load(f)

# Reverse dictionary to get class labels
class_labels = {v: k for k, v in class_indices.items()}

# Ensure the uploads folder exists
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "❌ No image uploaded", 400

    img_file = request.files['image']
    if img_file.filename == '':
        return "❌ No selected file", 400

    img_path = os.path.join(UPLOAD_FOLDER, img_file.filename)
    img_file.save(img_path)

    try:
        # Load and preprocess image
        img = image.load_img(img_path, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Make prediction
        prediction = model.predict(img_array)
        class_index = np.argmax(prediction)
        blood_type = class_labels.get(class_index, "Unknown")

        return render_template('result.html', blood_type=blood_type)

    except Exception as e:
        return f"❌ Error processing image: {e}", 500

if __name__ == '__main__':
    app.run(debug=True)
