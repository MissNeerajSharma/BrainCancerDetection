from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import traceback
from tensorflow.keras.preprocessing.image import load_img


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
import tensorflow as tf

app = Flask(__name__)

# Load the trained model
model = load_model("model/model123.h5")

# Class labels - update these as per your model's classes
class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']

# Folder for uploaded images
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def predict_image(image_path):
    """
    Load image, preprocess, predict tumor type and confidence.
    """
    try:
        image_size = 128  # Use same size as training images
        img = load_img(image_path, target_size=(image_size, image_size))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        predicted_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_idx]

        label = class_labels[predicted_idx]
        if label == 'notumor':
            result = "No Tumor Detected"
        else:
            result = f" Yes Person has Tumor, Class : {label}"

        return result, confidence


    except Exception as e:

        print("Error in prediction:")

        traceback.print_exc()  # This will print full error trace

        return "Prediction Error", 0

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', result="No file part", confidence=None, file_path=None)

        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', result="No selected file", confidence=None, file_path=None)

        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            result, confidence = predict_image(filepath)


            print("Model loaded:", model)
            print(model.summary())
            confidence_pct = f"{confidence * 100:.2f}%"

            return render_template('index.html', result=result, confidence=confidence_pct, file_path='/uploads/' + filename)

    # For GET requests
    return render_template('index.html', result=None, confidence=None, file_path=None)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
