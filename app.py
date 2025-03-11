from flask import Flask, request, jsonify, render_template, send_from_directory
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Define and set the upload folder path
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the CSV file
csv_path = 'waste_classification.csv'
waste_data = pd.read_csv(csv_path)

# Load the Saved Model
model_path = 'models/waste_classification_model_3.h5'  # Ensure the model path is correct
model = tf.keras.models.load_model(model_path)

# Preprocess Image
def preprocess_image(img_path, target_size=(224, 224)):
    """ Load and preprocess an image for prediction. """
    img = image.load_img(img_path, target_size=target_size)  # Load and resize image
    img_array = image.img_to_array(img)  # Convert to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess for MobileNetV2
    return img_array

# Predict Image Category
def predict_image_category(img_path):
    """ Predict the waste category of an image. """
    img_array = preprocess_image(img_path)  # Preprocess image
    predictions = model.predict(img_array)  # Get model prediction
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    
    # Get the Item name instead of Category
    row = waste_data.iloc[predicted_class_index]  # Select row based on predicted index
    predicted_item = row['Item']  # Get Item name
    waste_category = row['Category']
    subcategory = row['Subcategory']
    
    confidence = np.max(predictions)  # Get confidence score
    
    return predicted_item, waste_category, subcategory, confidence

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """ Handle image upload and return prediction results. """
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filename = secure_filename(file.filename)  # Secure the filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        predicted_class, waste_category, subcategory, confidence = predict_image_category(file_path)

        result = {
            "predicted_class": predicted_class,  # Now returns the Item name
            "category": waste_category,
            "subcategory": subcategory,
            "probability": float(confidence),
            "image_url": f"/uploads/{filename}"  # Serve uploaded image
        }
        return jsonify(result)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """ Serve uploaded files """
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
