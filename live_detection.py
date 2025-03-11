import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load Model
model_path = 'models/waste_classification_model_3.h5'
model = tf.keras.models.load_model(model_path)

# Load CSV
csv_path = 'waste_classification.csv'
waste_data = pd.read_csv(csv_path)

# Preprocess Image
def preprocess_frame(frame, target_size=(224, 224)):
    """Resize, preprocess, and normalize the frame."""
    frame_resized = cv2.resize(frame, target_size)
    img_array = image.img_to_array(frame_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Predict Image Category
def predict_frame_category(frame):
    """ Predict waste category from a single frame. """
    img_array = preprocess_frame(frame)
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]

    row = waste_data.iloc[predicted_class_index]
    predicted_item = row['Item']
    waste_category = row['Category']
    subcategory = row['Subcategory']
    confidence = np.max(predictions)

    return predicted_item, waste_category, subcategory, confidence

# Open Webcam
cap = cv2.VideoCapture(0)  # 0 for default camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Predict Category
    predicted_item, waste_category, subcategory, confidence = predict_frame_category(frame)

    # Display Result on Frame
    label = f"{predicted_item} ({waste_category}) - {confidence:.2f}"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show Frame
    cv2.imshow("Waste Classification - Live Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
