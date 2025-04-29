import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt

# Load the trained model
model = load_model('best_model_mobilenetv2.h5')  # or 'final_model_mobilenetv2.h5'

# Class labels in the same order as during training
class_labels = ['Open Sewers', 'Others', 'Potholes']

# Folder path for test images
test_folder = 'TestImages'
IMG_SIZE = 224

# Predict for each image
for img_name in os.listdir(test_folder):
    img_path = os.path.join(test_folder, img_name)

    try:
        # Load and preprocess image
        img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Predict
        prediction = model.predict(img_array)
        predicted_index = np.argmax(prediction)
        predicted_class = class_labels[predicted_index]
        confidence = prediction[0][predicted_index] * 100

        # Output result with confidence
        print(f"{img_name} → {predicted_class} ({confidence:.2f}% confidence)")

        # Optional: Show image with title
        plt.imshow(img)
        plt.title(f"{predicted_class} ({confidence:.2f}%)")
        plt.axis('off')
        plt.show()

    except Exception as e:
        print(f"❌ Could not process {img_name}: {e}")
