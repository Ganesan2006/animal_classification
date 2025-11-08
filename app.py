import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# Load the Keras model (update the filename/path accordingly)
model = tf.keras.models.load_model("animal_classification_model.keras", compile=False)

# Define the animal classes
classes = ['elephant', 'cheeta', 'wild boar']

# Image preprocessing function
def preprocess_image(image):
    img = image.resize((224, 224))
    img = np.array(img).astype('float32') / 255.0  # normalize to [0,1]
    if img.shape[2] == 4:  # remove alpha if present
        img = img[..., :3]
    img = np.expand_dims(img, axis=0)  # Add batch dim
    return img

# Prediction and drawing function
def predict_and_draw(image):
    input_img = preprocess_image(image)
    prediction = model.predict(input_img)

    # Debug info to check prediction output
    st.write("Prediction shape:", prediction.shape)
    st.write("Prediction output:", prediction)

    if prediction.size == 0:
        st.warning("Empty prediction output!")
        return image

    if prediction.shape[1] >= 7:
        class_probs = prediction[0][0:3]
        bbox = prediction[0][3:7]
    elif prediction.shape[1] >= 3:
        class_probs = prediction[0][0:3]
        bbox = None
    else:
        st.error("Unexpected prediction output shape.")
        return image

    class_id = int(np.argmax(class_probs))
    confidence = class_probs[class_id]

    img_cv = np.array(image.convert('RGB'))
    width, height = image.size

    if bbox is not None and confidence > 0.5:
        x_min = int(bbox[0] * width)
        y_min = int(bbox[1] * height)
        x_max = int(bbox[2] * width)
        y_max = int(bbox[3] * height)
        cv2.rectangle(img_cv, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    label = f"{classes[class_id]}: {confidence:.2f}"
    cv2.putText(img_cv, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return Image.fromarray(img_cv)

# Streamlit UI
st.title("Animal Detection with CNN")

# Capture image from webcam or upload file
img_file_buffer = st.camera_input("Capture Image or Upload")

if img_file_buffer:
    image = Image.open(img_file_buffer)
    output_img = predict_and_draw(image)
    st.image(output_img, caption="Detection Result", use_container_width=True)
else:
    st.info("Please capture an image or upload one to start detection.")


