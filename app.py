import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import os

st.set_page_config(page_title="Animal Detection with CNN")

# ---------------- Model loading ---------------- #

@st.cache_resource
def load_model():
    model_path = "animal_classification_model.keras"
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.stop()
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model = load_model()

# Classes (order must match model training)
classes = ['elephant', 'cheeta', 'wild boar']

# ---------------- Preprocessing ---------------- #

def preprocess_image(image: Image.Image) -> np.ndarray:
    img = image.resize((224, 224))
    img = np.array(img).astype("float32") / 255.0
    if img.ndim == 2:          # grayscale -> RGB
        img = np.stack([img] * 3, axis=-1)
    if img.shape[2] == 4:      # drop alpha
        img = img[..., :3]
    img = np.expand_dims(img, axis=0)
    return img

# ---------------- Prediction + drawing --------- #

def predict_and_draw(image: Image.Image) -> Image.Image:
    input_img = preprocess_image(image)
    prediction = model.predict(input_img)

    st.write("Prediction shape:", prediction.shape)
    st.write("Prediction output:", prediction)

    # Expect 2D: (1, N)
    if prediction.ndim != 2 or prediction.shape[0] == 0:
        st.error("Unexpected prediction output shape.")
        return image

    num_feats = prediction.shape[1]

    # Assume: [p_elephant, p_cheeta, p_wild_boar, x_min, y_min, x_max, y_max]
    if num_feats >= 7:
        class_probs = prediction[0, :3]
        bbox = prediction[0, 3:7]
    elif num_feats >= 3:
        class_probs = prediction[0, :3]
        bbox = None
    else:
        st.error("Model output has fewer than 3 values; cannot decode.")
        return image

    class_id = int(np.argmax(class_probs))
    confidence = float(class_probs[class_id])

    img_cv = np.array(image.convert("RGB"))
    width, height = image.size

    if bbox is not None and confidence > 0.5:
        x_min = int(bbox[0] * width)
        y_min = int(bbox[1] * height)
        x_max = int(bbox[2] * width)
        y_max = int(bbox[3] * height)
        cv2.rectangle(img_cv, (x_min, y_min), (x_max, y_max),
                      (0, 255, 0), 2)

    label = f"{classes[class_id]}: {confidence:.2f}"
    cv2.putText(
        img_cv,
        label,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    return Image.fromarray(img_cv)

# ---------------- Streamlit UI ----------------- #

st.title("Animal Detection with CNN")

st.write(
    "Capture an image using your webcam or upload a file. "
    "The model will predict whether it is an elephant, cheeta, or wild boar, "
    "and draw a bounding box if available."
)

img_file_buffer = st.camera_input("Capture Image or Upload")

if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
    output_img = predict_and_draw(image)
    st.image(output_img, caption="Detection Result", use_container_width=True)
else:
    st.info("Please capture an image or upload one to start detection.")
