import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Animal Detection with CNN",
    page_icon="🦁",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ============================================================================
# MODEL LOADING
# ============================================================================

@st.cache_resource
def load_model():
    """Load the cleaned animal classification model."""
    model_path = "animal_classification_model.keras"
    
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found: {model_path}")
        st.info("Make sure 'animal_classification_model.keras' is in the same directory as app.py")
        st.stop()
    
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        st.sidebar.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

# Load the model
model = load_model()

# Animal classes (must match your training data order)
classes = ['elephant', 'cheeta', 'wild boar']

# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess the input image to match model expectations.
    
    Args:
        image: PIL Image object
        
    Returns:
        numpy array of shape (1, 224, 224, 3) normalized to [0, 1]
    """
    # Resize to model input size
    img = image.resize((224, 224))
    
    # Convert to numpy and normalize
    img_array = np.array(img).astype("float32") / 255.0
    
    # Handle grayscale images
    if img_array.ndim == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    
    # Remove alpha channel if present
    if img_array.shape[2] == 4:
        img_array = img_array[..., :3]
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# ============================================================================
# PREDICTION & VISUALIZATION
# ============================================================================

def predict_and_draw(image: Image.Image):
    """
    Predict the animal class and draw bounding box if available.
    
    Args:
        image: PIL Image object
        
    Returns:
        PIL Image with predictions and bounding box drawn
    """
    # Preprocess image
    input_img = preprocess_image(image)
    
    # Get prediction
    prediction = model.predict(input_img, verbose=0)
    
    # Debug info (hidden by default but useful for troubleshooting)
    with st.expander("🔍 Debug Info"):
        st.write("Prediction shape:", prediction.shape)
        st.write("Raw prediction:", prediction)
    
    # Validate prediction output
    if prediction.ndim != 2 or prediction.shape[0] == 0:
        st.error("❌ Unexpected prediction output shape.")
        return image
    
    num_features = prediction.shape[1]
    
    # Parse prediction output
    # Assume: [class_prob_0, class_prob_1, class_prob_2, x_min, y_min, x_max, y_max]
    if num_features >= 7:
        class_probs = prediction[0, :3]
        bbox = prediction[0, 3:7]
    elif num_features >= 3:
        class_probs = prediction[0, :3]
        bbox = None
    else:
        st.error("❌ Model output has fewer than 3 values; cannot decode.")
        return image
    
    # Get predicted class
    class_id = int(np.argmax(class_probs))
    confidence = float(class_probs[class_id])
    
    # Convert PIL image to OpenCV format
    img_cv = np.array(image.convert("RGB"))
    width, height = image.size
    
    # Draw bounding box if available and confidence > 0.5
    if bbox is not None and confidence > 0.5:
        x_min = int(bbox[0] * width)
        y_min = int(bbox[1] * height)
        x_max = int(bbox[2] * width)
        y_max = int(bbox[3] * height)
        
        # Draw rectangle
        cv2.rectangle(img_cv, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    
    # Draw class label and confidence
    label = f"{classes[class_id].upper()}: {confidence:.2%}"
    cv2.putText(
        img_cv,
        label,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )
    
    # Convert back to PIL
    return Image.fromarray(img_cv)

# ============================================================================
# STREAMLIT UI
# ============================================================================

st.title("🦁 Animal Detection with CNN")

st.markdown("""
This app uses a Convolutional Neural Network to classify animals from images.
- **Supported animals**: Elephant, Cheetah, Wild Boar
- **Input size**: 224 × 224 pixels
- **Model**: EfficientNet-based CNN

Capture an image using your webcam or upload a file to get started!
""")

# Sidebar info
with st.sidebar:
    st.header("📋 About")
    st.write(f"**Model Status**: ✅ Loaded")
    st.write(f"**Classes**: {', '.join([c.capitalize() for c in classes])}")
    st.write(f"**Input shape**: (224, 224, 3)")

# Main content
st.subheader("Capture or Upload Image")

img_file_buffer = st.camera_input("📷 Capture Image or Upload File")

if img_file_buffer is not None:
    # Load image
    image = Image.open(img_file_buffer)
    
    # Display original image
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="📸 Original Image", use_container_width=True)
    
    # Predict
    with st.spinner("🔄 Processing image..."):
        output_img = predict_and_draw(image)
    
    with col2:
        st.image(output_img, caption="🎯 Detection Result", use_container_width=True)
    
    st.success("✅ Prediction complete!")

else:
    st.info("👆 Please capture an image using your webcam or upload a file to start detection.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Built with Streamlit & TensorFlow 🚀"
    "</div>",
    unsafe_allow_html=True,
)

