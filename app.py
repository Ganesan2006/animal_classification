import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import os
from urllib.request import urlopen
from io import BytesIO
import requests

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Animal Detection with CNN",
    page_icon="🦁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# TFLITE MODEL LOADING
# ============================================================================

@st.cache_resource
def load_tflite_model():
    """Load the quantized TensorFlow Lite model."""
    model_path = "animal_classification_model.tflite"
    
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found: {model_path}")
        st.info("Make sure 'animal_classification_model.tflite' is in the same directory as app.py")
        st.stop()
    
    try:
        # Load the TFLite model and allocate tensors
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        st.sidebar.success("✅ TFLite Model loaded successfully!")
        return interpreter
    except Exception as e:
        st.error(f"❌ Error loading TFLite model: {e}")
        st.stop()

# Load the TFLite interpreter
interpreter = load_tflite_model()

# Animal classes (must match your training data order)
classes = ['elephant', 'cheeta', 'wild boar']

# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess the input image to match TFLite model expectations.
    
    Args:
        image: PIL Image object
        
    Returns:
        numpy array of shape (1, 224, 224, 3) in correct format
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
    
    return img_array.astype("float32")

# ============================================================================
# TFLITE INFERENCE
# ============================================================================

def run_tflite_inference(input_data: np.ndarray) -> np.ndarray:
    """
    Run inference using the TFLite model.
    
    Args:
        input_data: Preprocessed image array
        
    Returns:
        Model predictions
    """
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Check if input is quantized
    input_dtype = input_details[0]['dtype']
    
    # Handle quantized input (int8)
    if input_dtype == np.int8:
        input_scale, input_zero_point = input_details[0]['quantization']
        input_data = (input_data / input_scale + input_zero_point).astype(np.int8)
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Run inference
    interpreter.invoke()
    
    # Get output
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Handle quantized output (int8)
    if output_details[0]['dtype'] == np.int8:
        output_scale, output_zero_point = output_details[0]['quantization']
        output_data = output_data.astype(np.float32)
        output_data = (output_data - output_zero_point) * output_scale
    
    return output_data

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
    
    # Get prediction using TFLite
    prediction = run_tflite_inference(input_img)
    
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
        
        # Ensure coordinates are within image bounds
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(width, x_max)
        y_max = min(height, y_max)
        
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
# IMAGE LOADING HELPERS
# ============================================================================

def load_image_from_url(url: str) -> Image.Image:
    """Load image from URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        return img
    except Exception as e:
        st.error(f"❌ Error loading image from URL: {e}")
        return None

def load_image_from_file(uploaded_file) -> Image.Image:
    """Load image from uploaded file."""
    try:
        img = Image.open(uploaded_file)
        return img
    except Exception as e:
        st.error(f"❌ Error loading image: {e}")
        return None

# ============================================================================
# STREAMLIT UI
# ============================================================================

st.title("🦁 Animal Detection with TFLite")

st.markdown("""
This app uses a quantized TensorFlow Lite model to classify animals from images.
- **Model Type**: TensorFlow Lite (Quantized)
- **Supported animals**: Elephant, Cheetah, Wild Boar
- **Input size**: 224 × 224 pixels
- **Architecture**: MobileNetV2-based CNN

Choose your preferred input method below!
""")

# Sidebar info
with st.sidebar:
    st.header("📋 About")
    st.write(f"**Model Type**: ✅ TFLite (Quantized)")
    st.write(f"**Classes**: {', '.join([c.capitalize() for c in classes])}")
    st.write(f"**Input shape**: (224, 224, 3)")
    st.write(f"**Model size**: Optimized for mobile/edge")

# ============================================================================
# INPUT METHOD SELECTOR
# ============================================================================

st.subheader("📸 Choose Input Method")

# Create tabs for different input methods
tab1, tab2, tab3, tab4 = st.tabs([
    "📷 Webcam Capture",
    "📁 Upload File",
    "🔗 Image URL",
    "💻 Paste Clipboard"
])

image = None
source_name = None

# ============================================================================
# TAB 1: WEBCAM CAPTURE
# ============================================================================

with tab1:
    st.write("Capture an image directly from your webcam:")
    img_file_buffer = st.camera_input("Capture Image from Webcam")
    
    if img_file_buffer is not None:
        image = load_image_from_file(img_file_buffer)
        source_name = "Webcam Capture"

# ============================================================================
# TAB 2: FILE UPLOAD
# ============================================================================

with tab2:
    st.write("Upload image file(s) from your device:")
    uploaded_files = st.file_uploader(
        "Choose image file(s)",
        type=["jpg", "jpeg", "png", "bmp", "gif"],
        accept_multiple_files=False
    )
    
    if uploaded_files is not None:
        image = load_image_from_file(uploaded_files)
        source_name = uploaded_files.name

# ============================================================================
# TAB 3: IMAGE URL
# ============================================================================

with tab3:
    st.write("Paste the URL of an image:")
    url_input = st.text_input(
        "Enter image URL",
        placeholder="https://example.com/image.jpg"
    )
    
    if url_input and st.button("🔗 Load from URL", key="url_button"):
        with st.spinner("Loading image from URL..."):
            image = load_image_from_url(url_input)
            source_name = "URL: " + url_input if image else None

# ============================================================================
# TAB 4: CLIPBOARD PASTE
# ============================================================================

with tab4:
    st.write("Screenshot and paste directly (Windows/Mac):")
    st.info("💡 **How to use:**\n1. Take a screenshot (Ctrl+PrintScreen or Cmd+Shift+4)\n2. Copy to clipboard\n3. Right-click in the text area below and paste")
    
    # Note: Direct clipboard paste from browser is limited
    # This is a workaround using text input
    st.warning("⚠️ Note: Browser security restricts direct clipboard access. Try these alternatives:\n"
               "- Use the **Upload File** tab to select from your downloads\n"
               "- Use the **Image URL** tab to link to online images\n"
               "- Use the **Webcam** tab for live capture")

# ============================================================================
# DISPLAY AND PREDICT
# ============================================================================

if image is not None:
    st.success(f"✅ Image loaded from: {source_name}")
    
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
    
    # Display detailed results
    st.subheader("📊 Prediction Details")
    
    # Re-run prediction to get confidence scores
    input_img = preprocess_image(image)
    prediction = run_tflite_inference(input_img)
    
    if prediction.shape[1] >= 3:
        class_probs = prediction[0, :3]
        
        # Create a DataFrame for better visualization
        import pandas as pd
        results_df = pd.DataFrame({
            'Animal': [c.capitalize() for c in classes],
            'Confidence': [f"{p:.2%}" for p in class_probs]
        })
        
        st.dataframe(results_df, use_container_width=True)
        
        # Show top prediction
        top_class = classes[int(np.argmax(class_probs))]
        top_confidence = class_probs[int(np.argmax(class_probs))]
        st.metric("Top Prediction", top_class.capitalize(), f"{top_confidence:.2%}")

else:
    st.info("👆 Please choose an input method above and provide an image to start detection.")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("🦁 **Elephant**\nLarge herbivore")
with col2:
    st.markdown("🐆 **Cheetah**\nFastest land animal")
with col3:
    st.markdown("🐗 **Wild Boar**\nWild pig species")

st.markdown(
    "<div style='text-align: center; color: gray; margin-top: 20px;'>"
    "Built with Streamlit & TensorFlow Lite 🚀"
    "</div>",
    unsafe_allow_html=True,
)
