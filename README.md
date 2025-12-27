

# Animal Classification (TFLite) – Streamlit Web App

A lightweight **Streamlit** web application that classifies animals using a **quantized TensorFlow Lite (`.tflite`)** model.  
The app supports multiple image input methods (webcam capture, file upload, and image URL) and displays prediction confidence.

## Features

- Quantized **TFLite model inference** (fast and deployment-friendly)
- Input methods:
  - Webcam capture
  - File upload (drag & drop)
  - Image URL paste
- Predicts 3 classes:
  - Elephant
  - Cheeta (Cheetah)
  - Wild boar
- Shows:
  - Top prediction
  - Confidence for all classes
- Works well on Streamlit Cloud (no heavy Keras `.keras` loading issues)

## Tech Stack

- Python
- Streamlit
- TensorFlow Lite (via `tensorflow`)
- OpenCV (headless)
- Pillow
- NumPy
- Requests (for URL images)
- Pandas (for results table)

## Project Structure

```
animal_classification/
├── app.py
├── requirements.txt
├── animal_classification_model.tflite
└── README.md
```

## Installation (Local)

1. Clone the repository:
```
git clone https://github.com/Ganesan2006/animal_classification.git
cd animal_classification
```

2. Create and activate a virtual environment:
```
python -m venv venv
# Linux/Mac
source venv/bin/activate
# Windows
venv\Scripts\activate
```

3. Install dependencies:
```
pip install -r requirements.txt
```

4. Run the app:
```
streamlit run app.py
```

Open the URL shown in terminal (usually `http://localhost:8501`).

## Model Input Details

- Input image size: `(224, 224)`
- Input channels: `3 (RGB)`
- Normalization: `float32` scaled to `[0, 1]`
- Quantized models (`int8`) are automatically handled using input/output scale + zero-point inside the app.

## How to Use

1. Open the app in your browser.
2. Choose an input method:
   - **Webcam Capture** → capture image from camera
   - **Upload File** → upload `.jpg/.jpeg/.png/...`
   - **Image URL** → paste a direct image URL
3. The app displays:
   - Original image
   - Prediction result and confidence

## Deployment (Streamlit Cloud)

1. Push your code to GitHub (make sure these files exist in the repo root):
   - `app.py`
   - `requirements.txt`
   - `animal_classification_model.tflite`

2. Go to Streamlit Community Cloud:
   - https://share.streamlit.io/

3. Select your repository and deploy.

### Notes for Streamlit Cloud

- Streamlit Cloud may use newer Python versions, so keep dependencies compatible.
- TFLite inference is typically more reliable in cloud deployments than `.keras` model loading.

## requirements.txt Example

Your `requirements.txt` should look like this (or similar):

```
streamlit==1.52.1
tensorflow==2.20.0
opencv-python-headless==4.10.0.84
pillow==11.1.0
requests
pandas
```
