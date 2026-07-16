import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image

# Page setup
st.set_page_config(page_title="Crop & Weed Detection", page_icon="🌾", layout="centered")
st.title("🌾 Smart Agriculture: Crop & Weed Detection")
st.write("Upload images of a field to automatically detect crops and weeds using AI.")

# Load the trained YOLOv8 model
@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

# Image uploader - YAHAN MULTIPLE FILES ENABLE KAR DIYA HAI
uploaded_files = st.file_uploader("Choose images (JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files: # Agar user ne files upload ki hain
    for uploaded_file in uploaded_files:
        st.write(f"### ➡️ Results for: {uploaded_file.name}")
        
        # Convert uploaded image to OpenCV format
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        # YOLO prediction
        with st.spinner(f"🔍 Analyzing {uploaded_file.name}..."):
            results = model.predict(source=img_array, conf=0.5)
        
        # Plot results on the image
        res_plotted = results[0].plot()
        
        # Display the result - YAHAN WARNING FIX KAR DI HAI (use_container_width=True)
        st.image(res_plotted, caption=f"Detection Result: {uploaded_file.name}", use_container_width=True)
        
        st.divider() # Har photo ke baad ek line draw karega
        
    st.success("✅ All Detections Complete!")