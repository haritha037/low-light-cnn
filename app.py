import streamlit as st
import os
from utils.model_utils import load_model, enhance_image
from utils.image_utils import load_image, save_image, display_image
import cv2
import numpy as np

# Load the model
model = load_model()

# Streamlit app
st.title("Low Light Image Enhancement")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded file
    input_image_path = "input_image.png"
    with open(input_image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display the original image
    st.image(load_image(input_image_path), caption="Original Image", use_column_width=True)

    # Enhance the image
    output_image_path = "enhanced_image.png"
    enhance_image(model, input_image_path, output_image_path)

    # Display the enhanced image
    st.image(load_image(output_image_path), caption="Enhanced Image", use_column_width=True)

    # Provide a download link for the enhanced image
    with open(output_image_path, "rb") as f:
        st.download_button("Download Enhanced Image", f, file_name="enhanced_image.png")