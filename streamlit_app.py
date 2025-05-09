import streamlit as st
import easyocr
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

st.set_page_config(page_title="🧾 EasyOCR Image Text Extractor", layout="centered")
st.title("🧾 Image OCR with EasyOCR")

# Upload image file
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

def process_image(image_path, gpu=False):
    reader = easyocr.Reader(['en'], gpu=gpu)
    result = reader.readtext(image_path)

    text_lines = [detection[1] for detection in result]
    saved_text = " ".join(text_lines)

    img = cv2.imread(image_path)
    for detection in result:
        top_left = tuple([int(val) for val in detection[0][0]])
        bottom_right = tuple([int(val) for val in detection[0][2]])
        text = detection[1]
        font = cv2.FONT_HERSHEY_SIMPLEX
        img = cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
        img = cv2.putText(img, text, top_left, font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return img, saved_text

if uploaded_file:
    st.success("✅ Image uploaded successfully!")

    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    with st.spinner("🔍 Running OCR..."):
        # Process image
        img_result, extracted_text = process_image(temp_path)

    st.success("🎉 OCR complete!")

    # Show image with bounding boxes
    st.image(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB), caption="Detected Text", use_column_width=True)

    # Show extracted text
    st.subheader("📄 Extracted Text")
    st.text_area("Detected Text", extracted_text, height=200)

    # Option to download text
    st.download_button("Download Extracted Text", extracted_text, file_name="extracted_text.txt")

    # Clean up temporary file
    os.remove(temp_path)

