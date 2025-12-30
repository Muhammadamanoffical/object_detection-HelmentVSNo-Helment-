import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.title("Helmet vs No Helmet Detection")

model = YOLO("best.pt")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    results = model.predict(image)
    result_img = results[0].plot()

    st.image(result_img, caption="Detection Result")
