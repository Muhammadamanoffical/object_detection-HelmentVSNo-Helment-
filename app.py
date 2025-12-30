import os
import torch
from ultralytics.nn.tasks import DetectionModel
torch.serialization.add_safe_globals([DetectionModel])

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Optional: hide OpenCV errors
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

st.title("Helmet vs No Helmet Detection")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    results = model.predict(
        source=np.array(image),
        save=False,
        show=False,
        verbose=False
    )

    result_img = results[0].plot()
    st.image(result_img, caption="Detection Result", use_container_width=True)
