import gradio as gr
from ultralytics import YOLO
import ultralytics
import torch
import cv2
import numpy as np
import os
import gdown

# -----------------------------
# Download model if not exists
# -----------------------------
if not os.path.exists("best.pt"):
    url = "YOUR_GOOGLE_DRIVE_DIRECT_LINK"  # Replace with your file link
    gdown.download(url, "best.pt", quiet=False)

# -----------------------------
# Load YOLO Model safely
# -----------------------------
@torch.no_grad()
def load_model():
    with torch.serialization.safe_globals([
        ultralytics.nn.tasks.DetectionModel,
        torch.nn.modules.container.Sequential,
        torch.nn.modules.module.Module
    ]):
        model = YOLO("best.pt")  # Your fine-tuned model
    return model

model = load_model()

# -----------------------------
# Object Detection Function
# -----------------------------
def object_detection(image):
    """
    image: numpy array
    returns: numpy array with detections
    """
    results = model.predict(image, imgsz=640)  # Run prediction
    detected_image = results[0].plot()         # Draw bounding boxes

    # Convert BGR to RGB for Gradio display
    if detected_image.shape[2] == 3:
        detected_image = cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB)

    return detected_image

# -----------------------------
# Gradio Interface
# -----------------------------
app = gr.Interface(
    fn=object_detection,
    inputs=gr.Image(type="numpy", label="Upload Image"),
    outputs=gr.Image(type="numpy", label="Detected Objects"),
    title="🧠 Helmet Detection System",
    description="Upload an image to detect objects (Helmet / No-Helmet)",
    allow_flagging="never"
)

# Launch Gradio app
app.launch()
    
