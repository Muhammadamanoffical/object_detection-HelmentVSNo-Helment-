import gradio as gr
import cv2
import PIL.Image
import numpy as np
from ultralytics import YOLO
import pandas as pd

# --- Model Loading ---
MODEL_PATH = "best (3).pt"
try:
    model = YOLO(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def predict(image, conf_threshold, iou_threshold):
    """
    Runs YOLO inference on the input image.
    Args:
        image: Input image (numpy array or PIL Image).
        conf_threshold: Confidence threshold for detection.
        iou_threshold: IoU threshold for NMS.
    Returns:
        Annotated image (numpy array), Class counts (dict/str), Detailed Data (DataFrame)
    """
    if model is None:
        return None, "Model not loaded.", None

    try:
        # Run inference
        results = model.predict(image, conf=conf_threshold, iou=iou_threshold)
        result = results[0]
        
        # Plot results
        res_plotted = result.plot()
        res_image = res_plotted[..., ::-1] # Convert BGR to RGB if needed, specifically for Gradio image output which usually expects RGB
        
        # Count classes
        class_counts = {}
        box_data = []
        
        for box in result.boxes:
            cls = int(box.cls[0])
            cls_name = model.names[cls]
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
            
            box_data.append({
                "Class": cls_name,
                "Confidence": float(box.conf[0]),
                "Coordinates": [round(x, 1) for x in box.xyxy[0].tolist()]
            })
            
        # Format class counts for display
        counts_summary = pd.DataFrame(list(class_counts.items()), columns=['Class', 'Count'])
        
        # Detailed data
        df = pd.DataFrame(box_data)
        
        return res_image, counts_summary, df

    except Exception as e:
        return None, f"Error: {e}", None

# --- Gradio UI ---
def create_interface():
    with gr.Blocks() as demo:
        gr.Markdown(
            """
            # 👷 Safety Helmet Detection Pro
            Upload an image to detect safety compliance (Head, Helmet, Person).
            """
        )
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Original Image", type="numpy")
                conf_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.25, step=0.05, label="Confidence Threshold")
                iou_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.45, step=0.05, label="IoU Threshold")
                run_btn = gr.Button("🔍 Run Detection", variant="primary")
            
            with gr.Column():
                output_image = gr.Image(label="Detected Output")
                gr.Markdown("### 📊 Detection Statistics")
                output_counts = gr.Dataframe(label="Class Counts")
                output_details = gr.Dataframe(label="Detailed Detection Data")

        run_btn.click(
            fn=predict,
            inputs=[input_image, conf_slider, iou_slider],
            outputs=[output_image, output_counts, output_details]
        )
        
        gr.Markdown("---")
        gr.Markdown("Model: standard YOLOv8n (Custom Trained) | Classes: Head, Helmet, Person")

    return demo

if __name__ == "__main__":
    demo = create_interface()
   
    demo.launch(theme=gr.themes.Soft()) # This relies on launch accepting theme?
    

