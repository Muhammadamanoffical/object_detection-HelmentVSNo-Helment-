import numpy as np
import cv2
from app import predict, model

def test_predict():
    print("Testing Gradio app logic...")
    
    if model is None:
        print("FAIL: Model failed to load.")
        return

    # Create a dummy image (black square)
    dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
    
    # Run prediction
    print("Running prediction on dummy image...")
    res_image, counts, df = predict(dummy_image, 0.25, 0.45)
    
    if res_image is not None:
        print("SUCCESS: Prediction returned an image.")
        print(f"Output Image Shape: {res_image.shape}")
    else:
        print("FAIL: Prediction returned None for image.")
        
    if counts is not None:
        print("SUCCESS: Counts returned.")
        print(counts)
    else:
        print("FAIL: Counts is None.")

    if df is not None:
        print("SUCCESS: Detailed DataFrame returned.")
        print(df)
    else:
        print("FAIL: DataFrame is None.")

if __name__ == "__main__":
    test_predict()
