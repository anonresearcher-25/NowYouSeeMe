import os
import cv2
import numpy as np
import tensorflow as tf


def initialize_landmark_detector():
    """Initializes the MediaPipe Face Mesh landmark detector."""
    current_dir = os.path.dirname(os.path.abspath(__file__))  # directory of utils.py
    landmarker_path = os.path.join(current_dir, "models", "MediaPipeFaceLandmarkDetector.tflite")

    interpreter = tf.lite.Interpreter(model_path=landmarker_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("✅ MediaPipe Face Mesh initialized.")
    print("📤 Model has", len(output_details), "output tensor(s):")
    for i, detail in enumerate(output_details):
        print(f"Output {i}: Name = {detail['name']}, Shape = {detail['shape']}")

    return interpreter, input_details, output_details


def get_landmarks_interpretor(face_roi_rgb, interpreter, input_details, output_details):
    """Gets landmarks for a specific face ROI and prints all outputs."""
    face_crop = cv2.resize(face_roi_rgb, (192, 192))  # Adjust size based on model
    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    input_tensor = face_rgb.astype(np.float32) / 255.0  # Normalize to [0, 1]
    input_tensor = np.expand_dims(input_tensor, axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()

    print("\n🔍 Model output tensors:")
    for i, output in enumerate(output_details):
        tensor_data = interpreter.get_tensor(output['index'])
        print(f"Output {i}: shape = {tensor_data.shape}")
        print(f"Sample values: {tensor_data.flatten()[:5]}\n")  # First 5 values

    # Assume landmarks are in output index 1 (as per your setup)
    output_data = interpreter.get_tensor(output_details[1]['index'])  # Shape: [1, 1404]
    landmarks = output_data[0].reshape(-1, 3)  # shape: (468, 3)

    return landmarks


def main():
    # === Initialize model ===
    interpreter, input_details, output_details = initialize_landmark_detector()

    # === Load test image ===
    image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_face.jpg")  # Replace with your test image
    if not os.path.exists(image_path):
        print("❌ Test image not found:", image_path)
        return

    image = cv2.imread(image_path)
    if image is None:
        print("❌ Failed to load image.")
        return

    # === Run landmark detection ===
    landmarks = get_landmarks_interpretor(image, interpreter, input_details, output_details)
    print("✅ Landmarks extracted. First 5 coordinates:\n", landmarks[:5])


if __name__ == "__main__":
    main()
