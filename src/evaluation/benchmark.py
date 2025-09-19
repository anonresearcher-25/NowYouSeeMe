# Used to benchmark face detection models: MediaPipe, SCRFD, and YuNet.
import cv2
import mediapipe as mp
import numpy as np
import insightface
import time
import os
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

# Configuration
NUM_RUNS = 5
TEST_VIDEOS = [f"input/test_input_{i}.mp4" for i in range(4, 5)]
DETECTORS = ["mediapipe", "scrfd", "yunet"]
frame_skip = 0
display = False

# Results storage
results = {
    "mediapipe": {"fps": [], "inference_ms": []},
    "scrfd": {"fps": [], "inference_ms": []},
    "yunet": {"fps": [], "inference_ms": []}
}

# Suppress warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def initialize_detectors():
    """Initialize all face detection models"""
    print("Initializing detectors...")
    
    # MediaPipe
    try:
        mp_face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.2)
        print("✓ MediaPipe initialized")
    except Exception as e:
        print(f"✗ MediaPipe initialization failed: {e}")
        mp_face_detection = None
    
    # SCRFD
    try:
        # insightface.scrfd.
        scrfd_model_path = "./models/det_500m.onnx"
        scrfd_detector = insightface.model_zoo.get_model(scrfd_model_path)
        scrfd_detector.prepare(ctx_id=-1, input_size=(640, 640))
        print("✓ SCRFD initialized")
    except Exception as e:
        print(f"✗ SCRFD initialization failed: {e}")
        scrfd_detector = None
    
    # YuNet
    try:
        yunet_model_path = "face_detection_yunet_2023mar_int8bq.onnx"
        yunet = cv2.FaceDetectorYN.create(
            model=yunet_model_path, 
            config="", 
            input_size=(640, 640), 
            score_threshold=0.3,
            top_k=5000,
            nms_threshold=0.45
        )
        print("✓ YuNet initialized")
    except Exception as e:
        print(f"✗ YuNet initialization failed: {e}")
        yunet = None
    
    return mp_face_detection, scrfd_detector, yunet

def run_benchmark(detector_name, detector, video_path, run_number):
    """Run benchmark for a specific detector and video"""
    print(f"Running {detector_name} on {os.path.basename(video_path)} (Run {run_number+1}/{NUM_RUNS})")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return None, None
    
    frame_count = 0
    fps_start_time = time.time()
    fps_counter = 0
    fps = 0
    inference_times = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        height, width = frame.shape[:2]
        if width > height:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            width, height = height, width
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_count += 1
        
        if frame_count % (frame_skip + 1) == 0:
            prev_bbox = None
            
            inference_start = time.time()
            
            # Run detection based on detector type
            if detector_name == "mediapipe" and detector:
                results = detector.process(rgb_frame)
                if results.detections:
                    prev_bbox = [
                        (int(d.location_data.relative_bounding_box.xmin * width),
                         int(d.location_data.relative_bounding_box.ymin * height),
                         int(d.location_data.relative_bounding_box.width * width),
                         int(d.location_data.relative_bounding_box.height * height))
                        for d in results.detections
                    ][0]
            
            elif detector_name == "scrfd" and detector:
                faces, _ = detector.detect(rgb_frame, input_size=(640, 640))
                if faces is not None and len(faces) > 0:
                    for face in faces:
                        x1, y1, x2, y2, _ = face.astype(int)
                        prev_bbox = (x1, y1, x2 - x1, y2 - y1)
            
            elif detector_name == "yunet" and detector:
                yunet_input = cv2.resize(frame, (640, 640))
                try:
                    faces = detector.detect(yunet_input)
                    if faces[1] is not None:
                        x, y, w, h = map(int, faces[1][0][:4])
                        prev_bbox = (int(x * (width / 640)), int(y * (height / 640)), 
                                     int(w * (width / 640)), int(h * (height / 640)))
                except Exception as e:
                    print(f"YuNet detection error: {e}")
            
            inference_time = (time.time() - inference_start) * 1000  # ms
            inference_times.append(inference_time)
            
            if prev_bbox and display:
                x, y, w, h = prev_bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # FPS calculation
        fps_counter += 1
        elapsed = time.time() - fps_start_time
        
        if elapsed > 1.0:  # Update every second
            fps = fps_counter / elapsed
            fps_counter = 0
            fps_start_time = time.time()
        
        if display:
            resized = cv2.resize(frame, (256, 256))
            cv2.putText(resized, f"{detector_name}: {fps:.1f} FPS", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Detection", resized)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    
    if len(inference_times) == 0:
        return None, None
    
    avg_inference = sum(inference_times) / len(inference_times)
    return fps, avg_inference

def display_results(results):
    """Display benchmark results in table format"""
    # Create summary DataFrame
    summary = {}
    
    for detector in DETECTORS:
        if results[detector]["fps"] and results[detector]["inference_ms"]:
            avg_fps = sum(results[detector]["fps"]) / len(results[detector]["fps"])
            avg_inf = sum(results[detector]["inference_ms"]) / len(results[detector]["inference_ms"])
            summary[detector] = {
                "Average FPS": f"{avg_fps:.2f}",
                "Average Inference (ms)": f"{avg_inf:.2f}",
                "Theoretical Max FPS": f"{1000/avg_inf:.2f}"
            }
    
    # Display table
    df = pd.DataFrame(summary).transpose()
    print("\n" + "="*50)
    print("BENCHMARK RESULTS")
    print("="*50)
    print(tabulate(df, headers="keys", tablefmt="grid"))
    
    # Create bar chart
    plt.figure(figsize=(12, 6))
    for i, metric in enumerate(["Average FPS", "Average Inference (ms)"]):
        plt.subplot(1, 2, i+1)
        values = [float(summary[d][metric]) for d in DETECTORS if d in summary]
        bars = plt.bar(
            [d for d in DETECTORS if d in summary], 
            values, 
            color=['#3498db', '#2ecc71', '#e74c3c'][:len(values)]
        )
        plt.title(metric)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'{height:.2f}',
                ha='center', va='bottom'
            )
    
    plt.tight_layout()
    plt.savefig("benchmark_results.png")
    print("\nBenchmark chart saved as 'benchmark_results.png'")

def main():
    mp_face_detection, scrfd_detector, yunet = initialize_detectors()
    
    detectors = {
        "mediapipe": mp_face_detection,
        "scrfd": scrfd_detector,
        "yunet": yunet
    }
    
    # Run benchmarks
    for detector_name, detector in detectors.items():
        if detector is None:
            print(f"Skipping {detector_name} - not initialized")
            continue
        
        for video_path in TEST_VIDEOS:
            for run in range(NUM_RUNS):
                fps, inference_time = run_benchmark(detector_name, detector, video_path, run)
                
                if fps is not None and inference_time is not None:
                    results[detector_name]["fps"].append(fps)
                    results[detector_name]["inference_ms"].append(inference_time)
    
    # Display results
    display_results(results)

    # Clean up
    cv2.destroyAllWindows()
    print("\nBenchmark completed successfully!")

if __name__ == "__main__":
    # Check if required packages are installed
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        from tabulate import tabulate
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.check_call(["pip", "install", "pandas", "matplotlib", "tabulate"])
        # Reimport after installation
        import pandas as pd
        import matplotlib.pyplot as plt
        from tabulate import tabulate
    
    main()