from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Cipher import PKCS1_OAEP
from Crypto.PublicKey import RSA
import base64
import json
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import uuid
import threading
import queue
import cv2
import time
import src.blur_functions as bf
import src.utils as utils
import subprocess
import torch
from src.backbones import get_model
from picamera2 import Picamera2
from libcamera import controls # Import controls for camera settings


read_queue = queue.Queue()
write_queue = queue.Queue()

def read_thread(config, picam2):
    global total_frame_read_times
    read_count = 0
    while read_count < config.CAM_FRAMES:
        st = time.perf_counter()
        frame = picam2.capture_array()
        en = time.perf_counter()
        total_frame_read_times += (en - st) * 1000
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        read_count += 1
        read_queue.put(rgb_frame)
    read_queue.put(None)
    picam2.stop()

def write_thread(config, out):
    global total_frame_write_times
    if config.SAVE_OUTPUT:
        write_count = 0
        while True:
            frame = write_queue.get()
            if frame is None:
                break
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            st = time.perf_counter()
            out.write(bgr_frame)
            en = time.perf_counter()
            total_frame_write_times += (en - st) * 1000
            write_count += 1

def process_thread(config, detector, interpretor, input_details, output_details, model, public_key):
    global total_detector_times, total_blur_times, total_landmarker_times, total_tracking_times, total_mse_times, total_inference_times
    frame_count = 0
    blur_count = 0
    detector_count = 0
    landmarker_count = 0
    tracking_count = 0
    mse_thresh_count = 0
    mse_count = 0

    detected_faces_boxes = []
    landmark_boxes = []

    frames_since_last_box = 0

    prev_gray = None
    prev_frame = None

    face_metadata = {}  # sidecar dict to collect encrypted metadata
    face_tracks = {}  # face_id: {"box": (x1,y1,x2,y2), "aes_key": key}
    MAX_CENTER_DIST = 60 # Vary this

    face_id_to_aes_key = {}

    while True:
        frame = read_queue.get()
        if config.APPLY_PROTOCOL == False:
            if frame is None:  # Sentinel received
                write_queue.put(None)
                break
            write_queue.put(frame)
            continue
        
        if frame is None:  # Sentinel received
            write_queue.put(None)
            break

        frame_start_time = time.perf_counter()
        frame_count += 1

        mse_values = []
        if prev_frame is not None and detected_faces_boxes:
            for face_region in detected_faces_boxes:
                st = time.perf_counter()
                mse = utils.calculate_frame_diff(prev_frame, frame, face_region)
                en = time.perf_counter()
                total_mse_times += (en-st)*1000
                mse_count += 1
                mse_values.append(mse)

        run_detector = (
            frame_count == 1 or 
            frame_count % (config.FRAME_SKIP + 1) == 0 or 
            (mse_values and max(mse_values) > config.MSE_THRESHOLD)
        )
        if (mse_values and max(mse_values) > config.MSE_THRESHOLD):
            mse_thresh_count += 1

        current_height, current_width = frame.shape[:2]
        rotated_frame = frame

        if config.INPUT_VIDEO_PATH != 0:
            if config.ROTATION_DEGREES == 90 and current_width > current_height:
                rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif config.ROTATION_DEGREES == 270 and current_width > current_height:
                rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        output_frame = rotated_frame
        output_h, output_w = output_frame.shape[:2]
        rgb_frame = output_frame
        prev_frame = output_frame

        if run_detector:
            detector_count += 1
            start_detector_time = time.perf_counter()
            faces = utils.detect_faces(rgb_frame, detector, config)
            end_detector_time = time.perf_counter()
            total_detector_times += (end_detector_time - start_detector_time) * 1000
            detected_faces_boxes = [utils.expand_box_with_margin(face["box"], margin=0.1, frame_width=output_w, frame_height=output_h) for face in faces]
            detected_face_scores = [face["score"] for face in faces]
            
            if detected_faces_boxes:
                frames_since_last_box = 0
            else:
                frames_since_last_box += 1
                if frames_since_last_box >= config.BOX_STALENESS_THRESHOLD:
                    detected_faces_boxes = []
                face_tracks = {}
        else:
            frames_since_last_box += 1
            if frames_since_last_box >= config.BOX_STALENESS_THRESHOLD:
                detected_faces_boxes = []
            if prev_gray is not None:
                if frame_count % (config.TRACKING_SKIP+1) == 0:
                    start = time.perf_counter()
                    detected_faces_boxes = [utils.track_box_with_optical_flow(prev_gray, rgb_frame, box, lk_params, scale_factor=0.4)[0] for box in detected_faces_boxes] 
                    end = time.perf_counter()
                    tracking_count += 1
                    total_tracking_times += (end-start)*1000

        landmark_boxes = []

        new_tracks = {}

        for i, box in enumerate(detected_faces_boxes):
            cx, cy = box_center(box)

            matched_id = None
            min_dist = float("inf")
            for face_id, info in face_tracks.items():
                pcx, pcy = box_center(info["box"])
                dist = np.linalg.norm(np.array([cx, cy]) - np.array([pcx, pcy]))
                if dist < MAX_CENTER_DIST and dist < min_dist:
                    matched_id = face_id
                    min_dist = dist

            if matched_id:
                new_tracks[matched_id] = {"box": box, "aes_key": face_tracks[matched_id]["aes_key"]}
            else:
                # Assign new face ID and AES key
                new_id = str(uuid.uuid4())
                new_key = get_random_bytes(16)
                # print("\n\n\n", new_key, "\n\n\n")
                face_id_to_aes_key[new_id] = new_key  # store plain key

                new_tracks[new_id] = {
                    "box": box,
                    "aes_key": new_key,
                }
                
            x1, y1, x2, y2 = box
            if x1 < x2 and y1 < y2:
                x1 = max(x1, 0)
                y1 = max(y1, 0)
                x2 = min(x2, current_width)
                y2 = min(y2, current_height)
                face_roi_rgb = rgb_frame[y1:y2, x1:x2]
                if face_roi_rgb.size > 0:
                    start_landmarker_time = time.perf_counter()
                    landmarks, score = utils.get_landmarks_interpretor_with_score(face_roi_rgb, interpretor, input_details, output_details)
                    end_landmarker_time = time.perf_counter()
                    total_landmarker_times += (end_landmarker_time - start_landmarker_time) * 1000
                    landmarker_count += 1

                    landmarks_scaled = utils.scale_landmarks(landmarks, x1, y1, x2 - x1, y2 - y1)
                    landmark_boxes.append(landmarks)

                    #---- encryption                        
                    # Save original face before blur for encryption
                    face_id = matched_id if matched_id else new_id
                    track_info = new_tracks[face_id]
                    aes_key = track_info["aes_key"]
                    # encrypted_box = track_info["box"]
                    # x1, y1, x2, y2 = encrypted_box

                    # new_x1, new_y1, new_x2, new_y2 = utils.expand_box_with_margin((x1, y1, x2, y2), margin=0.2, frame_width=output_w, frame_height=output_h)
                    
                    frame_height, frame_width = frame.shape[:2]
                    hull = cv2.convexHull(landmarks_scaled)
                    x, y, w, h = cv2.boundingRect(hull)
                    
                    # Ensure the roi is within frame boundaries
                    x = max(0, x)
                    y = max(0, y)
                    w = min(w, frame_width - x)
                    h = min(h, frame_height - y)

                    landmarks_adjusted = np.array(landmarks_scaled)  # shape: (468, 3)
                    landmarks_adjusted[:, 0] -= x  # x
                    landmarks_adjusted[:, 1] -= y  # y

                    face_image_for_encryption = rgb_frame[y:y+h, x:x+w]

                    _, buffer = cv2.imencode('.png', face_image_for_encryption)
                    face_image_encoded = base64.b64encode(buffer).decode('utf-8')
                    data = {
                        "frame": frame_count,
                        "landmarks": np.array(landmarks_adjusted).tolist(),
                        "face_image": face_image_encoded,
                        "box": [x, y, x+w, y+h],
                        "box_score": float(detected_face_scores[i]),
                        "land_score": float(score),
                        "face_id": face_id
                    }
                    # json_data = json.dumps(data).encode('utf-8')
                    # encrypted = encrypt_data_aes128(json_data, aes_key)
                    
                    # Initialize frame entry if not already
                    if face_id not in face_metadata:
                        face_metadata[face_id] = []

                    face_metadata[face_id].append(data)
                    
                if config.BLUR_ENABLED:
                    start_blur_time = time.perf_counter()
                    output_frame = bf.apply_blur_new(rgb_frame, hull, x, y, w, h)
                    # output_frame = bf.apply_gaussian_blur(rgb_frame, landmarks_scaled)
                    end_blur_time = time.perf_counter()
                    total_blur_times += (end_blur_time - start_blur_time) * 1000
                    blur_count += 1

        face_tracks = new_tracks

        if config.OVERLAY_DETECTOR_BOX:
            for (x1, y1, x2, y2) in detected_faces_boxes:
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        if config.OVERLAY_LANDMARKS:
            color = (0, 255, 0)
            for landmarks in landmark_boxes:
                for lx, ly in landmarks:
                    cv2.circle(output_frame, (lx, ly), 1, color, -1)
        
        prev_gray = output_frame

        if config.DISPLAY_VIDEO:
            display_frame = cv2.resize(output_frame, (512, 512))
            cv2.imshow("Face Blurring Output", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        frame_end_time = time.perf_counter()
        write_queue.put(output_frame)
        
        inference_time_ms = (frame_end_time - frame_start_time) * 1000
        total_inference_times += inference_time_ms

    if config.APPLY_PROTOCOL == False:
        return

    # Encrypt all collected metadata at the end, could have been done during the loop as well.
    encrypted_face_metadata = {}
    to_be_embedded = {}
    
    st = time.perf_counter()
    for face_id, entries in face_metadata.items():
        aes_key = face_id_to_aes_key[face_id]
        encrypted_entries = []
        best_entry = entries[0]
        best_land_score = best_entry.get("land_score", -1)
        # best_box_score = best_entry.get("box_score", -1)
        max_index = 0
        # why encrypt each entry when i can just encrypt every time this guys face shows up at once.
        for i, entry in enumerate(entries):
            json_data = json.dumps(entry).encode('utf-8')
            encrypted = encrypt_data_aes128(json_data, aes_key)
            encrypted_entries.append(encrypted)
            
            land_score = entry.get("land_score", -1)
            box_score = entry.get("box_score", -1)

            # Priority: land_score first, then box_score
            # if (land_score > best_land_score) or \
            # (land_score == best_land_score and box_score > best_box_score):
            if (land_score > best_land_score):
                best_land_score = land_score
                # best_box_score = box_score
                max_index = i
        to_be_embedded[face_id] = (entries[max_index].get("face_image", None), entries[max_index].get("landmarks", None))


        encrypted_face_metadata[f"frame_{face_id}"] = encrypted_entries
    
    en = time.perf_counter()
    encryption_time = (en - st) * 1000

    st = time.perf_counter()
    batch_tensor, face_id_order = utils.preprocess_face_images(to_be_embedded, config.EMBEDDING_MODEL)
    en = time.perf_counter()
    preprocessing_time_for_embedding = (en-st)*1000
    
    if config.EMBEDDING_MODEL == "edgeface_s_gamma_05":
        with torch.no_grad():
            st = time.perf_counter()
            embeddings = model(batch_tensor)  # shape: (12, emb_dim)
            en = time.perf_counter()
            embedding_time = (en - st) * 1000
            num_embeddings = len(face_id_order)
    else:
        st = time.perf_counter()
        embeddings = model.predict(batch_tensor)
        en = time.perf_counter()
        embedding_time = (en - st) * 1000
        num_embeddings = len(face_id_order)
    
    if config.EMBEDDING_MODEL == "edgeface_s_gamma_05":
        face_embeddings = embeddings.numpy().tolist()  # Shape: (N, 512)
    else:
        face_embeddings = embeddings.tolist()
        
    # New dicts to store AES key + embedding
    face_id_to_combined_key_embedding = {}
    cipher_rsa = PKCS1_OAEP.new(public_key)

    for i, face_id in enumerate(face_id_order):
        aes_key = face_id_to_aes_key[face_id]   # This is the raw AES key (bytes)
        embedding = face_embeddings[i]          # This is a list of 512 floats

        # Encrypt the AES key with RSA
        encrypted_aes_key = cipher_rsa.encrypt(aes_key)
        encrypted_aes_key_b64 = base64.b64encode(encrypted_aes_key).decode('utf-8')

        # Encrypt the embedding with AES
        encrypted_embedding_b64 = encrypt_data_aes128(json.dumps(embedding).encode('utf-8'), aes_key)

        # Store raw AES key + embedding
        face_id_to_combined_key_embedding[face_id] = {
            "aes_key": encrypted_aes_key_b64,  # Encode for JSON safety
            "embedding": encrypted_embedding_b64
        }
    
    video_identifier = str(uuid.uuid4())
    face_id_to_combined_key_embedding["video_identifier"] = video_identifier

    st = time.perf_counter()
    with open("../output/face_keys.json", "w") as f:
        json.dump(face_id_to_combined_key_embedding, f, indent=2)

    # Save encrypted metadata
    with open("../output/video_metadata_encrypted.json", "w") as f:
        json.dump(encrypted_face_metadata, f, indent=2)
    end = time.perf_counter()
    saving_time = (end - st) * 1000

    metrics = {
        "total_detector_times": (total_detector_times, detector_count),
        "total_blur_times": (total_blur_times, blur_count),
        "total_landmarker_times": (total_landmarker_times, landmarker_count),
        "total_tracking_times": (total_tracking_times, tracking_count),
        "total_mse_times": (total_mse_times, mse_count),
        "total_entire_frame_times": (total_inference_times, frame_count),
        "encryption_time": (encryption_time, 1),
        "preprocessing_time_for_embedding": (preprocessing_time_for_embedding, 1),
        "embedding_time": (embedding_time, num_embeddings),
        "saving_time (encrypted data + embeddings)": (saving_time, 1),
    }
    return metrics

# Helper: compute center of box
def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)

# Encryption utility functions
def pad(data):
    pad_len = 16 - (len(data) % 16)
    return data + bytes([pad_len]) * pad_len

def encrypt_data_aes128(data, key):
    cipher = AES.new(key, AES.MODE_ECB)
    padded_data = pad(data)
    encrypted = cipher.encrypt(padded_data)
    return base64.b64encode(encrypted).decode('utf-8')

# Configuration knob for encryption rate
ENCRYPTION_FRAME_INTERVAL = 1  # Encrypt every x frames

# Load public key from PEM file
with open("public_key.pem", "rb") as f:
    public_key = RSA.import_key(f.read())

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class Config:
    # INPUT_VIDEO_PATH = "../input/test.mov"
    INPUT_VIDEO_PATH = 0
    OUTPUT_VIDEO_PATH = "../output/blurred.mp4"
    SAVE_OUTPUT = True
    APPLY_PROTOCOL = True
    AUDIO = False
    DETECTOR_MODEL = "yunet" ## or "yunet"
    # EMBEDDING_MODEL = "ghost" # or "edgeface_s_gamma_05"
    EMBEDDING_MODEL = "edgeface_s_gamma_05"
    EMBEDDING_MODEL_PATH = "../src/models/edgeface_s_gamma_05.pt"
    # EMBEDDING_MODEL_PATH = "../src/models/GN_W0.5_S2_ArcFace_epoch16.h5"
    FRAME_SKIP = 5
    TRACKING_SKIP = 0
    BLUR_ENABLED = True
    ROTATION_DEGREES = 90
    CAM_FRAMES = 300  # Number of frames to capture if using webcam
    

    LANDMARK_STALENESS_THRESHOLD = 15
    BOX_STALENESS_THRESHOLD = 15
    MSE_THRESHOLD = 150

    # SCRFD Config
    SCRFD_MODEL_PATH = "../src/models/det_500m.onnx"
    SCRFD_INPUT_SIZE = (640, 640)

    # --- Visualization & Logging ---
    OVERLAY_LANDMARKS = False
    OVERLAY_DETECTOR_BOX = False
    DISPLAY_VIDEO = False

total_inference_times = 0
total_blur_times = 0
total_detector_times = 0
total_landmarker_times = 0
total_tracking_times = 0
total_mse_times = 0
total_frame_read_times = 0
total_frame_write_times = 0

lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

if __name__ == "__main__":
    # try:
    # All 3 Ml models be pre-loaded in the glasses at boot time
        config = Config()
        # Initialize video reader
        picam2 = Picamera2()
        # Configure the camera with a video configuration, using a 640x480 resolution
        # for processing. Use the YUV420 sensor format for efficiency.
        # The output format is XRGB which can be converted to RGB.
        picam2.configure(picam2.create_video_configuration(main={"size": (1000, 1000), "format": "XRGB8888"}))
        picam2.set_controls({"FrameRate": 30.0})
        picam2.start()
        
        # Initialize video writer if saving output
        if config.SAVE_OUTPUT:
            w, h = picam2.video_configuration.main.size
            out = cv2.VideoWriter(config.OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))

            
        detector = utils.initialize_detector(config)
        interpretor, input_details, output_details = utils.initialize_landmark_detector()
        
        if config.EMBEDDING_MODEL == "edgeface_s_gamma_05":
            model_name = "edgeface_s_gamma_05"
            model = get_model(model_name)
            model.load_state_dict(torch.load(config.EMBEDDING_MODEL_PATH, map_location='cpu'))
            model.eval()
        # elif config.EMBEDDING_MODEL == "ghost":
            # model = GhostFaceNets.buildin_models("ghostnetv1", dropout=0, emb_shape=512, output_layer='GDC', bn_momentum=0.9, bn_epsilon=1e-5)
            # model = GhostFaceNets.add_l2_regularizer_2_model(model, weight_decay=5e-4, apply_to_batch_normal=False)
            # model = GhostFaceNets.replace_ReLU_with_PReLU(model, target_activation='PReLU')
            # model.load_weights(config.EMBEDDING_MODEL_PATH)

        results = {}

        def process_thread_wrapper(config, detector, interpretor, input_details, output_details, model, public_key, results_container):
            metrics = process_thread(config, detector, interpretor, input_details, output_details, model, public_key)
            results_container['process_metrics'] = metrics

        t1 = threading.Thread(target=read_thread, args=(config, cap))
        t2 = threading.Thread(target=process_thread_wrapper, args=(config, detector, interpretor, input_details, output_details, model, public_key, results))
        t3 = threading.Thread(target=write_thread, args=(config, out))
        
        start_process_time = time.perf_counter()
        t1.start()
        t2.start()
        t3.start()

        t1.join()
        t2.join()
        t3.join()
        end_process_time = time.perf_counter()
    
        total_process_time = (end_process_time - start_process_time)
        print(f"\n\nTotal Protocol processing time: {total_process_time:.2f} s")
    
        # Access and print the detailed metrics
        if 'process_metrics' in results:
            metrics = results['process_metrics']
            print("\n--- Detailed Processing Metrics (in ms) ---")
            for metric_name, (total_time, count) in metrics.items():
                if count > 0:
                    avg_time = total_time / count
                    print(f"{metric_name.replace('total_', '').replace('_times', '').title()}:")
                    print(f"  Total Time: {total_time:.2f} ms")
                    print(f"  Count: {count}")
                    print(f"  Average Time: {avg_time:.2f} ms/op")
                    print("-----------------------------------------")
                else:
                    print(f"{metric_name.replace('total_', '').replace('_times', '').title()}:")
                    print("  No operations performed.")
                    print("-----------------------------------------")
        picam2.close()
        if out:
            out.release()
        cv2.destroyAllWindows()