#!/usr/bin/env python3
"""
Face Restoration Script
Decrypts face metadata and restores faces in blurred video using provided face IDs and AES keys.
"""

import cv2
import json
import base64
import numpy as np
from collections import defaultdict
from Crypto.Cipher import AES

# Configuration
class RestoreConfig:
    BLURRED_VIDEO_PATH = "../output/blurred.mp4"
    OUTPUT_VIDEO_PATH = "../output/restored.mp4"
    ENCRYPTED_METADATA_PATH = "../output/video_metadata_encrypted.json"
    FACE_KEYS_JSON_PATH = "../output/final_matching_results.json"  # Path to the JSON with face_ids and keys
    DISPLAY_VIDEO = False
    SAVE_OUTPUT = True

def unpad(data):
    """Remove PKCS7 padding from decrypted data"""
    pad_len = data[-1]
    return data[:-pad_len]

def decrypt_data_aes128(encrypted_b64, key):
    """Decrypt AES128 encrypted data"""
    try:
        encrypted_data = base64.b64decode(encrypted_b64)
        cipher = AES.new(key, AES.MODE_ECB)
        decrypted_padded = cipher.decrypt(encrypted_data)
        decrypted = unpad(decrypted_padded)
        return decrypted.decode('utf-8')
    except Exception as e:
        print(f"Decryption failed: {e}")
        return None

def load_face_keys(json_path):
    """Load face IDs and their corresponding AES keys"""
    try:
        with open(json_path, 'r') as f:
            face_data = json.load(f)
        
        face_id_to_key = {}
        for person_name, person_data in face_data.items():
            members = person_data.get("members", [])
            for member in members:
                face_id = member["face_id"]
                aes_key_b64 = member["aes_key"]
                aes_key = base64.b64decode(aes_key_b64)
                face_id_to_key[face_id] = aes_key
                
        print(f"Loaded {len(face_id_to_key)} face IDs with keys")
        return face_id_to_key
    except Exception as e:
        print(f"Error loading face keys: {e}")
        return {}

def load_encrypted_metadata(metadata_path):
    """Load encrypted face metadata"""
    try:
        with open(metadata_path, 'r') as f:
            encrypted_metadata = json.load(f)
        print(f"Loaded encrypted metadata for {len(encrypted_metadata)} face tracks")
        return encrypted_metadata
    except Exception as e:
        print(f"Error loading encrypted metadata: {e}")
        return {}

def decrypt_face_metadata(encrypted_metadata, face_id_to_key):
    """Decrypt all face metadata using provided keys"""
    decrypted_metadata = {}
    
    for frame_key, encrypted_entries in encrypted_metadata.items():
        # Extract face_id from frame key (format: "frame_<face_id>")
        if not frame_key.startswith("frame_"):
            continue
            
        face_id = frame_key[6:]  # Remove "frame_" prefix
        
        if face_id not in face_id_to_key:
            print(f"Warning: No key found for face_id {face_id}, skipping...")
            continue
            
        aes_key = face_id_to_key[face_id]
        decrypted_entries = []
        
        for encrypted_entry in encrypted_entries:
            decrypted_json = decrypt_data_aes128(encrypted_entry, aes_key)
            if decrypted_json:
                try:
                    decrypted_data = json.loads(decrypted_json)
                    decrypted_entries.append(decrypted_data)
                except json.JSONDecodeError as e:
                    print(f"Error parsing decrypted JSON for face_id {face_id}: {e}")
            else:
                print(f"Failed to decrypt entry for face_id {face_id}")
        
        if decrypted_entries:
            decrypted_metadata[face_id] = decrypted_entries
            print(f"Successfully decrypted {len(decrypted_entries)} entries for face_id {face_id}")
    
    return decrypted_metadata

def organize_metadata_by_frame(decrypted_metadata):
    """Organize decrypted metadata by frame number for efficient lookup"""
    frame_to_faces = defaultdict(list)
    
    for face_id, entries in decrypted_metadata.items():
        for entry in entries:
            frame_num = entry["frame"]
            frame_to_faces[frame_num].append({
                "face_id": face_id,
                "data": entry
            })
    
    return dict(frame_to_faces)

def decode_face_image(face_image_b64):
    """Decode base64 encoded face image"""
    try:
        image_data = base64.b64decode(face_image_b64)
        nparr = np.frombuffer(image_data, np.uint8)
        face_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return face_image
    except Exception as e:
        print(f"Error decoding face image: {e}")
        return None

def restore_faces_in_frame(frame, face_data_list):
    """Restore faces in a single frame using decrypted face data"""
    restored_frame = frame.copy()
    
    for face_info in face_data_list:
        face_data = face_info["data"]
        face_id = face_info["face_id"]
        
        # Get face image and bounding box
        face_image_b64 = face_data.get("face_image")
        box = face_data.get("box")
        landmarks = face_data.get("landmarks")
        
        if not face_image_b64 or not box:
            print(f"Missing face image or box for face_id {face_id}")
            continue
            
        # Decode face image
        face_image = decode_face_image(face_image_b64)
        if face_image is None:
            print(f"Failed to decode face image for face_id {face_id}")
            continue
        
        # Extract box coordinates
        x1, y1, x2, y2 = box
        box_width = x2 - x1
        box_height = y2 - y1
        
        # Ensure coordinates are within frame bounds
        frame_height, frame_width = restored_frame.shape[:2]
        x1 = max(0, min(x1, frame_width - 1))
        y1 = max(0, min(y1, frame_height - 1))
        x2 = max(x1 + 1, min(x2, frame_width))
        y2 = max(y1 + 1, min(y2, frame_height))
        
        # Resize face image to match box dimensions
        try:
            current_box_width = x2 - x1
            current_box_height = y2 - y1
            
            if current_box_width > 0 and current_box_height > 0:
                resized_face = cv2.resize(face_image, (current_box_width, current_box_height))
                
                # Replace the region in the frame
                restored_frame[y1:y2, x1:x2] = resized_face
                print(f"Restored face {face_id} at box ({x1}, {y1}, {x2}, {y2})")
            else:
                print(f"Invalid box dimensions for face_id {face_id}")
                
        except Exception as e:
            print(f"Error restoring face {face_id}: {e}")
    
    return restored_frame

def main():
    config = RestoreConfig()
    
    print("Starting face restoration process...")
    
    # Load face keys
    print("Loading face keys...")
    face_id_to_key = load_face_keys(config.FACE_KEYS_JSON_PATH)
    if not face_id_to_key:
        print("Error: Could not load face keys. Exiting.")
        return
    
    # Load encrypted metadata
    print("Loading encrypted metadata...")
    encrypted_metadata = load_encrypted_metadata(config.ENCRYPTED_METADATA_PATH)
    if not encrypted_metadata:
        print("Error: Could not load encrypted metadata. Exiting.")
        return
    
    # Decrypt metadata
    print("Decrypting face metadata...")
    decrypted_metadata = decrypt_face_metadata(encrypted_metadata, face_id_to_key)
    if not decrypted_metadata:
        print("Error: Could not decrypt any metadata. Exiting.")
        return
    
    # Organize metadata by frame
    frame_to_faces = organize_metadata_by_frame(decrypted_metadata)
    print(f"Organized metadata for {len(frame_to_faces)} frames")
    
    # Open input video
    cap = cv2.VideoCapture(config.BLURRED_VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video {config.BLURRED_VIDEO_PATH}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Initialize output video writer
    out = None
    if config.SAVE_OUTPUT:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(config.OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))
    
    if (out is None or not out.isOpened()):
        print(f"Error: Could not open output video {config.OUTPUT_VIDEO_PATH}")

    
    frame_count = 0
    restored_faces_count = 0
    
    print("Processing video frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Check if this frame has face data to restore
        if frame_count in frame_to_faces:
            face_data_list = frame_to_faces[frame_count]
            restored_frame = restore_faces_in_frame(frame, face_data_list)
            restored_faces_count += len(face_data_list)
        else:
            restored_frame = frame
        
        # Display progress
        if frame_count % 100 == 0:
            print(f"Processed {frame_count}/{total_frames} frames")
        
        # Save frame
        if config.SAVE_OUTPUT and out:
            out.write(restored_frame)
        
        # Display frame
        if config.DISPLAY_VIDEO:
            display_frame = cv2.resize(restored_frame, (512, 512))
            cv2.imshow("Face Restoration", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Cleanup
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    
    print(f"\nRestoration complete!")
    print(f"Total frames processed: {frame_count}")
    print(f"Total faces restored: {restored_faces_count}")
    print(f"Output saved to: {config.OUTPUT_VIDEO_PATH}")

if __name__ == "__main__":
    main()