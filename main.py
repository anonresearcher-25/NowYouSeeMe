# Initialize config class with appropriate parameters
# Run using python main.py

import os
import cv2
import time
import src.blur_functions as bf
import src.utils as utils
import subprocess

class Config:
    INPUT_VIDEO_PATH = "./input/test_input_6.mp4"
    # INPUT_VIDEO_PATH = 0
    OUTPUT_VIDEO_PATH = "./output/blurred.mp4"
    SAVE_OUTPUT = True
    AUDIO = True
    DETECTOR_MODEL = "scrfd" ## or "yunet"
    FRAME_SKIP = 0
    TRACKING_SKIP = 0
    BLUR_ENABLED = False
    ROTATION_DEGREES = 90
    GENERATE_LANDMARKS_FILE = False

    LANDMARK_STALENESS_THRESHOLD = 15
    BOX_STALENESS_THRESHOLD = 15
    MSE_THRESHOLD = 100
    
    # SCRFD Config
    SCRFD_MODEL_PATH = "./src/models/det_500m.onnx"
    SCRFD_INPUT_SIZE = (640, 640)

    # --- Visualization & Logging ---
    OVERLAY_LANDMARKS = True
    OVERLAY_DETECTOR_BOX = False
    DISPLAY_VIDEO = True

total_inference_times = []
total_blur_times = []
total_detector_times = []
total_landmarker_times = []
total_scale_times = []
total_tracking_times = []
total_mse_times = []

lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

if __name__ == "__main__":
    # try:
        config = Config()
        detector = utils.initialize_detector(config)
        interpretor, input_details, output_details = utils.initialize_landmark_detector()
        cap, out, orig_frame_width, orig_frame_height = utils.initialize_video_io(config)

        frame_count = 0
        blur_count = 0
        detector_count = 0
        landmarker_count = 0
        tracking_count = 0
        mse_thresh_count = 0
        start_process_time = time.time()

        detected_faces_boxes = []
        landmark_boxes = []
        last_landmark_boxes = []
        
        frames_since_last_landmark = 0
        frames_since_last_box = 0
        
        prev_gray = None
        prev_frame = None
        
        os.makedirs("landmarks", exist_ok=True)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            frame_start_time = time.perf_counter()

            mse_values = []
            if prev_frame is not None and detected_faces_boxes:
                for face_region in detected_faces_boxes:
                    st = time.perf_counter()
                    mse = utils.calculate_frame_diff(prev_frame, frame, face_region)
                    en = time.perf_counter()
                    total_mse_times.append((en-st)*1000)
                    mse_values.append(mse)
            
            # Determine if we should run detector
            run_detector = (
                frame_count == 1 or 
                frame_count % (config.FRAME_SKIP + 1) == 0 or 
                (mse_values and max(mse_values) > config.MSE_THRESHOLD)
            )
            # if mse_values:
            #     print(max(mse_values))
            if (mse_values and max(mse_values) > config.MSE_THRESHOLD):
                mse_thresh_count += 1

            current_height, current_width = frame.shape[:2]
            rotated_frame = frame

            if config.INPUT_VIDEO_PATH == 0:
                None
            else:
                if config.ROTATION_DEGREES == 90 and current_width > current_height:
                    None
                    rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                elif config.ROTATION_DEGREES == 270 and current_width > current_height:
                    rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            output_frame = rotated_frame
            output_h, output_w = output_frame.shape[:2]
            # rgb_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
            rgb_frame = output_frame
            prev_frame = output_frame

            # if frame_count==1 or frame_count % (config.FRAME_SKIP + 1) == 0 or detected_faces_boxes == []:
            if run_detector:   
                detector_count += 1
                start_detector_time = time.perf_counter()
                faces = utils.detect_faces(rgb_frame, detector, config)
                end_detector_time = time.perf_counter()
                total_detector_times.append((end_detector_time - start_detector_time) * 1000)
                detected_faces_boxes = [utils.expand_box_with_margin(face["box"], margin=0.1, frame_width=output_w, frame_height=output_h) for face in faces]
                if detected_faces_boxes:
                    frames_since_last_box = 0
                else:
                    frames_since_last_box += 1
                    if frames_since_last_box >= config.BOX_STALENESS_THRESHOLD:
                        detected_faces_boxes = []
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
                        total_tracking_times.append((end-start)*1000)
            landmark_boxes = []
            current_landmarks_found = False
            
            for box in detected_faces_boxes:
                x1, y1, x2, y2 = box
                if x1 < x2 and y1 < y2:
                    x1 = max(x1, 0)
                    y1 = max(y1, 0)
                    x2 = min(x2, current_width)
                    y2 = min(y2, current_height)
                    face_roi_rgb = rgb_frame[y1:y2, x1:x2]
                    if face_roi_rgb.size > 0:
                        start_landmarker_time = time.perf_counter()
                        # landmarks = get_landmarks(face_roi_rgb, mp_face_mesh)
                        landmarks = utils.get_landmarks_interpretor(face_roi_rgb, interpretor, input_details, output_details)
                        end_landmarker_time = time.perf_counter()
                        total_landmarker_times.append((end_landmarker_time - start_landmarker_time) * 1000)
                        landmarker_count += 1

                        if len(landmarks):
                            frames_since_last_landmark = 0
                            current_landmarks_found = True
                            landmarks_scaled = utils.scale_landmarks(landmarks, x1, y1, x2 - x1, y2 - y1)
                            
                            landmark_boxes.append(landmarks)
                            
                            if config.GENERATE_LANDMARKS_FILE:
                                # Save landmarks to file
                                landmark_file_path = os.path.join("landmarks", f"frame_{frame_count:04d}.txt")
                                with open(landmark_file_path, "w") as f:
                                    for (lx, ly) in landmarks_scaled:
                                        f.write(f"{lx},{ly}\n")
    
                            last_landmark_boxes = landmark_boxes.copy()
                            if config.BLUR_ENABLED:
                                start_blur_time = time.perf_counter()
                                # box = x1, y1, x2, y2
                                # output_frame = bf.apply_box_blur(output_frame, box)
                                # output_frame = bf.apply_ellipse_blur(output_frame, landmarks_scaled)
                                # output_frame = bf.apply_blur(output_frame, landmarks_scaled)
                                # output_frame = bf.apply_gaussian_blur(output_frame, landmarks_scaled, scale=1.0, ksize=5, sigma=5)
                                # output_frame = bf.apply_gaussian_blur(output_frame, landmarks_scaled)
                                output_frame = bf.apply_gaussian_blur(output_frame, landmarks_scaled, scale=0.3, ksize=15, sigma=15)
                                end_blur_time = time.perf_counter()
                                total_blur_times.append((end_blur_time - start_blur_time) * 1000)
                                blur_count += 1
                        else:
                            if config.BLUR_ENABLED:
                                start_blur_time = time.perf_counter()
                                box = x1, y1, x2, y2
                                output_frame = bf.apply_box_blur(output_frame, box)
                                end_blur_time = time.perf_counter()
                                total_blur_times.append((end_blur_time - start_blur_time) * 1000)
                                blur_count += 1
                        
            if config.OVERLAY_DETECTOR_BOX:
                 for (x1, y1, x2, y2) in detected_faces_boxes:
                    cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            if config.OVERLAY_LANDMARKS:
                color = (255, 0, 0)
                for landmarks in landmark_boxes:
                    for lx, ly, _ in landmarks:
                        cv2.circle(output_frame, (int(lx), int(ly)), 3, color, -1)
            
            prev_gray = output_frame
            
            frame_end_time = time.perf_counter()
            inference_time_ms = (frame_end_time - frame_start_time) * 1000
            total_inference_times.append(inference_time_ms)

            if config.DISPLAY_VIDEO:
                display_frame = cv2.resize(output_frame, (1024, 1024))
                cv2.imshow("Face Blurring Output", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if config.SAVE_OUTPUT and out:
                out.write(output_frame)

        end_process_time = time.time()
        cap.release()
        if out:
            out.release()
            
        cv2.destroyAllWindows()

        if config.AUDIO:
            print("Merging audio...")
            temp_audio_path = "./output/temp_audio.aac"
            # Extract audio from original video
            print(f"Extracting audio from {config.INPUT_VIDEO_PATH}...")
            print(f"Temp audio path: {temp_audio_path}")
            extract_audio_cmd = [
                "ffmpeg", "-y", "-i", config.INPUT_VIDEO_PATH, "-vn",
                "-acodec", "copy", temp_audio_path
            ]
            subprocess.run(extract_audio_cmd, check=True)

            # Merge audio with processed video
            final_output_path = "./output/blurred_audio.mp4"
            utils.merge_audio_with_video(config.OUTPUT_VIDEO_PATH, temp_audio_path, final_output_path)
            print(f"Final video saved to: {final_output_path}")
        
        if total_inference_times:
            avg_inference = sum(total_inference_times) / len(total_inference_times)
            avg_fps = 1000 / avg_inference if avg_inference > 0 else 0
            print(f"--- Processing Summary ---")
            print(f"Total frames processed: {frame_count}")
            print(f"Frames processed by Detector: {detector_count}")
            print(f"Frames Tracked: {tracking_count}")
            print(f"Frames processed by landmarker: {landmarker_count}")
            print(f"Frames hit by mse thresh: {mse_thresh_count}")
            print(f"Frames Blurred: {blur_count}")
            print(f"Detector Duration: {(sum(total_detector_times) / len(total_detector_times)):.4f} ms")
            print(f"Landmarker Duration: {(sum(total_landmarker_times) / len(total_landmarker_times)):.4f} ms")
            print(f"Blur Duration: {(sum(total_blur_times) / len(total_blur_times)):.4f} ms")
            print(f"Tracking Duration: {(sum(total_tracking_times)/len(total_tracking_times)):.4f} ms")
            print(f"MSE Duration: {(sum(total_mse_times)/len(total_mse_times)):.4f} ms")
            print(f"Average Frame Processing Time: {avg_inference:.2f} ms")
            print(f"Average FPS: {avg_fps:.2f}")
            print(f"Total processing duration: {end_process_time - start_process_time:.2f} seconds")
    # except Exception as e:
    #     print(f"An error occurred: {e}")
    #     # sys.exit(1)
    # finally:
    #     if 'cap' in locals() and cap.isOpened():
    #         cap.release()
    #     if 'out' in locals() and out is not None:
    #         out.release()
    #     cv2.destroyAllWindows()