# Experimental: Script with dynamic frame skipping based on motion detection

import cv2
import time
import blur_functions as bf
import sys
import utils
import subprocess

class Config:
    INPUT_VIDEO_PATH = "input/test2.mp4"
    # INPUT_VIDEO_PATH = 0
    OUTPUT_VIDEO_PATH = "output/test2_dynamic.mp4"
    SAVE_OUTPUT = True
    AUDIO = False
    DETECTOR_MODEL = "yunet" ## or "yunet"
    # Dynamic skipping parameters
    MIN_FRAME_SKIP = 0
    MAX_FRAME_SKIP = 10
    INITIAL_FRAME_SKIP = 3
    MSE_LOW_THRESHOLD = 75   # Low motion threshold
    MSE_HIGH_THRESHOLD = 100  # High motion threshold
    
    TRACKING_SKIP = 0
    BLUR_ENABLED = True
    ROTATION_DEGREES = 90

    LANDMARK_STALENESS_THRESHOLD = 15
    BOX_STALENESS_THRESHOLD = 15
    
    # SCRFD Config
    SCRFD_MODEL_PATH = "det_500m.onnx"
    SCRFD_INPUT_SIZE = (640, 640)

    # --- Visualization & Logging ---
    OVERLAY_LANDMARKS = False
    OVERLAY_DETECTOR_BOX = True
    DISPLAY_VIDEO = True
    SHOW_SKIP_VALUE = True  

def overlay_stats(frame, frame_count, current_frame_skip, avg_fps):
    # Text settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.5
    color = (0, 255, 0)  # Green text
    thickness = 5
    line_height = 25
    x, y = 10, 30  # Starting position

    cv2.putText(frame, f"Skip: {current_frame_skip}", (x, y + line_height), font, font_scale, color, thickness)

    return frame


# Dynamic frame skip manager
class DynamicSkipManager:
    def __init__(self, config):
        self.min_skip = config.MIN_FRAME_SKIP
        self.max_skip = config.MAX_FRAME_SKIP
        self.current_skip = config.INITIAL_FRAME_SKIP
        self.mse_low = config.MSE_LOW_THRESHOLD
        self.mse_high = config.MSE_HIGH_THRESHOLD
        # History to smooth out changes
        self.mse_history = []
        self.history_size = 5
        
    def update_skip_value(self, mse_values):
        if not mse_values:
            return self.current_skip
            
        # Get max MSE value (most significant movement)
        current_mse = max(mse_values)
        
        # Add to history and maintain history size
        self.mse_history.append(current_mse)
        if len(self.mse_history) > self.history_size:
            self.mse_history.pop(0)
            
        # Use average of recent MSE values for stability
        avg_mse = sum(self.mse_history) / len(self.mse_history)
        # print(avg_mse)
        # Update skip value based on motion
        if avg_mse < self.mse_low:
            # Low motion - increase skip (slower detection)
            self.current_skip = min(self.current_skip + 1, self.max_skip)
        elif avg_mse > self.mse_high:
            # High motion - decrease skip (faster detection)
            self.current_skip = max(self.current_skip - 2, self.min_skip)
        
        # Round to integer
        self.current_skip = int(self.current_skip)
        return self.current_skip

total_inference_times = []
total_blur_times = []
total_detector_times = []
total_landmarker_times = []
total_scale_times = []
total_tracking_times = []
total_mse_times = []

lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

if __name__ == "__main__":
    try:
        config = Config()
        skip_manager = DynamicSkipManager(config)
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
        
        current_frame_skip = skip_manager.current_skip
        frames_since_detection = 0
        
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
                current_frame_skip = skip_manager.update_skip_value(mse_values)
            
            frames_since_detection += 1
            
            # Determine if we should run detector
            run_detector = (
                frame_count == 1 or 
                frames_since_detection > current_frame_skip or
                (mse_values and max(mse_values) > config.MSE_HIGH_THRESHOLD)
            )
            # if mse_values:
            #     print(max(mse_values))
            if (mse_values and max(mse_values) > config.MSE_HIGH_THRESHOLD):
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
                frames_since_detection = 0
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
                            last_landmark_boxes = landmark_boxes.copy()
                            if config.BLUR_ENABLED:
                                start_blur_time = time.perf_counter()
                                # box = x1, y1, x2, y2
                                # output_frame = bf.apply_box_blur(output_frame, box)
                                # output_frame = bf.apply_ellipse_blur(output_frame, landmarks_scaled)
                                output_frame = bf.apply_blur(output_frame, landmarks_scaled)
                                # output_frame = bf.apply_gaussian_blur(output_frame, landmarks_scaled, scale=1.0, ksize=45, sigma=45)
                                # output_frame = bf.apply_gaussian_blur(output_frame, landmarks_scaled)
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
                color = (0, 255, 0)
                for landmarks in landmark_boxes:
                    for lx, ly in landmarks:
                        cv2.circle(output_frame, (lx, ly), 1, color, -1)
            
            prev_gray = output_frame
            
            frame_end_time = time.perf_counter()
            inference_time_ms = (frame_end_time - frame_start_time) * 1000
            total_inference_times.append(inference_time_ms)

            avg_fps = 0
            if total_inference_times:
                avg_inference = sum(total_inference_times) / len(total_inference_times)
                avg_fps = 1000 / avg_inference if avg_inference > 0 else 0

            # Overlay stats onto the frame (before display and saving)
            output_frame_with_stats = overlay_stats(output_frame.copy(), frame_count, current_frame_skip, avg_fps)

            # Display video (optional)
            if config.DISPLAY_VIDEO:
                display_frame = cv2.resize(output_frame_with_stats, (512, 512))
                cv2.imshow("Face Blurring Output", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Save output video with stats
            if config.SAVE_OUTPUT and out:
                out.write(output_frame_with_stats)

        end_process_time = time.time()
        cap.release()
        if out:
            out.release()
            
        cv2.destroyAllWindows()

        if config.AUDIO:
            print("Merging audio...")
            temp_audio_path = "output/temp_audio.aac"
            # Extract audio from original video
            extract_audio_cmd = [
                "ffmpeg", "-y", "-i", config.INPUT_VIDEO_PATH, "-vn",
                "-acodec", "copy", temp_audio_path
            ]
            subprocess.run(extract_audio_cmd, check=True)

            # Merge audio with processed video
            final_output_path = "output/blurred_audio.mp4"
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
    except Exception as e:
        sys.exit(1)
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        if 'out' in locals() and out is not None:
            out.release()
        cv2.destroyAllWindows()