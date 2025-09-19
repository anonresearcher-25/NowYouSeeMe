import cv2
import time
import src.blur_functions as bf
import sys
import src.utils as utils
import subprocess
import json

class Config:
    def __init__(self, input_path, output_path, frame_skip, tracking_skip, mse_on, mse_thresh, tracking_on, detector_type="scrfd"):
        self.INPUT_VIDEO_PATH = input_path
        self.OUTPUT_VIDEO_PATH = output_path
        self.SAVE_OUTPUT = True
        self.AUDIO = False
        self.DETECTOR_MODEL = detector_type
        self.FRAME_SKIP = frame_skip
        self.TRACKING_SKIP = tracking_skip
        self.BLUR_ENABLED = True
        self.ROTATION_DEGREES = 90
        self.MSE_ON = mse_on
        self.TRACKING_ON = tracking_on
        
        self.LANDMARK_STALENESS_THRESHOLD = 15
        self.BOX_STALENESS_THRESHOLD = 15
        self.MSE_THRESHOLD = mse_thresh
        
        # SCRFD Config
        self.SCRFD_MODEL_PATH = "./models/det_500m.onnx"
        self.SCRFD_INPUT_SIZE = (640, 640)

        # --- Visualization & Logging ---
        self.OVERLAY_LANDMARKS = False
        self.OVERLAY_DETECTOR_BOX = False
        self.DISPLAY_VIDEO = False

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
        if len(sys.argv) < 9:
            print("Incorrect Usage")
            sys.exit(1)
        input_path = sys.argv[1]
        output_path = sys.argv[2]
        frame_skip  = int(sys.argv[3])
        mse_thresh = int(sys.argv[4])
        mse_on = bool(int(sys.argv[5]))
        tracking_on = bool(int(sys.argv[6]))
        tracking_skip = int(sys.argv[7])
        detector_type = str(sys.argv[8])

        config = Config(
            input_path=input_path,
            output_path=output_path,
            frame_skip=frame_skip,
            tracking_skip=tracking_skip,
            mse_on=mse_on,
            mse_thresh=mse_thresh,
            tracking_on=tracking_on,
            detector_type=detector_type
        )
                
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
        
        prev_frame_tracked = None
        prev_frame = None
        
        mse_values = None
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            frame_start_time = time.perf_counter()

            current_height, current_width = frame.shape[:2]
            rotated_frame = frame
            
            if config.MSE_ON:
                mse_values = []
                if prev_frame is not None and detected_faces_boxes:
                    for face_region in detected_faces_boxes:
                        st = time.perf_counter()
                        mse = utils.calculate_frame_diff(prev_frame, frame, face_region)
                        en = time.perf_counter()
                        total_mse_times.append((en-st)*1000)
                        mse_values.append(mse)

            run_detector = (
                frame_count == 1 or 
                frame_count % (config.FRAME_SKIP + 1) == 0 or 
                (mse_values and max(mse_values) > config.MSE_THRESHOLD)
            )
            
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
                if prev_frame_tracked is not None and config.TRACKING_ON:
                    if frame_count % (config.TRACKING_SKIP+1) == 0:
                        start = time.perf_counter()
                        detected_faces_boxes = [utils.track_box_with_optical_flow(prev_frame_tracked, rgb_frame, box, lk_params, scale_factor=0.5)[0] for box in detected_faces_boxes] 
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
                        landmarks = utils.get_landmarks_interpretor(face_roi_rgb, interpretor, input_details, output_details)
                        end_landmarker_time = time.perf_counter()
                        total_landmarker_times.append((end_landmarker_time - start_landmarker_time) * 1000)
                        landmarker_count += 1
                        if len(landmarks):
                            frames_since_last_landmark = 0
                            current_landmarks_found = True

                            landmarks_scaled = utils.scale_landmarks(landmarks, x1, y1, x2 - x1, y2 - y1)
                            landmark_boxes.append(landmarks_scaled)
                            last_landmark_boxes = landmark_boxes.copy()
                            if config.BLUR_ENABLED:
                                start_blur_time = time.perf_counter()
                                # output_frame = bf.apply_ellipse_blur(output_frame, landmarks_scaled)
                                output_frame = bf.apply_blur(output_frame, landmarks_scaled)
                                # output_frame = bf.apply_gaussian_blur(output_frame, landmarks_scaled, scale=1.0, ksize=45, sigma=45)
                                # output_frame = bf.apply_gaussian_blur(output_frame, landmarks_scaled)
                                end_blur_time = time.perf_counter()
                                total_blur_times.append((end_blur_time - start_blur_time) * 1000)
                                blur_count += 1
                        else:
                            # Wont execute
                            if config.BLUR_ENABLED:
                                start_blur_time = time.perf_counter()
                                box = x1, y1, x2, y2
                                output_frame = bf.apply_box_blur(output_frame, box)
                                end_blur_time = time.perf_counter()
                                total_blur_times.append((end_blur_time - start_blur_time) * 1000)
                                blur_count += 1
            if config.TRACKING_ON:
                prev_frame_tracked = output_frame
                
            frame_end_time = time.perf_counter()
            inference_time_ms = (frame_end_time - frame_start_time) * 1000
            total_inference_times.append(inference_time_ms)
            
            if config.OVERLAY_DETECTOR_BOX:
                 for (x1, y1, x2, y2) in detected_faces_boxes:
                    cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            if config.OVERLAY_LANDMARKS:
                color = (0, 255, 0)
                for landmarks in landmark_boxes:
                    for lx, ly in landmarks:
                        cv2.circle(output_frame, (lx, ly), 1, color, -1)

            if config.DISPLAY_VIDEO:
                display_frame = cv2.resize(output_frame, (512, 512))
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
            avg_inference = utils.calculate_avg(total_inference_times)
            avg_fps = 1000 / avg_inference if avg_inference > 0 else 0

            avg_tracking = utils.calculate_avg(total_tracking_times)
            avg_mse = utils.calculate_avg(total_mse_times)
            avg_detector = utils.calculate_avg(total_detector_times)
            avg_landmarker = utils.calculate_avg(total_landmarker_times)
            avg_blur = utils.calculate_avg(total_blur_times)

            summary_stats = {
                "video": config.INPUT_VIDEO_PATH,
                "frames": frame_count,
                "detector_frames": detector_count,
                "landmarker_frames": landmarker_count,
                "tracked_frames": tracking_count,
                "blurred_frames": blur_count,
                "frames_mse_thresh_hit": mse_thresh_count,
                "avg_detector_ms": round(avg_detector, 4),
                "avg_landmarker_ms": round(avg_landmarker, 4),
                "avg_blur_ms": round(avg_blur, 4),
                "avg_tracking_ms": round(avg_tracking, 4),
                "avg_mse_ms": round(avg_mse, 4),
                "avg_frame_time_ms": round(avg_inference, 2),
                "avg_fps": round(avg_fps, 2),
                "duration_s": round(end_process_time - start_process_time, 2),
            }
            print(json.dumps(summary_stats))
    except Exception as e:
        print(e)
        sys.exit(1)
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        if 'out' in locals() and out is not None:
            out.release()
        cv2.destroyAllWindows()