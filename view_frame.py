import cv2
import matplotlib.pyplot as plt

def view_nth_frame_opencv(video_path, n, show_image=True, save_path=None):
    """
    Extract and view the nth frame from a video using OpenCV
    
    Args:
        video_path (str): Path to the video file
        n (int): Frame number to extract (0-indexed)
        show_image (bool): Whether to display the image
        save_path (str): Optional path to save the frame
    
    Returns:
        numpy.ndarray: The extracted frame
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video")
        return None
    
    # Get total frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if n >= total_frames:
        print(f"Error: Frame {n} is out of range. Video has {total_frames} frames.")
        cap.release()
        return None
    
    # Set the frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, n)
    
    # Read the frame
    ret, frame = cap.read()
    
    if ret:
        # Convert BGR to RGB for matplotlib
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if show_image:
            plt.figure(figsize=(10, 6))
            plt.imshow(frame_rgb)
            plt.title(f"Frame {n}")
            plt.axis('off')
            plt.show()
        
        if save_path:
            cv2.imwrite(save_path, frame)
            print(f"Frame saved to {save_path}")
        
        cap.release()
        return frame_rgb
    else:
        print("Error: Could not read frame")
        cap.release()
        return None

# Example usage
video_path = "output/blurred.mp4"  # Replace with your video path
frame_number = 6  # The frame you want to view

frame = view_nth_frame_opencv(video_path, frame_number)