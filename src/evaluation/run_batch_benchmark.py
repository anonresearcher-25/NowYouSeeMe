import subprocess
import json
import sys
import csv
from statistics import mean, stdev
import os

# Only modify these
input_folder = "input/videos"
output_folder = f"output/videos/test1"
frame_skip = 0
mse_thresh = 85
mse_on = False
tracking_on = False
tracking_skip = 0
csv_filename = f"results.csv"
detector_type = "scrfd" ## or "yunet"

NUM_RUNS = 31
SCRIPT_PATH = "blur_batch.py"

FIELDS_TO_AVERAGE = [
    "frames",
    "detector_frames",
    "landmarker_frames",
    "tracked_frames",
    "blurred_frames",
    "frames_mse_thresh_hit",
    "avg_detector_ms",
    "avg_landmarker_ms",
    "avg_blur_ms",
    "avg_tracking_ms",
    "avg_mse_ms",
    "avg_encryption_ms:",
    "avg_frame_time_ms",
    "avg_fps",
]

os.makedirs(output_folder, exist_ok=True)

# Collect the video files in the input folder
video_files = [f for f in os.listdir(input_folder) if f.endswith(".mov")]

if len(video_files) != NUM_RUNS:
    print(f"Warning: Expected {NUM_RUNS} video files, but found {len(video_files)}.")
    NUM_RUNS = len(video_files)  # Adjust the number of runs based on available videos

results = []

for i, video_file in enumerate(video_files):
    print(f"Run {i + 1}/{NUM_RUNS}...")
    input_video_path = os.path.join(input_folder, video_file)
    output_video_path = os.path.join(output_folder, f"output_{video_file}")

    try:
        process = subprocess.run(
            ["python", SCRIPT_PATH, input_video_path, output_video_path, str(frame_skip), str(mse_thresh), str(int(mse_on)), str(int(tracking_on)), str(tracking_skip), str(detector_type)],
            capture_output=True,
            text=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error in run {i + 1}: {e}")
        continue
    output_lines = process.stdout.strip().splitlines()

    # Try parsing the last line as JSON
    try:
        last_line = output_lines[-1]
        data = json.loads(last_line)
        results.append(data)
    except (json.JSONDecodeError, IndexError):
        print(f"Run {i + 1} failed to produce valid JSON. Skipping.")
        continue

with open(csv_filename, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["run"] + FIELDS_TO_AVERAGE)
    writer.writeheader()
    for i, row in enumerate(results):
        row_copy = {key: row.get(key, "N/A") for key in FIELDS_TO_AVERAGE}
        row_copy["run"] = i + 1
        writer.writerow(row_copy)

print(f"\nResults logged to: {csv_filename}")

# Compute and display stats
print("\n=== Averaged Results ===")
for field in FIELDS_TO_AVERAGE:
    values = [r[field] for r in results if field in r]
    if values:
        avg = mean(values)
        std = stdev(values) if len(values) > 1 else 0
        print(f"{field.replace('_', ' ').title()}: {avg:.2f} ± {std:.2f}")
    else:
        print(f"{field}: N/A")
