# A state-of-the-art lightweight Face Blurring Pipeline in Python

This project provides tools for applying face blurring to videos and evaluating the results.

## Setup

1. **Clone the repository** (if you haven't already):

   ```bash
   git clone <repository-url>
   cd Face-Blurring
   ```

2. **Install dependencies** using the provided `requirements.txt` file:

   ```bash
   pip install -r requirements.txt
   ```

3. **Install ffmpeg** using the link provided in the `requirements.txt` file:

## Running the Main Blurring Script

The main script for blurring videos is `main.py`.

### Usage

```bash
python main.py
```

### Changing Input and Output Video Paths

To specify which video to blur and where to save the output, edit the `Config` class in `main.py`. Locate the following lines:

```python
class Config:
    input_video_path = "path/to/input/video.mp4"
    output_video_path = "path/to/output/video.mp4"
    # ...other config options...
```

- Set `input_video_path` to the path of your input video file.
- Set `output_video_path` to the desired path for the blurred output video.

Save your changes and re-run `main.py`.

## Evaluation

To evaluate the blurring results, use the `run_batch_benchmark.py` script:

```bash
python run_batch_benchmark.py
```

This script will process the output videos and provide evaluation metrics.

## Synthetic Replacement

Warp a video with a synthetic face using `warping.py`, then optionally run **MobileFaceSwap** on the warped result all from one CLI.

### Prerequisites
- Python 3.10–3.12 in one environment.
- Install base deps: `pip install opencv-python numpy`.
- Place **MobileFaceSwap** next to `main.py`:
  - `MobileFaceSwap/video_test.py`
  - `MobileFaceSwap/checkpoints/arcface.pdparams` (and any other weights that repo requires)
- Place assets next to `main.py`:
  - `synthetic_faces/` (face images, e.g., `female1.jpg`)
  - `synthetic_landmarks/` (matching `.txt` files with the same stem, e.g., `female1.txt`)
  - `video_colors.json` mapping video basename → RGB  
    Example:
    ```json
    { "blurred": "(29,25,24)" }
    ```

### Expected landmarks layout (for the input video)
````

\<landmarks\_root>/
\<video\_basename>/
face\_0/
frame\_0000.txt
frame\_0001.txt
face\_1/
...

````

### Run (single video)
```bash
python main.py \
  --video "/absolute/path/to/your_video.mp4" \
  --landmarks_root "/absolute/path/to/video_landmarks" \
  --after_swap
````

> You do **not** need to pass `--synth_img_folder`, `--synth_lm_folder`, `--output_root`, or `--mfs_script`; the script uses local folders next to `main.py`.

### Outputs

* `<repo_dir>/<video_name>.mp4` — warped video
* `<repo_dir>/best_match/` — chosen synthetic (image + landmarks + manifest)
* MobileFaceSwap’s swapped result will also be written under `<repo_dir>` (naming per that script).


---

For further customization or troubleshooting, refer to comments within each script.
