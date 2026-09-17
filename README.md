# Re:Face

Automatically extract the sharpest frames from any video — no more manual scrolling.

## What it does

Scrolling through a video to find and extract sharp, usable frames is tedious. Re:Face automates this: it analyzes every frame, detects scene changes, and pulls out the sharpest ones. You can optionally filter by face or person to extract only the frames that matter.

## Features

- **Auto Sharp Frame Extraction** — scene-aware filtering picks the sharpest frame from each shot automatically
- **Face Filtering** — mark reference faces, then extract only frames containing matching faces
- **Person Filtering** — YOLO-based person detection to extract only frames with people
- **Multiple Sharpness Methods** — Laplacian, FFT, BRISQUE, Sobel, High-Pass, Unsharp Mask
- **Parallel FFmpeg Extraction** — multi-segment extraction for fast frame pulling
- **Face Matching** — search all frames for faces matching your marked references (with SSIM optimization)
- **Dark macOS Theme** — VSCode-inspired UI with Activity Bar navigation

## Quick Start

```
1. Open Video         (Ctrl+O)
2. Extract Frames     (sidebar → Extract → Extract Frames)
3. Mark Faces         (scrub to a face, click the bookmark icon)
4. Process Faces      (sidebar → Faces → Process Marked Faces)
5. Export             (save matched frames or sharp frames)
```

## Project Structure

```
video_project/
├── extracted_frames/    # All frames pulled from video
├── marked_faces/        # User-marked face crops + embeddings
├── matched_frames/      # Frames where face matches were found
├── detected_people/     # YOLO person crops
├── sharp_frames/        # Sharpness-filtered output
└── face_data.json       # Match results (frame numbers, bounding boxes)
```

## Requirements

- Python 3.10+
- FFmpeg (must be in PATH)
- NVIDIA GPU optional (auto-detects CUDA, falls back to CPU)

## Install

```bash
conda create -n face python=3.11
conda activate face
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Dependencies

| Package | Purpose |
|---------|---------|
| PySide6 | Qt6 GUI framework |
| opencv-python | Image/video processing |
| numpy | Array operations |
| insightface | Face detection + recognition |
| onnxruntime | Model inference (CUDA/CPU) |
| scikit-learn | Cosine similarity for matching |
| tqdm | Progress display in terminal |

## Author

**Umar Muzammil**

## License

MIT
