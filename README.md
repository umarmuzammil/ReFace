# Re:Face

A PySide6 desktop application for video face detection, matching, and sharp frame extraction.

## Features

- **Video Playback** — scrub, play/pause, and mark frames directly in the app
- **Face Marking** — detect and crop faces from any video frame with InsightFace
- **Face Matching** — search all extracted frames for faces matching your marked references
- **Person Detection** — YOLO-based person cropping with configurable padding
- **Sharpness Filtering** — scene-aware sharp frame extraction with multiple methods (Laplacian, BRISQUE, FFT, etc.)
- **Dedup & Rank** — deduplicate and rank frames by sharpness
- **Dark macOS Theme** — VSCode-inspired dark UI with Activity Bar navigation

## Workflow

1. **Open Video** — File > Open Video (Ctrl+O)
2. **Mark Faces** — scrub to a clear face frame, click Mark (bookmark icon)
3. **Extract Frames** — click Extract Frames in the sidebar (parallel FFmpeg extraction)
4. **Process Marked Faces** — search all frames for matching faces
5. **Export** — save matched frames or detected people

## Project Structure

```
video_project/
├── extracted_frames/    # FFmpeg-extracted frames in segment subdirs
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
