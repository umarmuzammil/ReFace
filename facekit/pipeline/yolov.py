import os
import re
import json
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from tqdm import tqdm

from .utils import ProjectUtils

torch.serialization.add_safe_globals([])

_model = None
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "yolov8n.pt")


def _get_model():
    global _model
    if _model is None:
        _model = YOLO(_MODEL_PATH)
    return _model


def crop_and_save(video_path, padding_factor=0.12, jpeg_quality=90):
    u = ProjectUtils(video_path)
    model = _get_model()
    people_data = []

    for entry in sorted(os.listdir(u.get_frames_dir())):
        seg = os.path.join(u.get_frames_dir(), entry)
        if not os.path.isdir(seg):
            continue
        out = os.path.join(u.get_detected_figures_dir(), entry)
        os.makedirs(out, exist_ok=True)

        for fname in tqdm(sorted(os.listdir(seg)), desc=entry):
            if not fname.lower().endswith((".jpg", ".png")):
                continue
            img = cv2.imread(os.path.join(seg, fname))
            if img is None:
                continue
            results = model(os.path.join(seg, fname), classes=[0], verbose=False)
            if not any(len(r.boxes) > 0 for r in results):
                continue

            boxes = []
            for r in results:
                for b in r.boxes:
                    boxes.append(b.xyxy[0].cpu().numpy())
            boxes = np.array(boxes)
            x1, y1 = np.min(boxes[:, 0]), np.min(boxes[:, 1])
            x2, y2 = np.max(boxes[:, 2]), np.max(boxes[:, 3])

            m = re.search(r"_(\d+)\.", fname)
            frame_no = int(m.group(1)) if m else 0
            people_data.append({
                "frame": str(frame_no).zfill(6),
                "box": [int(x1), int(y1), int(x2), int(y2)],
            })

            px = int((x2 - x1) * padding_factor)
            py = int((y2 - y1) * padding_factor)
            crop = img[
                max(0, int(y1 - py)) : min(img.shape[0], int(y2 + py)),
                max(0, int(x1 - px)) : min(img.shape[1], int(x2 + px)),
            ]
            if crop.size > 0:
                cv2.imwrite(os.path.join(out, fname), crop, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])

    if people_data:
        jp = os.path.join(u.get_detected_figures_dir(), "people_data.json")
        with open(jp, "w") as f:
            json.dump(people_data, f, indent=4)
        print(f"Saved {len(people_data)} people detections to {jp}")
