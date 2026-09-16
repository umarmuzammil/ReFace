import os
import shutil
import cv2
import numpy as np
from tqdm import tqdm
from facekit.pipeline.sharpness import get_sharpness_score


def ahash(image, hash_size=8):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    return (small > small.mean()).astype(np.uint8)


def hamming_distance(h1, h2):
    return int(np.bitwise_xor(h1, h2).sum())


def deduplicate(input_dir, threshold=5, duplicates_dir=None):
    if duplicates_dir is None:
        duplicates_dir = os.path.join(input_dir, "duplicates")
    os.makedirs(duplicates_dir, exist_ok=True)

    entries = [f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    hashes = []
    moved = 0
    for name in tqdm(entries, desc="Deduplicating"):
        path = os.path.join(input_dir, name)
        img = cv2.imread(path)
        if img is None:
            continue
        h = ahash(img)
        is_dup = any(hamming_distance(h, prior) <= threshold for prior, _ in hashes)
        if is_dup:
            shutil.move(path, os.path.join(duplicates_dir, name))
            moved += 1
        else:
            hashes.append((h, path))
    return moved


def rank_by_sharpness(input_dir, method="Laplacian", top_n=100, out_dir=None):
    if out_dir is None:
        out_dir = os.path.join(input_dir, f"sorted_{method}")
    os.makedirs(out_dir, exist_ok=True)

    entries = [f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    scored = []
    for name in tqdm(entries, desc=f"Scoring ({method})"):
        path = os.path.join(input_dir, name)
        img = cv2.imread(path)
        if img is None:
            continue
        scored.append((get_sharpness_score(img, method), path, name))

    scored.sort(key=lambda x: x[0], reverse=(method != "BRISQUE"))
    keep = scored[:top_n]
    kept_dir = os.path.join(out_dir, "top")
    os.makedirs(kept_dir, exist_ok=True)
    for _, path, name in keep:
        shutil.copy2(path, os.path.join(kept_dir, name))
    return kept_dir
