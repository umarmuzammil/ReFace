import os
import cv2
import numpy as np
import shutil
from tqdm import tqdm
import cv2.quality as quality


def get_fft_sharpness_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)
    rw, rh = int(cols * 0.1), int(rows * 0.1)
    if rw % 2 != 0:
        rw += 1
    if rh % 2 != 0:
        rh += 1
    mask[crow - rh // 2 : crow + rh // 2, ccol - rw // 2 : ccol + rw // 2] = 0
    return np.sum(np.abs(fshift * mask))


def get_sharpness_score(image, method):
    if image is None:
        return 0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if method == "BRISQUE":
        models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        model_path = os.path.join(models_dir, "brisque_model_live.yml")
        range_path = os.path.join(models_dir, "brisque_range_live.yml")
        brisque = quality.QualityBRISQUE.create(model_path, range_path)
        return brisque.compute(gray)[0]
    if method == "FFT":
        return get_fft_sharpness_score(image)
    if method == "Laplacian":
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    if method == "Sobel":
        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        return np.sqrt(sx**2 + sy**2).mean()
    if method == "High-Pass Filter":
        k = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        return cv2.filter2D(gray, -1, k).mean()
    if method == "Unsharp Mask":
        blur = cv2.GaussianBlur(gray, (0, 0), 3)
        sharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
        return cv2.Laplacian(sharp, cv2.CV_64F).var()
    if method == "Custom Kernel":
        return 0
    return 0


def _compute_ssim(img1, img2):
    """Simple SSIM between two grayscale images."""
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(img1 * img1, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 * img2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2

    num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    return np.mean(num / den)


def _detect_scene_changes(frames_data, threshold=0.6):
    """
    Detect scene changes using SSIM. Returns list of shots.
    Each shot is a list of frame indices (into frames_data).
    """
    if len(frames_data) <= 1:
        return [list(range(len(frames_data)))]

    shots = []
    current_shot = [0]

    for i in range(1, len(frames_data)):
        prev = frames_data[i - 1]["gray"]
        curr = frames_data[i]["gray"]
        ssim = _compute_ssim(prev, curr)
        if ssim < threshold:
            # Scene change detected — start new shot
            shots.append(current_shot)
            current_shot = [i]
        else:
            current_shot.append(i)

    shots.append(current_shot)
    return shots


def _detect_motion_blur(gray, threshold=15.0):
    """Detect if a frame has motion blur via directional gradient ratio."""
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sx ** 2 + sy ** 2).mean()
    # Low gradient = likely blurry or flat
    return mag < threshold


def _collect_frames(utils_instance, source):
    """Collect all frames from segments, returning list of dicts with path, name, gray."""
    frames = []
    if source == "Matched People":
        base_dir = utils_instance.get_detected_figures_dir()
    elif source == "Matched Frames":
        base_dir = utils_instance.get_saved_frames_dir()
    else:
        base_dir = utils_instance.get_frames_dir()

    if not os.path.isdir(base_dir):
        return frames

    for entry in sorted(os.listdir(base_dir)):
        seg = os.path.join(base_dir, entry)
        if not os.path.isdir(seg):
            continue
        files = sorted([f for f in os.listdir(seg) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
        for name in files:
            path = os.path.join(seg, name)
            img = cv2.imread(path)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            frames.append({"path": path, "name": name, "seg": entry, "gray": gray, "img": img})
    return frames


def extract_sharp_frames(utils_instance, method, source, sample_rate):
    """
    Scene-aware sharp frame extraction.
    1. Detect scene changes via SSIM
    2. Within each shot, pick the sharpest frame(s)
    3. Skip frames with motion blur
    """
    frames = _collect_frames(utils_instance, source)
    if not frames:
        print("No frames found.")
        return

    print(f"Collected {len(frames)} frames. Detecting scene changes...")
    shots = _detect_scene_changes(frames)
    print(f"Detected {len(shots)} shots.")

    out_dir = os.path.join(utils_instance.get_output_dir(), f"{source}_{method}")
    os.makedirs(out_dir, exist_ok=True)

    saved = 0
    for shot_frames in shots:
        # Score each frame in the shot
        scored = []
        for idx in shot_frames:
            f = frames[idx]
            # Skip motion-blurred frames
            if _detect_motion_blur(f["gray"]):
                continue
            score = get_sharpness_score(f["img"], method)
            scored.append((score, f))

        if not scored:
            continue

        # BRISQUE: lower is better; others: higher is better
        scored.sort(key=lambda x: x[0], reverse=(method != "BRISQUE"))

        # Take top N from this shot
        for _, f in scored[:sample_rate]:
            shutil.copy2(f["path"], os.path.join(out_dir, f["name"]))
            saved += 1

    print(f"Saved {saved} sharp frames to {out_dir}")
