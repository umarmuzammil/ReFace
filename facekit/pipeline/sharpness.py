import os
import re
import json
import cv2
import numpy as np
import concurrent.futures
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
    if len(frames_data) <= 1:
        return [list(range(len(frames_data)))]

    shots = []
    current_shot = [0]

    for i in range(1, len(frames_data)):
        prev = frames_data[i - 1]["gray"]
        curr = frames_data[i]["gray"]
        ssim = _compute_ssim(prev, curr)
        if ssim < threshold:
            shots.append(current_shot)
            current_shot = [i]
        else:
            current_shot.append(i)

    shots.append(current_shot)
    return shots


def _detect_motion_blur(gray, threshold=15.0):
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sx ** 2 + sy ** 2).mean()
    return mag < threshold


def _collect_frames(utils_instance, source):
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


def _parse_frame_number(filename):
    m = re.search(r"_(\d+)\.", filename)
    return int(m.group(1)) if m else None


def _detect_rescale_factor(utils_instance):
    """Auto-detect rescale factor by comparing extracted_frames to original video."""
    frames_dir = utils_instance.get_frames_dir()
    if not os.path.isdir(frames_dir):
        return 2

    for entry in sorted(os.listdir(frames_dir)):
        seg = os.path.join(frames_dir, entry)
        if not os.path.isdir(seg):
            continue
        for name in os.listdir(seg):
            if name.lower().endswith((".jpg", ".png")):
                img = cv2.imread(os.path.join(seg, name))
                if img is None:
                    continue
                fw, fh = img.shape[1], img.shape[0]

                from facekit.pipeline.frames_extract import get_video_duration, get_video_fps
                import subprocess
                vp = utils_instance.get_file_path()
                cmd = ["ffprobe", "-v", "0", "-select_streams", "v:0",
                       "-show_entries", "stream=width,height", "-of", "csv=p=0", vp]
                try:
                    out = subprocess.check_output(cmd, text=True).strip()
                    vw, vh = [int(x) for x in out.split(",")]
                    rf = max(1, round(vw / fw))
                    print(f"Auto-detected rescale factor: {rf} (video {vw}x{vh} -> frames {fw}x{fh})")
                    return rf
                except Exception:
                    return 2
    return 2


def _load_face_data(utils_instance):
    jp = os.path.join(utils_instance.get_project_dir(), "face_data.json")
    if not os.path.isfile(jp):
        return {}
    with open(jp) as f:
        data = json.load(f)
    by_frame = {}
    for entry in data:
        fn = int(entry["frame"])
        if fn not in by_frame:
            by_frame[fn] = entry
    return by_frame


def _load_people_data(utils_instance):
    people_dir = utils_instance.get_detected_figures_dir()
    jp = os.path.join(people_dir, "people_data.json")
    if not os.path.isfile(jp):
        return {}
    with open(jp) as f:
        data = json.load(f)
    by_frame = {}
    for entry in data:
        fn = int(entry.get("frame", 0))
        if fn not in by_frame:
            by_frame[fn] = entry
    return by_frame


def extract_sharp_frames(utils_instance, method, source, sample_rate,
                         crop_faces=False, crop_people=False):
    frames = _collect_frames(utils_instance, source)
    if not frames:
        print("No frames found.")
        return

    print(f"Collected {len(frames)} low-res frames. Scoring sharpness...")

    scored_all = []
    for f in frames:
        if _detect_motion_blur(f["gray"]):
            continue
        score = get_sharpness_score(f["img"], method)
        fn = _parse_frame_number(f["name"])
        if fn is not None:
            scored_all.append((score, fn, f["name"], f["seg"]))

    if not scored_all:
        print("No sharp frames selected.")
        return

    scored_all.sort(key=lambda x: x[0], reverse=(method != "BRISQUE"))

    if source in ("Matched Frames", "Matched People"):
        target_count = min(len(scored_all), sample_rate * 30)
    else:
        from facekit.pipeline.frames_extract import get_video_duration
        dur = get_video_duration(utils_instance.get_file_path())
        target_count = min(len(scored_all), sample_rate * max(1, int(dur)))

    selected = []
    for _, fn, name, seg in scored_all[:target_count]:
        selected.append({"frame": fn, "name": name, "seg": seg})

    if not selected:
        print("No sharp frames selected.")
        return

    print(f"Selected {len(selected)} sharp frames. Extracting at full resolution...")

    video_path = utils_instance.get_file_path()
    from facekit.pipeline.frames_extract import (
        detect_hardware_acceleration, get_video_duration, get_video_fps,
        extract_frames_target,
    )

    hw = detect_hardware_acceleration()
    dur = get_video_duration(video_path)
    fps = get_video_fps(video_path)
    nd = 8
    sd = dur / nd

    rescale = _detect_rescale_factor(utils_instance)

    face_data = _load_face_data(utils_instance) if (crop_faces or crop_people) else {}
    people_data = _load_people_data(utils_instance) if crop_people else {}

    if crop_people and not people_data:
        print("Warning: people_data.json not found. Re-run YOLO detection to generate people bounding boxes. Falling back to face data.")
    elif crop_people and people_data:
        print(f"Loaded {len(people_data)} people detections from people_data.json.")

    do_crop = crop_faces or crop_people
    frame_data = []
    for s in selected:
        entry = {"frame": str(s["frame"])}
        fn = s["frame"]

        if do_crop:
            box = None
            if crop_people and fn in people_data:
                box = people_data[fn].get("box")
            elif crop_faces and fn in face_data:
                box = face_data[fn]["box"]
            elif fn in face_data:
                box = face_data[fn]["box"]

            if box:
                entry["box"] = [int(box[0] * rescale), int(box[1] * rescale),
                                int(box[2] * rescale), int(box[3] * rescale)]

        frame_data.append(entry)

    out_dir = os.path.join(utils_instance.get_output_dir(), f"{source}_{method}")
    os.makedirs(out_dir, exist_ok=True)

    segment_data = [[] for _ in range(nd)]
    for entry in frame_data:
        fn = int(entry["frame"])
        ft = fn / fps
        seg = min(int(ft / sd), nd - 1)
        segment_data[seg].append(entry)

    total = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=nd) as pool:
        futs = [
            pool.submit(
                extract_frames_target, video_path, out_dir, segment_data[i],
                i * sd, sd, i, fps, hw, 1, 1, 28,
                do_crop, True,
            )
            for i in range(nd) if segment_data[i]
        ]
        for f in concurrent.futures.as_completed(futs):
            try:
                _, c = f.result()
                total += c
            except Exception as e:
                print(f"Error: {e}")

    print(f"Saved {total} sharp frames at full resolution to {out_dir}")
