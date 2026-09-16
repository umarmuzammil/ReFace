import os
import subprocess
import time
import platform
import cv2


def get_video_duration(video_path):
    command = ["ffmpeg", "-i", video_path, "-hide_banner"]
    result = subprocess.run(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    for line in result.stderr.split("\n"):
        if "Duration:" in line:
            t = line.split("Duration:")[1].split(",")[0].strip()
            h, m, s = t.split(":")
            return float(h) * 3600 + float(m) * 60 + float(s)
    return None


def detect_hardware_acceleration():
    try:
        r = subprocess.run(["nvidia-smi"], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        if r.returncode == 0:
            return "cuda"
    except FileNotFoundError:
        pass
    return "none"


def get_video_fps(video_path):
    cmd = [
        "ffprobe", "-v", "0", "-of", "csv=p=0",
        "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate",
        video_path,
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    num, den = out.split("/")
    return float(num) / float(den)


def crop_image_with_opencv(image_path, crop_box, output_path):
    img = cv2.imread(image_path)
    if img is None:
        return False
    h, w = img.shape[:2]
    x1, y1, x2, y2 = crop_box
    x1c, x2c = max(0, x1), min(w, x2)
    y1c, y2c = max(0, y1), min(h, y2)
    if x1c >= x2c or y1c >= y2c:
        cv2.imwrite(output_path, img)
        return True
    cv2.imwrite(output_path, img[y1c:y2c, x1c:x2c])
    return True


def extract_frames_segment(video_path, output_folder, start_time, duration, segment_id,
                           video_fps, hw_accel="none", rescale_factor=2, quality=8, sample_rate=1):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    seg_dir = os.path.join(output_folder, f"{video_name}_seg{segment_id:03d}")
    os.makedirs(seg_dir, exist_ok=True)

    frame_dur = 1 / video_fps
    adj_start = start_time + frame_dur if segment_id > 0 else start_time

    cmd = ["ffmpeg", "-ss", str(adj_start), "-t", str(duration), "-i", video_path]
    if hw_accel == "cuda":
        cmd[1:1] = ["-hwaccel", "cuda"]

    vf = f"scale=iw/{rescale_factor}:ih/{rescale_factor}"
    if sample_rate > 1:
        vf = f"select='not(mod(n\\,{sample_rate}))',setpts=N/FRAME_RATE/TB,{vf}"

    cmd.extend(["-vf", vf, "-vsync", "0", "-q:v", str(quality),
                os.path.join(seg_dir, "temp_%06d.jpg")])

    start = time.time()
    subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)

    global_start = int(adj_start * video_fps / sample_rate)
    frames = sorted([f for f in os.listdir(seg_dir) if f.endswith(".jpg")])
    for i, f in enumerate(frames):
        os.rename(
            os.path.join(seg_dir, f),
            os.path.join(seg_dir, f"{video_name}_{global_start + i:06d}.jpg"),
        )
    return seg_dir, len(frames)


def extract_frames_target(video_path, output_folder, frame_data, segment_start_time,
                          segment_duration, segment_id, video_fps, hw_accel="none", sample_rate=1,
                          rescale_factor=2, quality=8, crop=False, extract_at_original_scale=False):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    seg_dir = os.path.join(output_folder, f"{video_name}_seg{segment_id:03d}")
    os.makedirs(seg_dir, exist_ok=True)

    seg_end = segment_start_time + segment_duration
    to_extract = []
    for fd in frame_data:
        fn = int(fd["frame"])
        ft = fn / video_fps
        if segment_start_time <= ft < seg_end:
            to_extract.append({"frame_num": fn, "box": fd.get("box") if crop else None})

    if not to_extract:
        return seg_dir, 0

    to_extract.sort(key=lambda x: x["frame_num"])
    indices = [int((f["frame_num"] / video_fps - segment_start_time) * video_fps) for f in to_extract]
    sel = "+".join([f"eq(n\\,{i})" for i in indices])

    cmd = ["ffmpeg", "-ss", str(segment_start_time), "-t", str(segment_duration), "-i", video_path]
    if hw_accel == "cuda":
        cmd[1:1] = ["-hwaccel", "cuda"]

    vf = f"select='{sel}',scale=iw/{rescale_factor}:ih/{rescale_factor}"
    cmd.extend(["-vf", vf, "-vsync", "0", "-q:v", str(quality),
                os.path.join(seg_dir, f"{video_name}_%06d.jpg")])

    result = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    if result.returncode != 0:
        return seg_dir, 0

    extracted = sorted([f for f in os.listdir(seg_dir) if f.endswith(".jpg")])
    ok = 0
    for i, fname in enumerate(extracted):
        if i >= len(to_extract):
            break
        info = to_extract[i]
        cur = os.path.join(seg_dir, fname)
        final = os.path.join(seg_dir, f"{video_name}_{info['frame_num']:06d}.jpg")
        if crop and info["box"]:
            b = info["box"]
            m = 20
            sc = rescale_factor if extract_at_original_scale else 1
            box = [int(b[0] * sc - m), int(b[1] * sc - m), int(b[2] * sc + m), int(b[3] * sc + m)]
            if crop_image_with_opencv(cur, box, final):
                os.remove(cur)
                ok += 1
            else:
                os.rename(cur, final)
                ok += 1
        else:
            os.rename(cur, final)
            ok += 1
    return seg_dir, ok
