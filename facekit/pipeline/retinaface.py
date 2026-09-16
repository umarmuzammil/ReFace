import os
import json
import numpy as np
import cv2 as cv
import insightface
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


class ModelLoader:
    def __init__(self):
        self.app = None

    def load_model(self):
        if self.app is not None:
            return self.app

        # Try CUDA first, fallback to CPU
        for providers in [
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
            ["CPUExecutionProvider"],
        ]:
            try:
                self.app = insightface.app.FaceAnalysis(
                    name="buffalo_l",
                    allowed_modules=["detection", "recognition"],
                    providers=providers,
                )
                self.app.prepare(ctx_id=0, det_size=(640, 640))
                active = self.app.models["recognition"].session.get_providers()
                print(f"Model loaded. Providers: {active}")
                return self.app
            except Exception as e:
                print(f"Failed with {providers[0]}: {e}")
                self.app = None

        print("Error: Could not load InsightFace model with any provider.")
        return None

    def get_app(self):
        return self.app


class FaceProcessor:
    def __init__(self, app, utils_instance):
        self.app = app
        self.utils_instance = utils_instance

    def extract_faces(self, image_path, output_dir="output_faces", save_embedding=False, export=True):
        img = cv.imread(image_path)
        if img is None:
            return None, []
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        faces = self.app.get(img)
        os.makedirs(output_dir, exist_ok=True)

        for i, face in enumerate(faces):
            x1, y1, x2, y2 = face.bbox.astype(int)
            cv.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.putText(img_rgb, str(i), (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        if save_embedding and faces:
            img_id = os.path.splitext(os.path.basename(image_path))[0]
            self.save_embeddings(faces, img_id, output_dir)

        if export:
            fname = os.path.splitext(os.path.basename(image_path))[0]
            save_path = os.path.join(output_dir, f"{fname}_faces.png")
            cv.imwrite(save_path, cv.cvtColor(img_rgb, cv.COLOR_RGB2BGR))

        return img_rgb, faces

    def save_embeddings(self, faces, img_id, output_dir="embeddings"):
        embeddings = [face.embedding for face in faces]
        labels = [face.labels for face in faces]
        path = os.path.join(output_dir, f"{img_id}_embedding.npz")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, embeddings=np.array(embeddings), labels=np.array(labels))

    def load_embedding(self, embedding_path):
        if not os.path.exists(embedding_path):
            print(f"Embedding not found: {embedding_path}")
            return None, None
        data = np.load(embedding_path, allow_pickle=True)
        return data["embeddings"], data["labels"]

    def compare_face_embedding(self, embeddings, progress_callback=None):
        matched = []
        frames_dir = self.utils_instance.get_frames_dir()
        seg_dirs = sorted([
            os.path.join(frames_dir, d)
            for d in os.listdir(frames_dir)
            if os.path.isdir(os.path.join(frames_dir, d))
        ]) if os.path.isdir(frames_dir) else []

        # Count total frames for progress
        total = sum(len([f for f in os.listdir(s) if f.endswith(".jpg")]) for s in seg_dirs if os.path.isdir(s))
        done = 0

        global_frame = 0
        for seg_dir in seg_dirs:
            files = sorted([f for f in os.listdir(seg_dir) if f.endswith(".jpg")])
            for fname in files:
                img = cv.imread(os.path.join(seg_dir, fname))
                if img is None:
                    done += 1
                    if progress_callback and done % 10 == 0:
                        progress_callback(done / total if total else 1, f"Scanning frame {done}/{total}")
                    continue
                faces = self.app.get(img)
                for face in faces:
                    for emb in embeddings:
                        sim = cosine_similarity(emb.reshape(1, -1), face.embedding.reshape(1, -1))
                        if sim >= 0.4:
                            matched.append({
                                "image_path": os.path.join(seg_dir, fname),
                                "box": face.bbox.astype(int).tolist(),
                                "score": float(face.det_score),
                                "frame": str(global_frame).zfill(6),
                            })
                done += 1
                if progress_callback and done % 5 == 0:
                    progress_callback(done / total if total else 1, f"Scanning frame {done}/{total} — {len(matched)} match(es)")
                global_frame += 1

        out = os.path.join(self.utils_instance.get_project_dir(), "face_data.json")
        with open(out, "w") as f:
            json.dump(matched, f, indent=4)
        print(f"Saved {len(matched)} matched frames to {out}")
        return matched
